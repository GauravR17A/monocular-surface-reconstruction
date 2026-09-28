"""Deterministic source-ratio sampling for mixed GAMUS/legacy replay."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


class SourceTaggedDataset(Dataset[dict[str, Any]]):
    """Add an explicit source label without changing an underlying dataset."""

    def __init__(self, dataset: Dataset, source: str) -> None:
        normalized_source = source.strip().lower()
        if not normalized_source:
            raise ValueError("source must not be empty")
        self.dataset = dataset
        self.source = normalized_source

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("SourceTaggedDataset requires mapping samples")
        if "source" in sample:
            raise ValueError("Underlying dataset already defines a 'source' field")
        return {**sample, "source": self.source}


def assert_matching_sample_contract(*datasets: Dataset) -> frozenset[str]:
    """Fail early when datasets cannot share PyTorch's default collator.

    Only the sample schema is compared here. Tensor spatial sizes are allowed to
    differ in general, but mixed replay constructs both children with the same
    patch size before calling this helper.
    """

    if len(datasets) < 2:
        raise ValueError("At least two datasets are required for contract validation")
    contracts: list[dict[str, tuple[object, ...]]] = []
    for dataset in datasets:
        if len(dataset) <= 0:
            raise ValueError("Mixed-replay datasets must not be empty")
        sample = dataset[0]
        if not isinstance(sample, Mapping):
            raise TypeError("Mixed-replay datasets must return mapping samples")
        contract: dict[str, tuple[object, ...]] = {}
        for name, value in sample.items():
            if isinstance(value, torch.Tensor):
                contract[str(name)] = (
                    "tensor",
                    value.dtype,
                    value.ndim,
                    tuple(value.shape),
                )
            else:
                contract[str(name)] = ("value", type(value))
        contracts.append(contract)
    reference = contracts[0]
    for index, contract in enumerate(contracts[1:], start=1):
        if contract != reference:
            missing = sorted(set(reference) - set(contract))
            extra = sorted(set(contract) - set(reference))
            mismatched = sorted(
                name
                for name in set(reference) & set(contract)
                if reference[name] != contract[name]
            )
            raise ValueError(
                "Mixed-replay sample contracts differ for dataset "
                f"{index}: missing={missing}, extra={extra}, "
                f"mismatched={mismatched}"
            )
    return frozenset(reference)


class SourceRatioSampler(Sampler[int]):
    """Yield exact, deterministic GAMUS/legacy quotas in every batch.

    Indices ``[0, gamus_size)`` address the first child of a ``ConcatDataset``;
    legacy indices are offset by ``gamus_size``. Each source is consumed through
    independently shuffled, no-replacement cycles. ``set_epoch`` makes the full
    sequence reproducible across fresh runs and checkpoint resumes.
    """

    def __init__(
        self,
        gamus_size: int,
        legacy_size: int,
        *,
        num_samples: int,
        batch_size: int,
        gamus_fraction: float,
        seed: int = 0,
        legacy_landscapes: Sequence[str] | None = None,
        balance_legacy_landscapes: bool = False,
    ) -> None:
        if gamus_size <= 0 or legacy_size <= 0:
            raise ValueError("gamus_size and legacy_size must be positive")
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        if num_samples % batch_size:
            raise ValueError(
                "num_samples must be divisible by batch_size for exact source quotas"
            )
        if not 0.0 < gamus_fraction < 1.0:
            raise ValueError("gamus_fraction must be strictly between zero and one")
        gamus_per_batch_float = gamus_fraction * batch_size
        gamus_per_batch = round(gamus_per_batch_float)
        if abs(gamus_per_batch_float - gamus_per_batch) > 1.0e-9:
            raise ValueError(
                "gamus_fraction * batch_size must be an integer for an exact "
                "per-batch source ratio"
            )
        if gamus_per_batch <= 0 or gamus_per_batch >= batch_size:
            raise ValueError("Every mixed-replay batch must contain both sources")
        if legacy_landscapes is not None and len(legacy_landscapes) != legacy_size:
            raise ValueError("legacy_landscapes must contain one label per legacy item")
        if balance_legacy_landscapes and legacy_landscapes is None:
            raise ValueError(
                "balance_legacy_landscapes requires legacy_landscapes"
            )

        grouped_legacy_indices: dict[str, list[int]] = defaultdict(list)
        if legacy_landscapes is not None:
            for index, landscape in enumerate(legacy_landscapes):
                normalized = str(landscape).strip().lower()
                if not normalized:
                    raise ValueError("Legacy landscape labels must not be empty")
                grouped_legacy_indices[normalized].append(index)
        if balance_legacy_landscapes and len(grouped_legacy_indices) < 2:
            raise ValueError(
                "Balanced legacy sampling requires at least two landscapes"
            )

        self.gamus_size = int(gamus_size)
        self.legacy_size = int(legacy_size)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.gamus_fraction = float(gamus_fraction)
        self.gamus_per_batch = int(gamus_per_batch)
        self.legacy_per_batch = self.batch_size - self.gamus_per_batch
        self.seed = int(seed)
        self.balance_legacy_landscapes = bool(balance_legacy_landscapes)
        self._legacy_groups = {
            name: tuple(indices)
            for name, indices in sorted(grouped_legacy_indices.items())
        }
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    @property
    def source_counts(self) -> dict[str, int]:
        batches = self.num_samples // self.batch_size
        return {
            "gamus": batches * self.gamus_per_batch,
            "legacy": batches * self.legacy_per_batch,
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    @staticmethod
    def _draw_shuffled_cycles(
        population: Sequence[int], count: int, generator: torch.Generator
    ) -> list[int]:
        if not population:
            raise ValueError("Cannot sample from an empty population")
        draws: list[int] = []
        while len(draws) < count:
            permutation = torch.randperm(len(population), generator=generator).tolist()
            remaining = count - len(draws)
            draws.extend(population[index] for index in permutation[:remaining])
        return draws

    def _balanced_legacy_draws(
        self, count: int, generator: torch.Generator
    ) -> list[int]:
        names = tuple(self._legacy_groups)
        quotient, remainder = divmod(count, len(names))
        # Rotate the unavoidable remainder so repeated epochs do not privilege
        # the alphabetically first landscape.
        remainder_start = self.epoch % len(names)
        group_counts = {name: quotient for name in names}
        for offset in range(remainder):
            group_counts[names[(remainder_start + offset) % len(names)]] += 1

        draws_by_group = {
            name: iter(
                self._draw_shuffled_cycles(
                    self._legacy_groups[name], group_counts[name], generator
                )
            )
            for name in names
        }
        schedule = [
            name for name in names for _ in range(group_counts[name])
        ]
        order = torch.randperm(len(schedule), generator=generator).tolist()
        return [next(draws_by_group[schedule[index]]) for index in order]

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        counts = self.source_counts
        gamus_draws = self._draw_shuffled_cycles(
            tuple(range(self.gamus_size)), counts["gamus"], generator
        )
        if self.balance_legacy_landscapes:
            legacy_draws = self._balanced_legacy_draws(
                counts["legacy"], generator
            )
        else:
            legacy_draws = self._draw_shuffled_cycles(
                tuple(range(self.legacy_size)), counts["legacy"], generator
            )

        batches = self.num_samples // self.batch_size
        for batch_index in range(batches):
            gamus_start = batch_index * self.gamus_per_batch
            legacy_start = batch_index * self.legacy_per_batch
            batch = gamus_draws[
                gamus_start : gamus_start + self.gamus_per_batch
            ]
            batch.extend(
                self.gamus_size + index
                for index in legacy_draws[
                    legacy_start : legacy_start + self.legacy_per_batch
                ]
            )
            # Randomize source positions while preserving each source stream's
            # no-replacement cycle order.
            source_order = torch.randperm(
                self.batch_size, generator=generator
            ).tolist()
            gamus_batch = iter(batch[: self.gamus_per_batch])
            legacy_batch = iter(batch[self.gamus_per_batch :])
            for source_index in source_order:
                yield (
                    next(gamus_batch)
                    if source_index < self.gamus_per_batch
                    else next(legacy_batch)
                )
