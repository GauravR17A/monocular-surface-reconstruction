"""Paired crop-sampling continuation; fixed V1 starting weights, no app mutation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

import train_rgb_segmenter as legacy
from evaluate_rgb_segmenter_checkpoint import (
    KNOWN_EXPERIMENT, KNOWN_CHECKPOINT_SHA256, load_known_checkpoint,
    validate_sealed_run, verify_file_identities,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msr.data.gamus_rgb_segmentation import (
    GamusRgbSegmentationDataset, build_train_sampler, source_metadata_snapshot,
    authenticate_validation_content, canonical_sha256,
)
from msr.models.rgb_segmenter import RgbSegformer
from msr.evaluation.rgb_segmentation import evaluate_rgb_segmentation

ACTIVE = ROOT / "outputs/orchestration/rgb_sampling_v2_active.json"


def checked_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or legacy.sha256(path) != expected:
        raise RuntimeError(f"Sealed file identity mismatch: {path}")


def read_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["schema"] != "msr.rgb_sampling_paired.v2" or config["evaluation"]["app_promotion"] is not False:
        raise ValueError("Only the isolated paired protocol is permitted")
    settings = config["training"]
    if settings["epochs"] != 4 or settings["batch_size"] != 4 or settings["samples_per_epoch"] != 3837 or settings["num_workers"] != 2 or settings["precision"] != "bf16" or settings["early_stopping"] is not False:
        raise ValueError("Fixed four-epoch equal-budget protocol required")
    if config["sampling"]["control_probability"] != 0 or config["sampling"]["targeted_probability"] != .5:
        raise ValueError("Only the predeclared crop-selection contrast is allowed")
    if config["initial_checkpoint_sha256"] != KNOWN_CHECKPOINT_SHA256 or legacy.resolve(config["initial_checkpoint"]).resolve() != (KNOWN_EXPERIMENT / "checkpoint_best_guarded.pt").resolve():
        raise ValueError("Both arms must start from the protected V1 candidate")
    if config["evaluation"]["external_data_policy"] != "forbidden_in_this_experiment" or config["evaluation"]["primary_comparison_epoch"] != settings["epochs"]:
        raise ValueError("External reuse or unequal comparison budget is prohibited")
    return config


def sources():
    names = ["scripts/train_rgb_sampling_v2.py", "scripts/evaluate_rgb_segmenter_checkpoint.py",
             "src/msr/data/gamus_rgb_targeted_v2.py", "src/msr/evaluation/rgb_sampling_v2.py",
             "scripts/audit_rgb_training_crops_v2.py"]
    return {**legacy.source_manifest(), **{name: legacy.sha256(name) for name in names}}


def build_data(config, base, probability):
    from msr.data.gamus_rgb_targeted_v2 import GamusRgbTargetedCropDatasetV2
    data = base["data"]
    kwargs = {"approved_index_path": legacy.resolve(data["approved_index_path"]),
              "approved_index_sha256": data["approved_index_file_sha256"],
              "rgb_scale": data["rgb_scale"], "dark_pixel_threshold": data["dark_pixel_threshold"]}
    train = GamusRgbTargetedCropDatasetV2(data["root"], "train", patch_size=384, random_crop=True, augment=True,
                targeted_index_path=legacy.resolve(config["sampling"]["index_path"]),
                targeted_index_sha256=config["sampling"]["index_sha256"], target_probability=probability, **kwargs)
    val = GamusRgbSegmentationDataset(data["root"], "val", patch_size=1024, **kwargs)
    return train, val


def assert_binding(pair, config_path, binding, train, val):
    checked_hash(config_path, binding["config_sha256"])
    checked_hash(pair / "config.yaml", binding["config_sha256"])
    if sources() != binding["sources"]:
        raise RuntimeError("Sealed paired-run source changed")
    for name, expected in binding["sources"].items():
        checked_hash(pair / "source_snapshot" / name, expected)
    for item in binding["inputs"]:
        checked_hash(legacy.resolve(item["path"]), item["sha256"])
    verify_file_identities(binding["v1_identities"])
    if {"train": source_metadata_snapshot(train), "val": source_metadata_snapshot(val)} != binding["data_metadata"]:
        raise RuntimeError("Training or validation source metadata changed")


def epoch_indices(dataset, base, seed, epoch):
    # Independent from loader/model/augmentation RNG. Identical for both arms.
    generator = torch.Generator().manual_seed(seed+epoch*1009)
    return list(build_train_sampler(dataset, base, generator))


def fresh_model(payload):
    model = RgbSegformer.from_architecture(payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.cuda()


def criterion_for(base):
    train = base["training"]
    return legacy.SixClassFocalLoss(train["fine_semantic_class_weights"], train["fine_semantic_focal_gamma"]).cuda()


def training_loss(model, batch, criterion, device="cuda"):
    # The sealed legacy precision helper accepts torch.device, not a string.
    return legacy.loss_for(model, batch, criterion, torch.device(device), "bf16")


def make_optimizer(model, settings):
    return torch.optim.AdamW(model.parameter_groups(settings["encoder_learning_rate"], settings["decoder_learning_rate"], settings["weight_decay"]))


def proof(config, base, payload, datasets, val, pair, binding):
    results = {}
    for arm, dataset in datasets.items():
        legacy.seed_everything(config["training"]["seed"])
        model = fresh_model(payload)
        optimizer = make_optimizer(model, config["training"])
        loss_function = criterion_for(base)
        indices = epoch_indices(dataset, base, config["training"]["seed"], 1)[:4]
        batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=4, num_workers=0)))
        model.eval()
        with torch.no_grad():
            initial = float(training_loss(model, batch, loss_function))
        max_gradient = 0.
        for _ in range(config["proof"]["steps_per_arm"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, batch, loss_function)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training-only proof loss")
            loss.backward()
            gradient = sum(float(p.grad.float().square().sum()) for p in model.network.segformer.parameters() if p.grad is not None)**.5
            max_gradient = max(gradient, max_gradient)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = float(training_loss(model, batch, loss_function))
        reduction = (initial-final)/max(initial, 1e-12)
        passed = bool(math.isfinite(final) and reduction >= config["proof"]["minimum_loss_reduction_fraction"] and max_gradient > 0)
        results[arm] = {"initial_loss": initial, "final_loss": final, "reduction": reduction, "encoder_gradient_norm": max_gradient,
                        "steps": config["proof"]["steps_per_arm"], "sample_ids": list(batch["sample_id"]), "passes": passed}
        del model, optimizer, loss_function
        torch.cuda.empty_cache()
        if not passed:
            legacy.atomic_json(pair / "proof.json", {"passes": False, "arms": results})
            raise RuntimeError(f"{arm} train-only proof failed; no full training")
    legacy.seed_everything(config["training"]["seed"])
    model = fresh_model(payload)
    indices = [next(i for i,r in enumerate(val.records) if r.city == city) for city in ("DC", "PHL")]
    torch.cuda.reset_peak_memory_stats()
    smoke = evaluate_rgb_segmentation(model, DataLoader(Subset(val, indices), batch_size=1), "cuda",
                expected_ids=[val.records[i].sample_id for i in indices], precision="bf16", native_size=1024)
    legacy.atomic_json(pair / "proof.json", {"passes": True, "arms": results, "native_smoke": smoke,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "proof_weights_discarded": True,
                        "binding_sha256": canonical_sha256(binding), "initial_checkpoint_sha256": KNOWN_CHECKPOINT_SHA256})
    del model
    torch.cuda.empty_cache()


def write_checkpoint(directory, name, payload):
    path = directory / name
    if name.startswith("checkpoint_epoch_") and path.exists():
        # A crash before commit can leave an orphan; retain it before retrying.
        shutil.copy2(path, directory / f"uncommitted_{legacy.run_id()}_{name}")
    legacy.atomic_checkpoint(path, payload)
    legacy.atomic_json(path.with_suffix(".sha256.json"), {"sha256": legacy.sha256(path)})


def resume_payload(directory, binding, arm):
    commit_path = directory / "commit.json"
    if not commit_path.exists():
        return None
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    name = commit["checkpoint"]
    if Path(name).name != name or name != f"checkpoint_epoch_{commit['epoch']:03d}.pt":
        raise RuntimeError("Invalid committed checkpoint path")
    path = directory / name
    checked_hash(path, commit["sha256"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("binding") != binding or payload.get("arm") != arm or payload.get("promotion_eligible") is not False:
        raise RuntimeError("Incompatible paired checkpoint")
    if payload["epoch"] != commit["epoch"] or [row["epoch"] for row in payload["history"]] != list(range(1, payload["epoch"]+1)):
        raise RuntimeError("Inconsistent saved epoch history")
    # Epoch file + atomic commit are authoritative; aliases are repairable.
    shutil.copy2(path, directory / "checkpoint_latest.pt")
    legacy.atomic_json((directory / "checkpoint_latest.pt").with_suffix(".sha256.json"), {"sha256": commit["sha256"]})
    if payload["best_safe"] is not None:
        best_path = directory / f"checkpoint_epoch_{payload['best_safe']['epoch']:03d}.pt"
        checked_hash(best_path, commit["best_safe_sha256"])
        shutil.copy2(best_path, directory / "checkpoint_best_safe.pt")
        legacy.atomic_json((directory / "checkpoint_best_safe.pt").with_suffix(".sha256.json"), {"sha256": commit["best_safe_sha256"]})
    return payload


def train_arm(config, base, initial, pair, config_path, binding, arm, dataset, val, reference, control_history):
    from msr.evaluation.rgb_sampling_v2 import compare_epoch
    directory = pair / arm
    directory.mkdir(exist_ok=True)
    saved = resume_payload(directory, binding, arm)
    settings = config["training"]
    legacy.seed_everything(settings["seed"])
    model = fresh_model(initial)
    optimizer = make_optimizer(model, settings)
    total_steps = settings["epochs"]*math.ceil(len(dataset)/settings["batch_size"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: legacy.learning_rate_factor(
        step, total_steps, settings["warmup_optimizer_steps"], settings["minimum_learning_rate_factor"]))
    loss_function = criterion_for(base)
    history = []
    best_safe = None
    if saved:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        # Per-epoch sampler/loader generators are reconstructed independently.
        legacy.restore_rng(saved["rng_state"], torch.Generator())
        history, best_safe = saved["history"], saved["best_safe"]
        legacy.write_history(directory / "metrics.jsonl", history)
    for epoch in range(len(history)+1, settings["epochs"]+1):
        assert_binding(pair, config_path, binding, dataset, val)
        indices = epoch_indices(dataset, base, settings["seed"], epoch)
        plan_hash = canonical_sha256(indices)
        if arm == "targeted" and control_history[epoch-1]["sample_order_sha256"] != plan_hash:
            raise RuntimeError("Paired arms have different training tile draws")
        loader = DataLoader(dataset, sampler=indices, batch_size=4, num_workers=settings["num_workers"],
                    worker_init_fn=legacy.seed_worker, generator=torch.Generator().manual_seed(settings["seed"]+epoch*2003), pin_memory=True)
        losses = []
        started = time.monotonic()
        model.train()
        for batch_index, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, batch, loss_function)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                legacy.status(pair, "training", arm=arm, epoch=epoch, total_epochs=4, batch=batch_index, total_batches=len(loader), loss=losses[-1])
            if batch_index % 200 == 0:
                print(f"{arm} epoch {epoch}/4 | {batch_index}/{len(loader)} | loss {losses[-1]:.4f}", flush=True)
        def progress(update):
            done = update["completed_batches"]
            if done == 1 or done % 20 == 0 or done == update["total_batches"]:
                legacy.status(pair, "validation", arm=arm, epoch=epoch, completed_batches=done, total_batches=update["total_batches"])
        val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=settings["num_workers"], pin_memory=True,
                    worker_init_fn=legacy.seed_worker, generator=torch.Generator().manual_seed(settings["seed"]+epoch*3001))
        evaluation = evaluate_rgb_segmentation(model, val_loader, "cuda", expected_ids=list(val.sample_ids),
                    precision="bf16", native_size=1024, progress_callback=progress)
        gate_config = {**config["evaluation"], "comparison_epoch": epoch, "control_epoch": epoch,
                       "primary_epoch": 4, "comparison_kind": "primary" if epoch == 4 else "exploratory"}
        control = control_history[epoch-1]["validation"] if arm == "targeted" else None
        gate = compare_epoch(evaluation, reference, gate_config, control_eval=control)
        score = evaluation["overall"]["six_class_identification"]["macro_f1"]
        row = {"epoch": epoch, "arm": arm, "validation": evaluation, "gate": gate,
               "sample_order_sha256": plan_hash, "optimizer_steps": scheduler.last_epoch,
               "training_batches": len(loader), "train_loss": float(np.mean(losses)),
               "elapsed_seconds": time.monotonic()-started, "updated_at_utc": legacy.utc_now(), "promotion_eligible": False}
        history.append(row)
        safe = gate["safety_vs_v1"]["passes"]
        improved = safe and (best_safe is None or score > best_safe["macro_f1"])
        if improved:
            best_safe = {"epoch": epoch, "macro_f1": score}
        assert_binding(pair, config_path, binding, dataset, val)
        checkpoint = {"model_type": model.model_type, "architecture": model.architecture, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                      "rng_state": legacy.rng_state(torch.Generator()), "binding": binding, "arm": arm, "epoch": epoch,
                      "history": history, "best_safe": best_safe, "promotion_eligible": False,
                      "input_contract": "raw_rgb_01_imagenet_normalized_in_model"}
        write_checkpoint(directory, f"checkpoint_epoch_{epoch:03d}.pt", checkpoint)
        epoch_path = directory / f"checkpoint_epoch_{epoch:03d}.pt"
        legacy.atomic_json(directory / "commit.json", {"epoch": epoch, "checkpoint": epoch_path.name,
                           "sha256": legacy.sha256(epoch_path),
                           "best_safe_sha256": legacy.sha256(directory / f"checkpoint_epoch_{best_safe['epoch']:03d}.pt") if best_safe else None})
        write_checkpoint(directory, "checkpoint_latest.pt", checkpoint)
        if improved:
            write_checkpoint(directory, "checkpoint_best_safe.pt", checkpoint)
        legacy.write_history(directory / "metrics.jsonl", history)
        legacy.status(pair, "epoch_complete", arm=arm, epoch=epoch, macro_f1=score, guards_pass=safe)
        print(f"{arm} epoch {epoch} | F1 {score:.4f} | DC lowveg {evaluation['by_city']['DC']['six_class_identification']['per_class']['low_vegetation']['f1']:.4f} | safety {'PASS' if safe else 'FAIL'}", flush=True)
    del model, optimizer, loss_function
    torch.cuda.empty_cache()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rgb_sampling_v2.yaml")
    parser.add_argument("--resume-pair", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = legacy.resolve(args.config).resolve()
    config = read_config(config_path)
    base_path = legacy.resolve(config["base_config"])
    base = legacy.load_config(base_path)
    # Retain original sampler, weights, masks and validation; equal LR reset in both arms.
    base["training"]["samples_per_epoch"] = config["training"]["samples_per_epoch"]
    checked_hash(config["fixed_v1_replay"], config["fixed_v1_replay_sha256"])
    checked_hash(config["sampling"]["index_path"], config["sampling"]["index_sha256"])
    replay = json.loads(legacy.resolve(config["fixed_v1_replay"]).read_text(encoding="utf-8"))
    if replay.get("replay_verified") is not True or replay["checkpoint"]["sha256"] != KNOWN_CHECKPOINT_SHA256:
        raise RuntimeError("Require authenticated V1 replay")
    initial = load_known_checkpoint(KNOWN_EXPERIMENT)
    _, _, identities = validate_sealed_run(KNOWN_EXPERIMENT, initial)
    control, val = build_data(config, base, 0.)
    targeted, _ = build_data(config, base, .5)
    datasets = {"control": control, "targeted": targeted}
    auth = authenticate_validation_content(val, legacy.resolve(base["protocol"]["fixed_v3_independent_replay"]),
                base["protocol"]["fixed_v3_independent_replay_sha256"],
                ROOT / "outputs/diagnostics/rgb_sampling_v2_validation_cache.json")
    # Cache-hit counters are operational metadata, not recipe identity.
    auth.pop("newly_hashed_file_count", None)
    binding = {"config_sha256": legacy.sha256(config_path), "sources": sources(), "software": legacy.software_versions(),
               "v1_identities": identities, "validation_identity": auth,
               "inputs": [{"path": config[key], "sha256": config[key+"_sha256"]} for key in ("initial_checkpoint", "fixed_v1_replay")]
                         + [{"path": config["sampling"]["index_path"], "sha256": config["sampling"]["index_sha256"]}],
               "data_metadata": {"train": source_metadata_snapshot(control), "val": source_metadata_snapshot(val)}}
    if args.preflight_only:
        print(json.dumps({"passes": True, "training_images": len(control), "validation_images": len(val), "sources": binding["sources"]}))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 required; no silent CPU fallback")
    torch.set_num_threads(4)
    if args.resume_pair:
        pair = args.resume_pair.resolve()
        if not pair.is_relative_to(ROOT / "experiments") or not pair.name.endswith("_rgb_sampling_v2_pair") or json.loads((pair / "binding.json").read_text()) != binding:
            raise RuntimeError("Incompatible paired run resume")
        saved_proof = json.loads((pair / "proof.json").read_text())
        if not saved_proof["passes"] or saved_proof.get("binding_sha256") != canonical_sha256(binding) or saved_proof.get("initial_checkpoint_sha256") != KNOWN_CHECKPOINT_SHA256:
            raise RuntimeError("Missing successful paired proof")
    else:
        pair = ROOT / "experiments" / f"{legacy.run_id()}_rgb_sampling_v2_pair"
        pair.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, pair / "config.yaml")
        for name in binding["sources"]:
            destination = pair / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        legacy.atomic_json(pair / "binding.json", binding)
    legacy.atomic_json(ACTIVE, {"pair_dir": str(pair), "pid": os.getpid(), "started_at_utc": legacy.utc_now()})
    try:
        assert_binding(pair, config_path, binding, control, val)
        if not args.resume_pair:
            legacy.status(pair, "train_only_proofs")
            proof(config, base, initial, datasets, val, pair, binding)
        histories = {}
        for arm, dataset in datasets.items():
            histories[arm] = train_arm(config, base, initial, pair, config_path, binding, arm, dataset, val, replay["evaluation"], histories.get("control", []))
        result = {"stage": "completed", "primary_epoch": 4, "primary_comparison": histories["targeted"][-1]["gate"],
                  "exploratory_best_epoch": {arm: max(rows, key=lambda row: row["validation"]["overall"]["six_class_identification"]["macro_f1"])["epoch"] for arm, rows in histories.items()},
                  "promotion_performed": False, "height_evaluated": False, "external_evaluated": False}
        legacy.atomic_json(pair / "comparison_report.json", result)
        legacy.status(pair, "completed", report="comparison_report.json", app_model_changed=False)
        print(f"Completed paired experiment: {pair}", flush=True)
    except BaseException as error:
        legacy.atomic_json(pair / f"failure_{legacy.run_id()}.json", {"error": repr(error), "updated_at_utc": legacy.utc_now()})
        legacy.status(pair, "failed", error=repr(error), app_model_changed=False)
        raise


if __name__ == "__main__":
    main()
