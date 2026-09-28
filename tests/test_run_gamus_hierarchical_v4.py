import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_gamus_hierarchical_v4.ps1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_pins_every_executable_training_input() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    pinned = {
        "expectedComparatorConfigSha256": ROOT
        / "configs"
        / "multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml",
        "expectedBuilderSha256": ROOT
        / "scripts"
        / "build_gamus_hierarchical_v4_config.py",
        "expectedPreflightSha256": ROOT
        / "scripts"
        / "preflight_gamus_hierarchical_v4.py",
        "expectedTrainerSha256": ROOT / "scripts" / "train_multidomain.py",
        "expectedModelCodeSha256": ROOT
        / "src"
        / "msr"
        / "models"
        / "domain_surface_net.py",
        "expectedLossCodeSha256": ROOT
        / "src"
        / "msr"
        / "training"
        / "losses.py",
        "expectedStateAuditorSha256": ROOT
        / "scripts"
        / "audit_gamus_hierarchical_v4_checkpoint.py",
        "expectedHeightAuditorSha256": ROOT
        / "scripts"
        / "audit_height_output_identity.py",
        "expectedPairedReplayEvaluatorSha256": ROOT
        / "scripts"
        / "evaluate_gamus_dcphl_paired_replay.py",
        "expectedRuntimeDataSealerSha256": ROOT
        / "scripts"
        / "seal_gamus_v4_runtime_data.py",
    }
    for variable, path in pinned.items():
        match = re.search(rf'\${variable} = "([0-9a-f]{{64}})"', source)
        assert match is not None, variable
        assert match.group(1) == _sha(path), variable


def test_runner_requires_reviewed_baseline_and_two_step_config_seal() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "ApprovedBaselineAuditSha256" in source
    assert "ApprovedBaselineReplaySha256" in source
    assert "ApprovedV4ConfigSha256" in source
    assert "ApprovedRuntimeDataSealSha256" in source
    assert "SealConfigOnly" in source
    assert "hash printed by a reviewed -SealConfigOnly run" in source
    assert source.index("build_gamus_hierarchical_v4_config.py") < source.index(
        "train_multidomain.py"
    )
    assert '"--candidate-config", $config' in source
    assert (
        '"--expected-candidate-config-sha256", $ApprovedV4ConfigSha256'
        in source
    )
    assert '"--fixed-v3-independent-replay", $baselineReplay' in source
    assert '"--paired-independent-replay", $pairedReplayOutput' in source
    assert '"--expected-paired-independent-replay-sha256", $pairedReplaySha256' in source
    assert '"--expected-seal-sha256", $ApprovedRuntimeDataSealSha256' in source
    assert "refusing to overwrite" in source


def test_runner_proves_height_identity_before_freezing_candidate() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    height_call = source.index('"--protected-checkpoint", $protected')
    replay_call = source.index('"--v3-checkpoint", $comparatorCheckpoint')
    state_call = source.index('"--height-output-identity", $heightOutput')
    eligibility_check = source.index(
        "fully_eligible_for_locked_classifier_excluded_city_evaluation"
    )
    freeze_write = source.index("candidate_freeze_manifest.json")
    assert height_call < replay_call < state_call < eligibility_check < freeze_write


def test_runner_keeps_final_class_gates_out_of_training_selection() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "checkpoint_best_diagnostic.pt" not in source
    assert "checkpoint_best_landscape.pt" in source
    assert "failed at least one paired gate" in source
    assert "NYC stays closed and no candidate is frozen" in source


def test_runner_never_evaluates_nyc_reuses_official_test_or_promotes() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "evaluate_gamus_locked_nyc.py" not in source
    assert "--evaluate-locked" not in source
    assert "select_showcase_checkpoint" not in source
    assert 'nyc_evaluated = $false' in source
    assert 'official_test_used_by_v4 = $false' in source
    assert (
        'official_test_global_status = "previously_consumed_forbidden_for_reuse"'
        in source
    )
    assert 'promotion_performed = $false' in source
    assert "VERIFIED live pointer and protected checkpoint are unchanged" in source
