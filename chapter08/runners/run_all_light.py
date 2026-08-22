"""Run Chapter 8 lightweight checks that do not download large models."""
from __future__ import annotations

import csv
from pathlib import Path

from _common.dataset_registry import list_assets
from _common.tables import write_rows
from checks.artifact_health import artifact_health_failures, write_artifact_health
from checks.chapter_audit import required_failures, write_chapter_audit
from checks.data_profile import write_dataset_profile
from checks.decision_consistency import decision_consistency_failures, write_decision_consistency
from checks.environment_profile import environment_profile_failures, write_environment_profile
from checks.external_setup import write_external_setup_matrix
from checks.implementation_decisions import implementation_decision_failures, write_implementation_decisions
from checks.model_coverage import write_model_coverage
from checks.model_reference_audit import model_reference_failures, write_model_reference_audit
from checks.notebook_matrix import write_notebook_matrix
from checks.phase_gates import phase_gate_failures, write_phase_gate_summary
from checks.readiness_report import readiness_local_failures, write_readiness_report
from evaluation.comparison_table import write_model_comparison_from_outputs
from finetune.ace_step_lora_train import write_lora_train_plan
from finetune.musicgen_finetune import write_musicgen_finetune_plan
from model_runners.ace_step import ACEStepRunner
from model_runners.audioldm2 import AudioLDM2Runner
from model_runners.base import RunnerStatus, write_runner_status_table
from model_runners.conditioning import write_conditioning_table
from model_runners.musicgen import MusicGenRunner
from model_runners.stable_audio_open import StableAudioOpenRunner
from model_runners.yue import YuERunner
from runners.run_pretrained_available import (
    REQUEST_PLAN_FIELDS,
    RUN_PLAN_FIELDS,
    build_request_plan_rows,
    build_run_plan_rows,
)
from runners.training_smoke import training_smoke_failures, write_training_smoke


def main() -> None:
    table_dir = Path("outputs/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    write_rows(table_dir / "08_dataset_assets.csv", list_assets())
    environment_rows = write_environment_profile(table_dir / "08_environment_profile.csv")
    write_dataset_profile(table_dir / "08_dataset_profile.csv")
    decision_ledger_rows = write_implementation_decisions(table_dir / "08_implementation_decisions.csv", chapter_root=Path("."))
    external_rows = write_external_setup_matrix(table_dir / "08_external_setup_matrix.csv")
    runners = [
        MusicGenRunner(),
        AudioLDM2Runner(),
        StableAudioOpenRunner(),
        YuERunner(),
        ACEStepRunner(),
    ]
    statuses = [runner.check_environment() for runner in runners]
    write_runner_status_table(
        table_dir / "08_model_runner_status.csv",
        statuses,
    )
    write_conditioning_table(table_dir / "08_pretrained_conditioning_matrix.csv")
    write_pretrained_plan_tables(table_dir, statuses, notebook_root=Path.cwd())
    coverage_rows = write_model_coverage(
        table_dir / "08_model_coverage.csv",
        chapter_root=Path("."),
        runner_status_rows=[status.as_row() for status in statuses],
    )
    reference_rows = write_model_reference_audit(
        table_dir / "08_model_reference_audit.csv",
        coverage_rows=coverage_rows,
    )
    write_notebook_matrix(
        table_dir / "08_notebook_matrix.csv",
        chapter_root=Path("."),
        runner_status_rows=[status.as_row() for status in statuses],
    )
    write_model_comparison_from_outputs(Path("."))
    write_lora_train_plan(table_dir / "08_8_lora_plan.csv", chapter_root=Path("."))
    write_musicgen_finetune_plan(table_dir / "08_8_musicgen_finetune_plan.csv", chapter_root=Path("."))
    smoke_rows = write_training_smoke(table_dir / "08_training_smoke.csv", chapter_root=Path("."))
    write_artifact_health(table_dir / "08_artifact_health.csv")
    decision_rows = write_decision_consistency(table_dir / "08_decision_consistency.csv")
    # Audit/health/readiness tables are self-referential, so seed them before final checks.
    write_readiness_report(
        table_dir / "08_readiness_report.csv",
        phase_rows=[],
        external_setup_rows=external_rows,
    )
    audit_rows = write_chapter_audit(table_dir / "08_chapter_audit.csv")
    phase_rows = write_phase_gate_summary(
        table_dir / "08_phase_gate_summary.csv",
        audit_rows=audit_rows,
        runner_status_rows=[status.as_row() for status in statuses],
        chapter_root=Path("."),
    )
    readiness_rows = write_readiness_report(
        table_dir / "08_readiness_report.csv",
        phase_rows=phase_rows,
        external_setup_rows=external_rows,
        notebook_status_rows=load_notebook_run_status_rows(table_dir),
    )
    audit_rows = write_chapter_audit(table_dir / "08_chapter_audit.csv")
    health_rows = write_artifact_health(table_dir / "08_artifact_health.csv")
    failures = required_failures(audit_rows)
    if failures:
        details = "\n".join(f"- {row['category']}:{row['item']} -> {row['status']}" for row in failures)
        raise RuntimeError("Chapter 8 local audit failed:\n" + details)
    gate_failures = phase_gate_failures(phase_rows)
    if gate_failures:
        details = "\n".join(f"- Phase {row['phase_id']} {row['phase_name']} -> {row['missing_required']}" for row in gate_failures)
        raise RuntimeError("Chapter 8 phase gates failed:\n" + details)
    readiness_failures = readiness_local_failures(readiness_rows)
    if readiness_failures:
        details = "\n".join(f"- {row['check_id']} -> {row['detail']}" for row in readiness_failures)
        raise RuntimeError("Chapter 8 readiness report found local failures:\n" + details)
    health_failures = artifact_health_failures(health_rows)
    if health_failures:
        details = "\n".join(f"- {row['artifact_type']}:{row['path']} -> {row['status']}" for row in health_failures)
        raise RuntimeError("Chapter 8 artifact health failed:\n" + details)
    consistency_failures = decision_consistency_failures(decision_rows)
    if consistency_failures:
        details = "\n".join(f"- {row['check_id']} -> {row['matches']}" for row in consistency_failures)
        raise RuntimeError("Chapter 8 decision consistency failed:\n" + details)
    reference_failures = model_reference_failures(reference_rows)
    if reference_failures:
        details = "\n".join(f"- {row['reference_id']} -> {row['detail']}" for row in reference_failures)
        raise RuntimeError("Chapter 8 model reference audit failed:\n" + details)
    decision_ledger_failures = implementation_decision_failures(decision_ledger_rows)
    if decision_ledger_failures:
        details = "\n".join(f"- {row['decision_id']} -> {row['detail']}" for row in decision_ledger_failures)
        raise RuntimeError("Chapter 8 implementation decisions failed:\n" + details)
    smoke_failures = training_smoke_failures(smoke_rows)
    if smoke_failures:
        details = "\n".join(f"- {row['check_id']} -> {row['status']} ({row['detail']})" for row in smoke_failures)
        raise RuntimeError("Chapter 8 training smoke failed:\n" + details)
    env_failures = environment_profile_failures(environment_rows)
    if env_failures:
        details = "\n".join(f"- {row['component']} -> {row['status']}" for row in env_failures)
        raise RuntimeError("Chapter 8 environment profile failed:\n" + details)
    print(f"Wrote lightweight Chapter 8 tables to {table_dir}")
    print(f"Wrote {len(phase_rows)} phase gate rows")
    print(f"Wrote {len(readiness_rows)} readiness rows")


def write_pretrained_plan_tables(
    table_dir: Path,
    statuses: list[RunnerStatus],
    notebook_root: Path,
) -> dict[str, list[dict[str, object]]]:
    """Write pretrained run/request plans from the same status snapshot."""
    run_rows = build_run_plan_rows(statuses, notebook_root=notebook_root)
    request_rows = build_request_plan_rows(statuses, notebook_root=notebook_root)
    write_rows(table_dir / "08_pretrained_run_plan.csv", run_rows, fieldnames=RUN_PLAN_FIELDS)
    write_rows(
        table_dir / "08_pretrained_request_plan.csv",
        request_rows,
        fieldnames=REQUEST_PLAN_FIELDS,
    )
    return {"run": run_rows, "request": request_rows}


def load_notebook_run_status_rows(table_dir: Path) -> list[dict[str, object]]:
    """Load notebook runner status rows when the runner tables already exist."""
    rows: list[dict[str, object]] = []
    for name in ("08_core_notebook_run_status.csv", "08_light_notebook_run_status.csv"):
        path = table_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


if __name__ == "__main__":
    main()
