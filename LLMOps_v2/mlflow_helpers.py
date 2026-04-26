"""
llmops_core/mlflow_helpers.py

Centralised MLflow logging helpers.
Every phase imports from here instead of scattering mlflow.log_* calls
throughout the pipeline notebooks.

All functions accept explicit arguments rather than reading globals so they
are easy to unit-test and reuse across projects.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow


# ── Experiment run logging (Phase 1) ─────────────────────────────────────────

def log_experiment_run(
    run                : mlflow.ActiveRun,
    template_params    : dict,
    metrics            : dict,
    turn_records       : list[dict],
    template_dict      : dict,
    run_name           : str,
    run_judge          : bool,
) -> None:
    """Log params, metrics, and artifacts for one Phase 1 experiment run."""
    mlflow.log_params(template_params)
    mlflow.set_tags({
        "template.name"   : template_params.get("template.name", ""),
        "template.version": template_params.get("template.version", ""),
        "eval_set_size"   : str(metrics.get("eval_set_size", 0)),
        "judge_enabled"   : str(run_judge),
        "run_date"        : datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    artifact_data = {
        "run_id"      : run.info.run_id,
        "run_name"    : run_name,
        "template"    : template_dict,
        "timestamp"   : datetime.now(timezone.utc).isoformat(),
        "turn_records": turn_records,
    }
    _write_and_log_artifact(
        data          = artifact_data,
        filename      = f"{run_name}_results.json",
        artifact_path = "eval_results",
    )
    _write_and_log_artifact(
        data          = template_dict,
        filename      = f"{run_name}_template.json",
        artifact_path = "prompt_config",
    )


# ── Registration run logging (Phase 1 → registry) ────────────────────────────

def log_registration_run(
    run            : mlflow.ActiveRun,
    template_params: dict,
    composite_score: float,
    rule_pass_rate : float,
    cfg_path       : str | Path,
) -> None:
    """Log params and the prompt_config artifact for the registration run."""
    mlflow.log_params(template_params)
    mlflow.log_metrics({
        "composite_score": round(composite_score, 3),
        "rule_pass_rate" : round(rule_pass_rate,  3),
    })
    mlflow.set_tags({
        "stage"                : "candidate",
        "template.name"        : template_params.get("template.name", ""),
        "template.config_hash" : template_params.get("template.config_hash", ""),
    })
    mlflow.log_artifact(str(cfg_path), artifact_path="prompt_config")


# ── CI run logging (Phase 2) ──────────────────────────────────────────────────

def log_ci_run(
    run               : mlflow.ActiveRun,
    run_name          : str,
    model_name        : str,
    candidate_version : str,
    candidate_run_id  : str,
    template_cfg      : dict,
    eval_results      : list[dict],
    gates             : list[Any],
    metrics           : dict,
    git_sha           : str,
    git_ref           : str,
    git_pr            : str,
    all_gates_passed  : bool,
    n_eval_cases      : int,
) -> None:
    """Log everything for one Phase 2 CI run."""
    mlflow.log_params({
        "candidate_model_name"   : model_name,
        "candidate_model_version": candidate_version,
        "candidate_run_id"       : candidate_run_id,
        "template_name"          : template_cfg.get("name", ""),
        "template_version"       : template_cfg.get("version", ""),
        "model"                  : template_cfg.get("model", ""),
        "eval_set_size"          : n_eval_cases,
    })
    mlflow.log_metrics(metrics)
    mlflow.set_tags({
        "ci.result"              : "PASS" if all_gates_passed else "FAIL",
        "ci.git_sha"             : git_sha,
        "ci.git_ref"             : git_ref,
        "ci.git_pr"              : git_pr or "none",
        "ci.run_date"            : datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "candidate.model_name"   : model_name,
        "candidate.model_version": candidate_version,
    })

    artifact = {
        "ci_run_id"   : run.info.run_id,
        "ci_run_name" : run_name,
        "git_sha"     : git_sha,
        "git_ref"     : git_ref,
        "all_passed"  : all_gates_passed,
        "gates"       : [
            {
                "name"     : g.name,
                "passed"   : g.passed,
                "actual"   : g.actual,
                "threshold": g.threshold,
                "direction": g.direction,
            }
            for g in gates
        ],
        "eval_results": eval_results,
        "timestamp"   : datetime.now(timezone.utc).isoformat(),
    }
    _write_and_log_artifact(artifact, f"{run_name}_ci_results.json", "ci_results")


# ── CD staging run logging (Phase 3) ─────────────────────────────────────────

def log_cd_staging_run(
    run               : mlflow.ActiveRun,
    run_name          : str,
    model_name        : str,
    candidate_version : str,
    template_cfg      : dict,
    smoke_results     : list[dict],
    smoke_passed      : bool,
    git_sha           : str,
    git_ref           : str,
) -> None:
    """Log everything for one Phase 3 staging run."""
    n = len(smoke_results)
    mlflow.log_params({
        "model_name"      : model_name,
        "model_version"   : candidate_version,
        "template_name"   : template_cfg.get("name", ""),
        "template_version": template_cfg.get("version", ""),
        "model"           : template_cfg.get("model", ""),
        "deployment_stage": "staging",
    })
    mlflow.log_metrics({
        "cd/smoke_pass_rate": sum(r["passed"] for r in smoke_results) / n,
        "cd/avg_latency_ms" : sum(r["latency_ms"] for r in smoke_results) / n,
        "cd/all_passed"     : int(smoke_passed),
    })
    mlflow.set_tags({
        "cd.stage"         : "staging",
        "cd.result"        : "PASS" if smoke_passed else "FAIL",
        "cd.git_sha"       : git_sha,
        "cd.git_ref"       : git_ref,
        "cd.run_date"      : datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "candidate.version": candidate_version,
    })
    artifact = {
        "cd_run_id"    : run.info.run_id,
        "stage"        : "staging",
        "passed"       : smoke_passed,
        "git_sha"      : git_sha,
        "smoke_results": smoke_results,
        "timestamp"    : datetime.now(timezone.utc).isoformat(),
    }
    _write_and_log_artifact(artifact, f"{run_name}.json", "cd_results")


# ── CD production run logging (Phase 3) ──────────────────────────────────────

def log_cd_production_run(
    run                      : mlflow.ActiveRun,
    run_name                 : str,
    model_name               : str,
    candidate_version        : str,
    prev_production_version  : str | None,
    template_cfg             : dict,
    approved_by              : str,
    git_sha                  : str,
    git_ref                  : str,
    deployed_at              : str,
) -> None:
    """Log an immutable audit record for a Phase 3 production deployment."""
    mlflow.log_params({
        "model_name"              : model_name,
        "model_version"           : candidate_version,
        "prev_production_version" : prev_production_version or "none",
        "template_name"           : template_cfg.get("name", ""),
        "template_version"        : template_cfg.get("version", ""),
        "model"                   : template_cfg.get("model", ""),
        "deployment_stage"        : "production",
    })
    mlflow.log_metrics({"cd/promoted": 1})
    mlflow.set_tags({
        "cd.stage"        : "production",
        "cd.result"       : "deployed",
        "cd.approved_by"  : approved_by,
        "cd.git_sha"      : git_sha,
        "cd.git_ref"      : git_ref,
        "cd.deployed_at"  : deployed_at,
        "cd.prev_version" : prev_production_version or "none",
        "candidate.version": candidate_version,
    })
    audit = {
        "event"                  : "production_deployment",
        "cd_run_id"              : run.info.run_id,
        "model_name"             : model_name,
        "model_version"          : candidate_version,
        "prev_production_version": prev_production_version,
        "template_name"          : template_cfg.get("name", ""),
        "template_version"       : template_cfg.get("version", ""),
        "llm_model"              : template_cfg.get("model", ""),
        "approved_by"            : approved_by,
        "git_sha"                : git_sha,
        "git_ref"                : git_ref,
        "deployed_at"            : deployed_at,
    }
    _write_and_log_artifact(audit, f"{run_name}_audit.json", "cd_audit")


# ── Champion/challenger run logging (Phase 4) ─────────────────────────────────

def log_cc_run(
    run               : mlflow.ActiveRun,
    run_name          : str,
    model_name        : str,
    champ_version     : str,
    chall_version     : str,
    champ_cfg         : dict,
    chall_cfg         : dict,
    champ_results     : dict,
    chall_results     : dict,
    score_delta       : float,
    promote           : bool,
    min_improvement   : float,
    git_sha           : str,
    evaluated_at      : str,
    n_eval_cases      : int,
) -> None:
    """Log everything for one Phase 4 champion/challenger comparison run."""
    mlflow.log_params({
        "model_name"          : model_name,
        "champion_version"    : champ_version,
        "challenger_version"  : chall_version,
        "champion_template"   : champ_cfg.get("name", ""),
        "challenger_template" : chall_cfg.get("name", ""),
        "champion_model"      : champ_cfg.get("model", ""),
        "challenger_model"    : chall_cfg.get("model", ""),
        "min_improvement"     : min_improvement,
        "eval_set_size"       : n_eval_cases,
    })
    mlflow.log_metrics({
        "champion/composite"       : champ_results["composite"],
        "champion/rule_pass_rate"  : champ_results["rule_pass_rate"],
        "champion/avg_rule_score"  : champ_results["avg_rule_score"],
        "champion/avg_judge_score" : champ_results["avg_judge_score"],
        "champion/avg_latency_ms"  : champ_results["avg_latency_ms"],
        "champion/avg_cost_usd"    : champ_results["avg_cost_usd"],
        "challenger/composite"     : chall_results["composite"],
        "challenger/rule_pass_rate": chall_results["rule_pass_rate"],
        "challenger/avg_rule_score": chall_results["avg_rule_score"],
        "challenger/avg_judge_score":chall_results["avg_judge_score"],
        "challenger/avg_latency_ms": chall_results["avg_latency_ms"],
        "challenger/avg_cost_usd"  : chall_results["avg_cost_usd"],
        "delta/composite"          : round(score_delta, 4),
        "decision/promoted"        : int(promote),
    })
    decision = "challenger_promoted" if promote else "champion_retained"
    mlflow.set_tags({
        "cc.result"            : decision,
        "cc.champion_version"  : champ_version,
        "cc.challenger_version": chall_version,
        "cc.score_delta"       : str(round(score_delta, 4)),
        "cc.min_improvement"   : str(min_improvement),
        "cc.git_sha"           : git_sha,
        "cc.evaluated_at"      : evaluated_at,
    })

    comparison = {
        "run_id"         : run.info.run_id,
        "decision"       : decision,
        "promoted"       : promote,
        "score_delta"    : score_delta,
        "min_improvement": min_improvement,
        "evaluated_at"   : evaluated_at,
        "git_sha"        : git_sha,
        "champion"       : {"version": champ_version, **champ_results},
        "challenger"     : {"version": chall_version, **chall_results},
    }
    _write_and_log_artifact(comparison, f"{run_name}_comparison.json", "cc_results")


# ── Internal helper ───────────────────────────────────────────────────────────

def _write_and_log_artifact(
    data          : dict,
    filename      : str,
    artifact_path : str,
) -> None:
    """Write a dict to /tmp as JSON and log it as an MLflow artifact."""
    p = Path("/tmp") / filename
    p.write_text(json.dumps(data, indent=2))
    mlflow.log_artifact(str(p), artifact_path=artifact_path)
