"""
llmops_core/pipeline_cd.py

Shared Phase 3 CD logic.
Two entry points:
  run_staging(cfg, client, ...)    → smoke-tests the candidate, tags version
  run_production(cfg, client, ...) → promotes candidate alias, writes audit log

Both are called by the thin Databricks notebook wrapper which handles
widget reading and sys.exit().
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from .mlflow_helpers import log_cd_staging_run, log_cd_production_run
from .project_config import ProjectConfig


# ── Smoke probe schema ────────────────────────────────────────────────────────
# Projects supply their own smoke probes in project.yaml under `smoke_probes:`.
# Each probe is a dict with these keys:
#   id               str
#   user_message     str
#   must_contain     list[str]
#   must_not_contain list[str]
#   max_latency_ms   float


@dataclass
class CDResult:
    passed        : bool
    mlflow_run_id : str


# ── Staging ───────────────────────────────────────────────────────────────────

def run_staging(
    cfg               : ProjectConfig,
    client,
    candidate_version : str,
    template_cfg      : dict,
    smoke_probes      : list[dict],
    git_sha           : str = "local",
    git_ref           : str = "local",
) -> CDResult:
    """
    Run smoke probes against the candidate model.
    Tags the model version with cd_staging=pass/fail.
    Returns CDResult — caller decides whether to sys.exit(1).
    """
    print(f"\n{'═'*56}")
    print(f"  Phase 3 — Staging  |  {cfg.mlflow_model_name} v{candidate_version}")
    print(f"{'═'*56}")

    smoke_passed, smoke_results = _run_smoke_probes(client, template_cfg, smoke_probes)

    run_name = f"cd-staging-{git_sha[:8]}-{'pass' if smoke_passed else 'fail'}"
    with mlflow.start_run(run_name=run_name) as cd_run:
        log_cd_staging_run(
            run               = cd_run,
            run_name          = run_name,
            model_name        = cfg.mlflow_model_name,
            candidate_version = candidate_version,
            template_cfg      = template_cfg,
            smoke_results     = smoke_results,
            smoke_passed      = smoke_passed,
            git_sha           = git_sha,
            git_ref           = git_ref,
        )
        run_id = cd_run.info.run_id

    mlflow_client = mlflow.MlflowClient()
    mlflow_client.set_model_version_tag(
        name    = cfg.mlflow_model_name,
        version = candidate_version,
        key     = "cd_staging",
        value   = "pass" if smoke_passed else "fail",
    )

    print(f"\n  MLflow run : {run_id}")
    print(f"  Result     : {'PASS' if smoke_passed else 'FAIL'}")
    return CDResult(passed=smoke_passed, mlflow_run_id=run_id)


# ── Production ────────────────────────────────────────────────────────────────

def run_production(
    cfg               : ProjectConfig,
    candidate_version : str,
    template_cfg      : dict,
    approved_by       : str,
    git_sha           : str = "local",
    git_ref           : str = "local",
) -> CDResult:
    """
    Promote the candidate to production alias.
    Writes an immutable MLflow audit record.
    Returns CDResult — caller decides whether to sys.exit(1).
    """
    print(f"\n{'═'*56}")
    print(f"  Phase 3 — Production  |  {cfg.mlflow_model_name} v{candidate_version}")
    print(f"  Approved by: {approved_by}")
    print(f"{'═'*56}")

    mlflow_client = mlflow.MlflowClient()

    # verify staging gate
    try:
        mv = mlflow_client.get_model_version(cfg.mlflow_model_name, candidate_version)
        staging_tag = {t.key: t.value for t in mv.tags}.get("cd_staging", "")
        if staging_tag != "pass":
            raise RuntimeError(
                f"v{candidate_version} has cd_staging='{staging_tag}'. "
                "Staging must pass before production promotion."
            )
        print(f"  Staging gate confirmed: cd_staging={staging_tag}")
    except mlflow.exceptions.MlflowException as exc:
        print(f"  WARNING: Could not verify staging gate: {exc}")

    # capture previous production version for the audit log
    prev_version = None
    try:
        prev_mv      = mlflow_client.get_model_version_by_alias(cfg.mlflow_model_name, "production")
        prev_version = prev_mv.version
        print(f"  Previous production: v{prev_version}")
    except mlflow.exceptions.MlflowException:
        print("  No previous production version (first deployment).")

    # promote
    mlflow_client.set_registered_model_alias(
        name    = cfg.mlflow_model_name,
        alias   = "production",
        version = candidate_version,
    )
    print(f"  Alias 'production' → v{candidate_version}  DONE")

    deployed_at = datetime.now(timezone.utc).isoformat()
    for key, val in [
        ("cd_production",  "deployed"),
        ("cd_deployed_at", deployed_at),
        ("cd_approved_by", approved_by),
        ("cd_git_sha",     git_sha),
    ]:
        mlflow_client.set_model_version_tag(cfg.mlflow_model_name, candidate_version, key, val)
    if prev_version:
        mlflow_client.set_model_version_tag(
            cfg.mlflow_model_name, candidate_version, "cd_replaced_version", prev_version
        )

    run_name = f"cd-production-{git_sha[:8]}"
    with mlflow.start_run(run_name=run_name) as cd_run:
        log_cd_production_run(
            run                     = cd_run,
            run_name                = run_name,
            model_name              = cfg.mlflow_model_name,
            candidate_version       = candidate_version,
            prev_production_version = prev_version,
            template_cfg            = template_cfg,
            approved_by             = approved_by,
            git_sha                 = git_sha,
            git_ref                 = git_ref,
            deployed_at             = deployed_at,
        )
        run_id = cd_run.info.run_id

    print(f"\n  Audit logged: {run_id}")
    print(f"  CD_RESULT: PASS")
    return CDResult(passed=True, mlflow_run_id=run_id)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_smoke_probes(
    client       : object,
    template_cfg : dict,
    probes       : list[dict],
) -> tuple[bool, list[dict]]:
    """Run all smoke probes. Returns (all_passed, results)."""
    results = []
    print(f"\nRunning {len(probes)} smoke probes…")

    for probe in probes:
        reply, latency_ms = _call_model(client, template_cfg, probe["user_message"])
        reply_lower = reply.lower()

        contains_ok     = all(kw.lower() in reply_lower for kw in probe.get("must_contain", []))
        not_contains_ok = all(kw.lower() not in reply_lower for kw in probe.get("must_not_contain", []))
        latency_ok      = latency_ms <= probe.get("max_latency_ms", 8000)
        passed          = contains_ok and not_contains_ok and latency_ok

        status = "PASS" if passed else "FAIL"
        print(f"  [{probe['id']}] {status}  latency={latency_ms:.0f}ms")
        if not contains_ok:
            print(f"    missing: {[kw for kw in probe.get('must_contain', []) if kw.lower() not in reply_lower]}")
        if not not_contains_ok:
            print(f"    VIOLATIONS: {[kw for kw in probe.get('must_not_contain', []) if kw.lower() in reply_lower]}")

        results.append({
            "probe_id"       : probe["id"],
            "user_message"   : probe["user_message"],
            "reply_snippet"  : reply[:120],
            "latency_ms"     : latency_ms,
            "contains_ok"    : contains_ok,
            "not_contains_ok": not_contains_ok,
            "latency_ok"     : latency_ok,
            "passed"         : passed,
        })

    return all(r["passed"] for r in results), results


def _call_model(client, template_cfg: dict, user_message: str) -> tuple[str, float]:
    """Single model call. Returns (reply, latency_ms)."""
    messages = [{"role": "system", "content": template_cfg["system"]}]
    messages.extend(template_cfg.get("few_shots", []))
    messages.append({"role": "user", "content": user_message})

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model       = template_cfg["model"],
        max_tokens  = template_cfg["max_tokens"],
        temperature = template_cfg["temperature"],
        messages    = messages,
    )
    return response.choices[0].message.content, round((time.perf_counter() - t0) * 1000, 1)
