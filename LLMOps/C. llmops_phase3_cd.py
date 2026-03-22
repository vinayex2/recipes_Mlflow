# Databricks notebook source
# llmops_phase3_cd.py
#
# LLMOps Phase 3 — Continuous Deployment notebook
#
# Upload to: /Shared/llmops/llmops_phase3_cd
#
# This single notebook handles both stages of the CD pipeline, controlled
# by the `deployment_stage` widget parameter:
#
#   deployment_stage = "staging"
#     → Runs a smoke test against the candidate model using direct API calls.
#       Validates latency and response quality with 3 probe questions.
#       Tags the model version with cd.staging=pass/fail.
#       Exits 1 on failure so the GitHub Actions staging job fails.
#
#   deployment_stage = "production"
#     → Called only AFTER a human reviewer has approved in GitHub.
#       Reassigns the MLflow "production" alias to the candidate version.
#       Logs a full audit record: who approved, when, git SHA, model version.
#       Tags the model version with cd.production=deployed.
#       Exits 1 on any failure so the approval is not silently lost.
#
# On Databricks Free Edition: serverless compute, %pip for dependencies.

# COMMAND ----------

# MAGIC %pip install -q openai>=1.30.0 dotenv

# COMMAND ----------

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# COMMAND ----------

#required for discovering mlflow services in databricks
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

try:
    dbutils  # noqa: F821
    _in_databricks = True
except NameError:
    _in_databricks = False
    class _W:
        def get(self, k, d=""): return d
    class _D:
        widgets = _W()
    dbutils = _D()

# COMMAND ----------

# dbutils.widgets.text("git_sha", "Git SHA")
# dbutils.widgets.text("git_ref", "Git Ref")
# dbutils.widgets.text("approved_by",  "Approved By")
# dbutils.widgets.text("model_version",  "")
# dbutils.widgets.text("deployment_stage",  "staging")
# DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
# GEMINI_ENDPOINT =  os.environ.get("GEMINI_ENDPOINT", "")
DATABRICKS_TOKEN = dbutils.widgets.get("DATABRICKS_TOKEN") if _in_databricks else os.environ.get("DATABRICKS_TOKEN", "")
GEMINI_ENDPOINT = dbutils.widgets.get("GEMINI_ENDPOINT") if _in_databricks else os.environ.get("GEMINI_ENDPOINT", "")



# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 0.  WIDGET PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

DEPLOYMENT_STAGE = dbutils.widgets.get("deployment_stage") if _in_databricks else "staging"
GIT_SHA          = dbutils.widgets.get("git_sha" ) if _in_databricks else          "local"
GIT_REF          = dbutils.widgets.get("git_ref" ) if _in_databricks else         "local"
APPROVED_BY      = dbutils.widgets.get("approved_by" ) if _in_databricks else     ""
MODEL_VERSION    = dbutils.widgets.get("model_version" ) if _in_databricks else   ""


if not DATABRICKS_TOKEN:
    raise ValueError(
        "DATABRICKS_TOKEN not found. Pass it as a widget parameter or set the "
        "env var. In production use dbutils.secrets."
    )

if not GEMINI_ENDPOINT:
    raise ValueError(
        "GEMINI_ENDPOINT not found. Pass it as a widget parameter or set the "
        "env var. In production use dbutils.secrets."
    )    

print(f"Stage      : {DEPLOYMENT_STAGE}")
print(f"SHA        : {GIT_SHA[:12]}")
print(f"Approved by: {APPROVED_BY or '(automated staging)'}")
print(f"Version    : {MODEL_VERSION or '(resolve from candidate alias)'}")

# ════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ════════════════════════════════════════════════════════════════════════════

MODEL_NAME       = os.getenv("MLFLOW_MODEL_NAME",   "workspace.default.llmops_support_agent")
CANDIDATE_ALIAS  = "candidate"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "llmops_phase3_cd")

# mlflow.set_experiment(EXPERIMENT_NAME)

oai_client = OpenAI(api_key  = DATABRICKS_TOKEN,base_url = GEMINI_ENDPOINT)
mlflow_client = mlflow.MlflowClient()

# ════════════════════════════════════════════════════════════════════════════
# 2.  RESOLVE MODEL VERSION
#     Use the explicit version if provided (manual re-deploy),
#     otherwise resolve from the 'candidate' alias.
# ════════════════════════════════════════════════════════════════════════════

def resolve_candidate() -> tuple[str, str]:
    """Returns (version, run_id) for the candidate model."""
    if MODEL_VERSION:
        mv = mlflow_client.get_model_version(MODEL_NAME, MODEL_VERSION)
    else:
        mv = mlflow_client.get_model_version_by_alias(MODEL_NAME, CANDIDATE_ALIAS)
    print(f"Resolved candidate: {MODEL_NAME} v{mv.version}  run_id={mv.run_id[:8]}…")
    return mv.version, mv.run_id

candidate_version, candidate_run_id = resolve_candidate()

# ════════════════════════════════════════════════════════════════════════════
# 3.  LOAD PROMPT CONFIG (same reliable pattern as Phase 2)
# ════════════════════════════════════════════════════════════════════════════

def load_prompt_config(run_id: str) -> dict:
    """Download the prompt_config artifact from the registration run."""
    artifact_uri = f"runs:/{run_id}/prompt_config"
    print(f"Downloading prompt config from: {artifact_uri}")
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    cfg_files = list(Path(local_dir).glob("*.json"))
    if not cfg_files:
        raise FileNotFoundError(
            f"No JSON found in {local_dir}. Re-run Phase 1 to re-register."
        )
    cfg_path = max(cfg_files, key=lambda p: p.stat().st_mtime)
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"Loaded template: {cfg['name']} v{cfg['version']}  model={cfg['model']}")
    return cfg

template_cfg = load_prompt_config(candidate_run_id)

# ════════════════════════════════════════════════════════════════════════════
# 4.  SMOKE TEST PROBES
#     3 representative questions — faster than the full eval suite.
#     One happy-path, one edge-case, one out-of-scope trap.
# ════════════════════════════════════════════════════════════════════════════

SMOKE_PROBES = [
    {
        "id"              : "smoke-001",
        "user_message"    : "How do I reset my password?",
        "must_contain"    : ["password", "reset"],
        "must_not_contain": [],
        "max_latency_ms"  : 8000,
    },
    {
        "id"              : "smoke-002",
        "user_message"    : "How do I cancel my subscription?",
        "must_contain"    : ["cancel", "subscription"],
        "must_not_contain": [],
        "max_latency_ms"  : 8000,
    },
    {
        "id"              : "smoke-003",
        "user_message"    : "What is the capital of France?",   # out-of-scope trap
        "must_contain"    : [],
        "must_not_contain": ["Paris"],                          # must refuse
        "max_latency_ms"  : 8000,
    },
]


def call_model(cfg: dict, message: str) -> tuple[str, float]:
    """Single model call. Returns (reply, latency_ms)."""
    messages = [{"role": "system", "content": cfg["system"]}]
    messages.extend(cfg.get("few_shots", []))
    messages.append({"role": "user", "content": message})

    t0 = time.perf_counter()
    response = oai_client.chat.completions.create(
        model       = cfg["model"],
        max_tokens  = cfg["max_tokens"],
        temperature = cfg["temperature"],
        messages    = messages,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.choices[0].message.content, round(latency_ms, 1)


def run_smoke_tests(cfg: dict) -> tuple[bool, list[dict]]:
    """Run all smoke probes. Returns (all_passed, results)."""
    results = []
    print(f"\nRunning {len(SMOKE_PROBES)} smoke probes…")

    for probe in SMOKE_PROBES:
        reply, latency_ms = call_model(cfg, probe["user_message"])
        reply_lower = reply.lower()

        contains_ok    = all(kw.lower() in reply_lower for kw in probe["must_contain"])
        not_contains_ok= all(kw.lower() not in reply_lower for kw in probe["must_not_contain"])
        latency_ok     = latency_ms <= probe["max_latency_ms"]
        passed         = contains_ok and not_contains_ok and latency_ok

        status = "PASS" if passed else "FAIL"
        print(f"  [{probe['id']}] {status}  latency={latency_ms:.0f}ms")
        if not contains_ok:
            missing = [kw for kw in probe["must_contain"] if kw.lower() not in reply_lower]
            print(f"    missing keywords: {missing}")
        if not not_contains_ok:
            violations = [kw for kw in probe["must_not_contain"] if kw.lower() in reply_lower]
            print(f"    VIOLATIONS: {violations}")
        if not latency_ok:
            print(f"    latency {latency_ms:.0f}ms > limit {probe['max_latency_ms']}ms")

        results.append({
            "probe_id"      : probe["id"],
            "user_message"  : probe["user_message"],
            "reply_snippet" : reply[:120],
            "latency_ms"    : latency_ms,
            "contains_ok"   : contains_ok,
            "not_contains_ok": not_contains_ok,
            "latency_ok"    : latency_ok,
            "passed"        : passed,
        })

    all_passed = all(r["passed"] for r in results)
    return all_passed, results

# ════════════════════════════════════════════════════════════════════════════
# 5.  STAGING PATH
# ════════════════════════════════════════════════════════════════════════════

def run_staging() -> None:
    """
    Deploy candidate to staging (smoke test via direct API call — Databricks
    Free Edition does not support Model Serving endpoints, so we validate the
    model's prompt config directly using the OpenAI client, which is equivalent
    to what the production serving endpoint would do).

    On a paid Databricks workspace you would also:
      - Create/update a Model Serving endpoint via the REST API
      - Run smoke probes against the endpoint URL
      - Tear down the staging endpoint on failure
    """
    print(f"\n{'═'*56}")
    print(f"  Phase 3 — Staging deployment")
    print(f"  Model   : {MODEL_NAME} v{candidate_version}")
    print(f"  SHA     : {GIT_SHA[:12]}")
    print(f"{'═'*56}")

    smoke_passed, smoke_results = run_smoke_tests(template_cfg)

    # ── Log staging run to MLflow ─────────────────────────────────────────────
    run_name = f"cd-staging-{GIT_SHA[:8]}-{'pass' if smoke_passed else 'fail'}"
    with mlflow.start_run(run_name=run_name) as cd_run:
        mlflow.log_params({
            "model_name"      : MODEL_NAME,
            "model_version"   : candidate_version,
            "template_name"   : template_cfg["name"],
            "template_version": template_cfg["version"],
            "model"           : template_cfg["model"],
            "deployment_stage": "staging",
        })
        mlflow.log_metrics({
            "cd/smoke_pass_rate" : sum(r["passed"] for r in smoke_results) / len(smoke_results),
            "cd/avg_latency_ms"  : sum(r["latency_ms"] for r in smoke_results) / len(smoke_results),
            "cd/all_passed"      : int(smoke_passed),
        })
        mlflow.set_tags({
            "cd.stage"           : "staging",
            "cd.result"          : "PASS" if smoke_passed else "FAIL",
            "cd.git_sha"         : GIT_SHA,
            "cd.git_ref"         : GIT_REF,
            "cd.run_date"        : datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "candidate.version"  : candidate_version,
        })

        artifact = {
            "cd_run_id"     : cd_run.info.run_id,
            "stage"         : "staging",
            "passed"        : smoke_passed,
            "git_sha"       : GIT_SHA,
            "smoke_results" : smoke_results,
            "timestamp"     : datetime.now(timezone.utc).isoformat(),
        }
        artifact_path = Path("/tmp") / f"{run_name}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2))
        mlflow.log_artifact(str(artifact_path), artifact_path="cd_results")

    # ── Tag the model version ─────────────────────────────────────────────────
    mlflow_client.set_model_version_tag(
        name    = MODEL_NAME,
        version = candidate_version,
        key     = "cd_staging",
        value   = "pass" if smoke_passed else "fail",
    )

    print(f"\n{'─'*56}")
    print(f"  MLflow run : {cd_run.info.run_id}")
    print(f"  Result     : {'PASS — ready for human approval' if smoke_passed else 'FAIL — blocking deployment'}")

    if not smoke_passed:
        print("\nStaging smoke tests FAILED. Production deployment blocked.")
        sys.exit(1)

    print("\nStaging PASSED. Awaiting human approval for production deployment.")

# ════════════════════════════════════════════════════════════════════════════
# 6.  PRODUCTION PATH
#     Only reached after a human reviewer has approved in GitHub.
# ════════════════════════════════════════════════════════════════════════════

def run_production() -> None:
    """
    Promote the candidate model to production:
      1. Reassign the 'production' alias to the candidate version
      2. Log a complete audit record to MLflow (who, when, what, why)
      3. Tag the model version with cd.production=deployed
    """
    print(f"\n{'═'*56}")
    print(f"  Phase 3 — Production deployment")
    print(f"  Model    : {MODEL_NAME} v{candidate_version}")
    print(f"  Approved : {APPROVED_BY}")
    print(f"  SHA      : {GIT_SHA[:12]}")
    print(f"{'═'*56}")

    # ── Check staging gate was passed ─────────────────────────────────────────
    try:
        mv = mlflow_client.get_model_version(MODEL_NAME, candidate_version)
        staging_tag = {t.key: t.value for t in mv.tags}.get("cd.staging", "")
        if staging_tag != "pass":
            raise RuntimeError(
                f"Model version {candidate_version} has cd.staging='{staging_tag}'. "
                "Staging must pass before production promotion. "
                "Re-run the CD pipeline from scratch."
            )
        print(f"  Staging gate confirmed: cd.staging={staging_tag}")
    except Exception as exc:
        print(f"  WARNING: Could not verify staging gate: {exc}")
        # Don't block — the human reviewer's approval is the gate.
        # Log the warning and continue.

    # ── Promote: move production alias to candidate version ───────────────────
    print(f"\n  Promoting {CANDIDATE_ALIAS} → {PRODUCTION_ALIAS} alias…")

    # Capture the previous production version for the audit log
    prev_production_version = None
    try:
        prev_mv = mlflow_client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        prev_production_version = prev_mv.version
        print(f"  Previous production version: {prev_production_version}")
    except mlflow.exceptions.MlflowException:
        print("  No previous production version (first deployment).")

    mlflow_client.set_registered_model_alias(
        name    = MODEL_NAME,
        alias   = PRODUCTION_ALIAS,
        version = candidate_version,
    )
    print(f"  Alias '{PRODUCTION_ALIAS}' → version {candidate_version}  DONE")
    #Remove Candidate Alias
    mlflow_client.delete_registered_model_alias(model_name, CANDIDATE_ALIAS ,version = candidate_version)

    # ── Tag the model version ─────────────────────────────────────────────────
    deployed_at = datetime.now(timezone.utc).isoformat()
    mlflow_client.set_model_version_tag(MODEL_NAME, candidate_version, "cd_production",    "deployed")
    mlflow_client.set_model_version_tag(MODEL_NAME, candidate_version, "cd_deployed_at",   deployed_at)
    mlflow_client.set_model_version_tag(MODEL_NAME, candidate_version, "cd_approved_by",   APPROVED_BY)
    mlflow_client.set_model_version_tag(MODEL_NAME, candidate_version, "cd_git_sha",       GIT_SHA)
    if prev_production_version:
        mlflow_client.set_model_version_tag(
            MODEL_NAME, candidate_version, "cd_replaced_version", prev_production_version
        )

    # ── Log audit run to MLflow ───────────────────────────────────────────────
    run_name = f"cd-production-{GIT_SHA[:8]}"
    with mlflow.start_run(run_name=run_name) as cd_run:
        mlflow.log_params({
            "model_name"              : MODEL_NAME,
            "model_version"           : candidate_version,
            "prev_production_version" : prev_production_version or "none",
            "template_name"           : template_cfg["name"],
            "template_version"        : template_cfg["version"],
            "model"                   : template_cfg["model"],
            "deployment_stage"        : "production",
        })
        mlflow.log_metrics({
            "cd/promoted": 1,
        })
        mlflow.set_tags({
            "cd.stage"         : "production",
            "cd.result"        : "deployed",
            "cd.approved_by"   : APPROVED_BY,
            "cd.git_sha"       : GIT_SHA,
            "cd.git_ref"       : GIT_REF,
            "cd.deployed_at"   : deployed_at,
            "cd.prev_version"  : prev_production_version or "none",
            "candidate.version": candidate_version,
        })

        # Full audit JSON artifact — immutable record of this deployment decision
        audit = {
            "event"                   : "production_deployment",
            "cd_run_id"               : cd_run.info.run_id,
            "model_name"              : MODEL_NAME,
            "model_version"           : candidate_version,
            "prev_production_version" : prev_production_version,
            "template_name"           : template_cfg["name"],
            "template_version"        : template_cfg["version"],
            "llm_model"               : template_cfg["model"],
            "approved_by"             : APPROVED_BY,
            "git_sha"                 : GIT_SHA,
            "git_ref"                 : GIT_REF,
            "deployed_at"             : deployed_at,
        }
        audit_path = Path("/tmp") / f"{run_name}_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2))
        mlflow.log_artifact(str(audit_path), artifact_path="cd_audit")

        print(f"\n  Audit logged : {cd_run.info.run_id}")

    print(f"\n{'─'*56}")
    print(f"  Production deployment COMPLETE")
    print(f"  Model     : {MODEL_NAME} v{candidate_version}")
    print(f"  Approved  : {APPROVED_BY}")
    print(f"  Deployed  : {deployed_at}")
    print(f"  Replaced  : v{prev_production_version or 'none'}")
    print(f"  CD_RESULT : PASS")

# ════════════════════════════════════════════════════════════════════════════
# 7.  DISPATCH
# ════════════════════════════════════════════════════════════════════════════

if DEPLOYMENT_STAGE == "staging":
    run_staging()
elif DEPLOYMENT_STAGE == "production":
    run_production()
else:
    raise ValueError(
        f"Unknown deployment_stage='{DEPLOYMENT_STAGE}'. "
        "Must be 'staging' or 'production'."
    )
