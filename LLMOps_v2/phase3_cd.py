# Databricks notebook source
# phase3_cd.py
#
# Phase 3 — CD notebook. Upload to: /Shared/llmops/phase3_cd
#
# Change PROJECT_YAML to point at a different project — nothing else changes.
# deployment_stage widget: "staging" or "production"

# COMMAND ----------

%pip install -q openai>=1.30.0 pyyaml

# COMMAND ----------

import json
import os
import sys
from pathlib import Path

import mlflow
from openai import OpenAI

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

try:
    dbutils
    _in_databricks = True
except NameError:
    _in_databricks = False
    class _W:
        def get(self, k, d=""): return d
    class _D:
        widgets = _W()
    dbutils = _D()

DEPLOYMENT_STAGE = dbutils.widgets.get("deployment_stage") or "staging"
GIT_SHA          = dbutils.widgets.get("git_sha")          or "local"
GIT_REF          = dbutils.widgets.get("git_ref")          or "local"
APPROVED_BY      = dbutils.widgets.get("approved_by")      or ""
MODEL_VERSION    = dbutils.widgets.get("model_version")    or ""
DATABRICKS_TOKEN = dbutils.widgets.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_TOKEN", "")
GEMINI_ENDPOINT  = dbutils.widgets.get("GEMINI_ENDPOINT")  or os.environ.get("GEMINI_ENDPOINT", "")

if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN not set.")
if not GEMINI_ENDPOINT:
    raise ValueError("GEMINI_ENDPOINT not set.")

print(f"Stage: {DEPLOYMENT_STAGE}  SHA: {GIT_SHA[:12]}  Approved by: {APPROVED_BY or '(auto)'}")

# COMMAND ----------

# ── Project selection ─────────────────────────────────────────────────────────

_project = os.environ.get("PROJECT") or "support_agent"
if "--project" in sys.argv:
    _project = sys.argv[sys.argv.index("--project") + 1]

PROJECT_YAML = Path(__file__).parent / "projects" / _project / "project.yaml"

# COMMAND ----------

import yaml
from llmops_core.project_config import load_project_config
from llmops_core.pipeline_cd    import run_staging, run_production

cfg = load_project_config(PROJECT_YAML)
mlflow.set_experiment(cfg.mlflow_experiment)

# smoke_probes live in the YAML alongside everything else
with open(PROJECT_YAML) as f:
    raw_yaml = yaml.safe_load(f)
smoke_probes = raw_yaml.get("smoke_probes", [])
if not smoke_probes:
    raise ValueError("project.yaml must define at least one smoke probe under 'smoke_probes:'")

client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=GEMINI_ENDPOINT)

# COMMAND ----------

# ── Resolve candidate version ─────────────────────────────────────────────────

mlflow_client = mlflow.MlflowClient()
if MODEL_VERSION:
    mv = mlflow_client.get_model_version(cfg.mlflow_model_name, MODEL_VERSION)
else:
    mv = mlflow_client.get_model_version_by_alias(cfg.mlflow_model_name, "candidate")

candidate_version = mv.version
candidate_run_id  = mv.run_id

artifact_uri = f"runs:/{candidate_run_id}/prompt_config"
local_dir    = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
cfg_files    = list(Path(local_dir).glob("*.json"))
template_cfg = json.loads(max(cfg_files, key=lambda p: p.stat().st_mtime).read_text())

print(f"Candidate: {cfg.mlflow_model_name} v{candidate_version}")
print(f"Template : {template_cfg['name']} v{template_cfg['version']}")

# COMMAND ----------

# ── Dispatch ──────────────────────────────────────────────────────────────────

if DEPLOYMENT_STAGE == "staging":
    result = run_staging(
        cfg               = cfg,
        client            = client,
        candidate_version = candidate_version,
        template_cfg      = template_cfg,
        smoke_probes      = smoke_probes,
        git_sha           = GIT_SHA,
        git_ref           = GIT_REF,
    )
    if not result.passed:
        print("\nStaging FAILED. Blocking production deployment.")
        sys.exit(1)
    print("\nStaging PASSED. Awaiting human approval.")

elif DEPLOYMENT_STAGE == "production":
    result = run_production(
        cfg               = cfg,
        candidate_version = candidate_version,
        template_cfg      = template_cfg,
        approved_by       = APPROVED_BY,
        git_sha           = GIT_SHA,
        git_ref           = GIT_REF,
    )
    if not result.passed:
        sys.exit(1)

else:
    raise ValueError(f"Unknown deployment_stage='{DEPLOYMENT_STAGE}'")
