# Databricks notebook source
# phase2_ci.py
#
# Phase 2 — CI notebook. Upload to: /Shared/llmops/phase2_ci
#
# Change PROJECT_YAML to point at a different project — nothing else changes.

# COMMAND ----------

%pip install -q openai>=1.30.0 pyyaml

# COMMAND ----------

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

GIT_SHA          = dbutils.widgets.get("git_sha")  or "local"
GIT_REF          = dbutils.widgets.get("git_ref")  or "local"
GIT_PR           = dbutils.widgets.get("git_pr")   or ""
DATABRICKS_TOKEN = dbutils.widgets.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_TOKEN", "")
GEMINI_ENDPOINT  = dbutils.widgets.get("GEMINI_ENDPOINT")  or os.environ.get("GEMINI_ENDPOINT", "")

if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN not set.")
if not GEMINI_ENDPOINT:
    raise ValueError("GEMINI_ENDPOINT not set.")

# COMMAND ----------

# ── Project selection ─────────────────────────────────────────────────────────

_project = os.environ.get("PROJECT") or "support_agent"
if "--project" in sys.argv:
    _project = sys.argv[sys.argv.index("--project") + 1]

PROJECT_YAML = Path(__file__).parent / "projects" / _project / "project.yaml"

# COMMAND ----------

from llmops_core.project_config import load_project_config
from llmops_core.pipeline_ci    import run_ci

cfg = load_project_config(PROJECT_YAML)
mlflow.set_experiment(cfg.mlflow_experiment)

client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=GEMINI_ENDPOINT)

# COMMAND ----------

# ── Load candidate from registry ──────────────────────────────────────────────

mlflow_client = mlflow.MlflowClient()
mv            = mlflow_client.get_model_version_by_alias(cfg.mlflow_model_name, "candidate")
candidate_version = mv.version
candidate_run_id  = mv.run_id

import json
from pathlib import Path as _Path
artifact_uri = f"runs:/{candidate_run_id}/prompt_config"
local_dir    = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
cfg_files    = list(_Path(local_dir).glob("*.json"))
if not cfg_files:
    raise FileNotFoundError(f"No prompt_config JSON found in {local_dir}")
template_cfg = json.loads(max(cfg_files, key=lambda p: p.stat().st_mtime).read_text())

print(f"Candidate: {cfg.mlflow_model_name} v{candidate_version}")
print(f"Template : {template_cfg['name']} v{template_cfg['version']}")

# COMMAND ----------

# ── Run CI ────────────────────────────────────────────────────────────────────

result = run_ci(
    cfg               = cfg,
    client            = client,
    candidate_version = candidate_version,
    candidate_run_id  = candidate_run_id,
    template_cfg      = template_cfg,
    git_sha           = GIT_SHA,
    git_ref           = GIT_REF,
    git_pr            = GIT_PR,
)

# COMMAND ----------

# ── Tag model version ─────────────────────────────────────────────────────────

if _in_databricks:
    mlflow_client.set_model_version_tag(
        name    = cfg.mlflow_model_name,
        version = candidate_version,
        key     = "ci_last_result",
        value   = "PASS" if result.all_gates_passed else "FAIL",
    )
    mlflow_client.set_model_version_tag(
        name    = cfg.mlflow_model_name,
        version = candidate_version,
        key     = "ci_last_run_id",
        value   = result.mlflow_run_id,
    )

# COMMAND ----------

# ── Final verdict ─────────────────────────────────────────────────────────────

print(f"\n{'═'*54}")
if result.all_gates_passed:
    print("CI_RESULT: PASS")
else:
    failed = [g.name for g in result.gates if not g.passed]
    print(f"CI_RESULT: FAIL  reason=gates_failed:{','.join(failed)}")
    sys.exit(1)
