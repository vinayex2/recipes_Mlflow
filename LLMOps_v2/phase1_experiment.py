# Databricks notebook source
# phase1_experiment.py
#
# Phase 1 — Local experiment runner.
# Upload to: /Shared/llmops/phase1_experiment
#
# To run for a different project, pass --project <name> on the CLI,
# set the PROJECT env var, or edit the PROJECT_YAML line below.
# The project name must match a folder under projects/.

# COMMAND ----------

# %pip install -q openai>=1.30.0 pyyaml mlflow tiktoken python-dotenv

# COMMAND ----------

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# ── Project selection ─────────────────────────────────────────────────────────

_project = os.environ.get("PROJECT") or "support_agent"

if "--project" in sys.argv:
    _project = sys.argv[sys.argv.index("--project") + 1]

PROJECT_YAML = Path(__file__).parent / "projects" / _project / "project.yaml"

# COMMAND ----------

from llmops_core.project_config      import load_project_config
from llmops_core.pipeline_experiment import run_all_experiments, register_best_candidate

cfg = load_project_config(PROJECT_YAML)
print(f"Project    : {cfg.name}")
print(f"Experiment : {cfg.mlflow_experiment}")
print(f"Templates  : {[t.name for t in cfg.templates]}")
print(f"Eval cases : {len(cfg.golden_dataset)}")

mlflow.set_experiment(cfg.mlflow_experiment)

# COMMAND ----------

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
client = OpenAI(
    api_key  = DATABRICKS_TOKEN,
    base_url = os.environ.get("GEMINI_ENDPOINT"),
)

# COMMAND ----------

summaries = run_all_experiments(cfg, client, run_judge=True)

# COMMAND ----------

# prompt_model.py can live alongside each project's YAML for a project-specific
# pyfunc, or fall back to the shared root copy used by all projects.
model_code_path = PROJECT_YAML.parent / "prompt_model.py"
if not model_code_path.exists():
    model_code_path = Path(__file__).parent / "prompt_model.py"

uri = register_best_candidate(cfg, summaries, model_code_path)

if uri:
    print(f"\n  Candidate registered: {uri}")
    print("  Next: run phase2_ci.py")
else:
    print("\n  No candidate met the threshold — iterate on templates in project.yaml")
