# Databricks notebook source
# phase4_champion_challenger.py
#
# Phase 4 — Champion/challenger. Upload to: /Shared/llmops/phase4_champion_challenger
#
# Change PROJECT_YAML to point at a different project — nothing else changes.

# COMMAND ----------

%pip install -q openai>=1.30.0 pyyaml numpy

# COMMAND ----------

import json
import os
import sys
import time
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
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

_in_databricks = False

CHAMPION_VERSION   = dbutils.widgets.get("champion_version")   or ""
CHALLENGER_VERSION = dbutils.widgets.get("challenger_version") or ""
GIT_SHA            = dbutils.widgets.get("git_sha")            or "local"
DATABRICKS_TOKEN   = dbutils.widgets.get("DATABRICKS_TOKEN")   or os.environ.get("DATABRICKS_TOKEN", "")
GEMINI_ENDPOINT    = dbutils.widgets.get("GEMINI_ENDPOINT")    or os.environ.get("GEMINI_ENDPOINT", "")

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
from llmops_core.evaluators     import rule_eval, llm_judge
from llmops_core.mlflow_helpers import log_cc_run

cfg = load_project_config(PROJECT_YAML)
mlflow.set_experiment(cfg.mlflow_experiment)

client        = OpenAI(api_key=DATABRICKS_TOKEN, base_url=GEMINI_ENDPOINT)
mlflow_client = mlflow.MlflowClient()

MIN_IMPROVEMENT = cfg.gates.cc_min_improvement

# COMMAND ----------

# ── Resolve versions ──────────────────────────────────────────────────────────

def resolve_version(alias, override, fallback_alias=None):
    if override:
        mv = mlflow_client.get_model_version(cfg.mlflow_model_name, override)
        return mv.version, mv.run_id, f"explicit:{override}"
    try:
        mv = mlflow_client.get_model_version_by_alias(cfg.mlflow_model_name, alias)
        return mv.version, mv.run_id, alias
    except mlflow.exceptions.MlflowException:
        if fallback_alias:
            mv = mlflow_client.get_model_version_by_alias(cfg.mlflow_model_name, fallback_alias)
            return mv.version, mv.run_id, fallback_alias
        raise

champ_version, champ_run_id, champ_resolved = resolve_version("champion",   CHAMPION_VERSION,   fallback_alias="production")
chall_version, chall_run_id, chall_resolved = resolve_version("candidate",  CHALLENGER_VERSION)

if champ_version == chall_version:
    print("Champion and challenger are the same version — nothing to compare.")
    sys.exit(0)

print(f"Champion  : v{champ_version} (via {champ_resolved})")
print(f"Challenger: v{chall_version} (via {chall_resolved})")

# COMMAND ----------

# ── Load prompt configs ───────────────────────────────────────────────────────

def load_prompt_config(run_id, label):
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/prompt_config")
    files     = list(Path(local_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON found for {label} run_id={run_id}")
    return json.loads(max(files, key=lambda p: p.stat().st_mtime).read_text())

champ_cfg = load_prompt_config(champ_run_id, "champion")
chall_cfg = load_prompt_config(chall_run_id, "challenger")

# COMMAND ----------

# ── Eval runner ───────────────────────────────────────────────────────────────

COST_PER_1K_INPUT  = cfg.cost.per_1k_input_usd
COST_PER_1K_OUTPUT = cfg.cost.per_1k_output_usd

def call_model(template_cfg, message):
    messages = [{"role": "system", "content": template_cfg["system"]}]
    messages.extend(template_cfg.get("few_shots", []))
    messages.append({"role": "user", "content": message})
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=template_cfg["model"], max_tokens=template_cfg["max_tokens"],
        temperature=template_cfg["temperature"], messages=messages,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return (resp.choices[0].message.content, round(latency_ms, 1),
            resp.usage.prompt_tokens, resp.usage.completion_tokens)

def run_full_eval(template_cfg, label):
    print(f"\n  Evaluating {label} ({template_cfg['name']} v{template_cfg['version']})…")
    case_results = []

    for case in cfg.golden_dataset:
        reply, latency_ms, in_tok, out_tok = call_model(template_cfg, case["user_message"])
        cost = cfg.cost.estimate(in_tok, out_tok)
        r    = rule_eval(reply, case)
        j    = llm_judge(
            user_message = case["user_message"],
            response     = reply,
            eval_id      = case["id"],
            client       = client,
            judge_model  = cfg.judge_model,
        )
        print(f"    [{case['id']}] rule={'PASS' if r.passed else 'FAIL'} "
              f"score={r.score}  judge={j.avg_score}  latency={latency_ms:.0f}ms")

        case_results.append({
            "eval_id"           : case["id"],
            "rule_passed"       : r.passed,
            "rule_score"        : r.score,
            "topic_misses"      : r.topic_misses,
            "violations"        : r.must_not_violated,
            "judge_helpfulness" : j.helpfulness,
            "judge_faithfulness": j.faithfulness,
            "judge_conciseness" : j.conciseness,
            "judge_safety"      : j.safety,
            "judge_avg"         : j.avg_score,
            "judge_reasoning"   : j.reasoning,
            "latency_ms"        : latency_ms,
            "input_tokens"      : in_tok,
            "output_tokens"     : out_tok,
            "cost_usd"          : round(cost, 6),
        })

    n               = len(case_results)
    rule_pass_rate  = sum(r["rule_passed"] for r in case_results) / n
    avg_rule_score  = sum(r["rule_score"]  for r in case_results) / n
    avg_judge_score = sum(r["judge_avg"]   for r in case_results) / n
    avg_latency_ms  = sum(r["latency_ms"]  for r in case_results) / n
    avg_cost_usd    = sum(r["cost_usd"]    for r in case_results) / n
    composite       = (avg_rule_score * 0.5) + ((avg_judge_score / 5.0) * 0.5)

    print(f"    composite={composite:.4f}  rule={rule_pass_rate:.0%}  "
          f"judge={avg_judge_score:.2f}  latency={avg_latency_ms:.0f}ms")
    return {
        "label"          : label,
        "rule_pass_rate" : round(rule_pass_rate,  4),
        "avg_rule_score" : round(avg_rule_score,  4),
        "avg_judge_score": round(avg_judge_score, 4),
        "avg_latency_ms" : round(avg_latency_ms,  1),
        "avg_cost_usd"   : round(avg_cost_usd,    6),
        "composite"      : round(composite,        4),
        "case_results"   : case_results,
    }

# COMMAND ----------

print(f"\n{'═'*56}")
print(f"  Phase 4 — Champion / Challenger")
print(f"  Champion  : v{champ_version}   Challenger: v{chall_version}")
print(f"  Min improvement: {MIN_IMPROVEMENT}")
print(f"{'═'*56}")

champ_results = run_full_eval(champ_cfg, "champion")
chall_results = run_full_eval(chall_cfg, "challenger")

# COMMAND ----------

# ── Decision ──────────────────────────────────────────────────────────────────

score_delta = chall_results["composite"] - champ_results["composite"]
promote     = score_delta > MIN_IMPROVEMENT
evaluated_at = datetime.now(timezone.utc).isoformat()

print(f"\n  Champion  : {champ_results['composite']:.4f}")
print(f"  Challenger: {chall_results['composite']:.4f}")
print(f"  Delta     : {score_delta:+.4f}  (need > +{MIN_IMPROVEMENT})")
print(f"  Decision  : {'PROMOTE' if promote else 'RETAIN'}")

if promote:
    mlflow_client.set_registered_model_alias(cfg.mlflow_model_name, "champion", chall_version)
    for key, val in [
        ("cc_result",         "promoted_to_champion"),
        ("cc_promoted_at",    evaluated_at),
        ("cc_score_delta",    str(round(score_delta, 4))),
        ("cc_replaced_champ", champ_version),
    ]:
        mlflow_client.set_model_version_tag(cfg.mlflow_model_name, chall_version, key, val)
    mlflow_client.set_model_version_tag(cfg.mlflow_model_name, champ_version, "cc_result", "retired_by_challenger")
    print(f"\n  Alias 'champion' → v{chall_version}  DONE")
else:
    mlflow_client.set_model_version_tag(cfg.mlflow_model_name, chall_version, "cc_result", "not_promoted")
    mlflow_client.set_model_version_tag(
        cfg.mlflow_model_name, chall_version, "cc_rejection_reason",
        f"delta={score_delta:+.4f} did not exceed min_improvement={MIN_IMPROVEMENT}"
    )
    mlflow_client.set_registered_model_alias(cfg.mlflow_model_name, "champion", champ_version)

# COMMAND ----------

# ── Log to MLflow ─────────────────────────────────────────────────────────────

run_name = f"cc-{'promoted' if promote else 'retained'}-{GIT_SHA[:8]}"
with mlflow.start_run(run_name=run_name) as cc_run:
    log_cc_run(
        run             = cc_run,
        run_name        = run_name,
        model_name      = cfg.mlflow_model_name,
        champ_version   = champ_version,
        chall_version   = chall_version,
        champ_cfg       = champ_cfg,
        chall_cfg       = chall_cfg,
        champ_results   = champ_results,
        chall_results   = chall_results,
        score_delta     = score_delta,
        promote         = promote,
        min_improvement = MIN_IMPROVEMENT,
        git_sha         = GIT_SHA,
        evaluated_at    = evaluated_at,
        n_eval_cases    = len(cfg.golden_dataset),
    )

print(f"\n  MLflow run: {cc_run.info.run_id}")
print(f"  {'CHALLENGER PROMOTED' if promote else 'CHAMPION RETAINED'}")
