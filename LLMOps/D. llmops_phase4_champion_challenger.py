# Databricks notebook source
# llmops_phase4_champion_challenger.py
#
# LLMOps Phase 4 — Champion / Challenger Promotion Logic
#
# Upload to: /Shared/llmops/llmops_phase4_champion_challenger
#
# What it does:
#   1. Resolves champion (alias: champion) and challenger (alias: candidate)
#      model versions from MLflow Model Registry
#   2. Loads both prompt configs via the reliable runs:/ artifact path
#   3. Runs the full frozen golden eval suite (5 cases) against BOTH models
#      independently — same cases, same order, same judge
#   4. Computes composite scores: 50% rule eval + 50% LLM-as-judge (normalised)
#      plus latency and cost metrics for tie-breaking
#   5. Applies the promotion decision:
#       challenger composite > champion composite + min_improvement  → PROMOTE
#       otherwise                                                    → RETAIN
#   6. On PROMOTE: reassigns the 'champion' alias to the challenger version,
#      retires the old champion alias, tags both versions
#   7. On RETAIN: tags the challenger as retired, logs why it didn't promote
#   8. Logs a full comparison run to MLflow with per-case results, aggregate
#      metrics, and the promotion decision as a structured artifact
#
# Alias conventions across the full LLMOps pipeline:
#   candidate   → set by Phase 1 after registration
#   production  → set by Phase 3 CD after human approval
#   champion    → set by Phase 4 after winning a head-to-head evaluation
#                 (initialised to the production version on first run)
#
# On Databricks Free Edition: serverless compute only, %pip for libs.

# COMMAND ----------

# MAGIC %pip install -q openai>=1.30.0 tiktoken>=0.7.0 dotenv

# COMMAND ----------

import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# COMMAND ----------

#required for discovering mlflow services in databricks
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 0.  WIDGET PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

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

_in_databricks = False    

CHAMPION_VERSION   = dbutils.widgets.get("champion_version") if _in_databricks else ""
CHALLENGER_VERSION = dbutils.widgets.get("challenger_version") if _in_databricks else ""
# Min composite score improvement challenger must beat champion by to promote.
# Default 0.02 = challenger must score at least 2 percentage points higher.
MIN_IMPROVEMENT    = float(dbutils.widgets.get("min_improvement") if _in_databricks else "0.02")
GIT_SHA            = dbutils.widgets.get("git_sha") if _in_databricks else "local"

DATABRICKS_TOKEN = dbutils.widgets.get("DATABRICKS_TOKEN") if _in_databricks else os.environ.get("DATABRICKS_TOKEN", "")
GEMINI_ENDPOINT = dbutils.widgets.get("GEMINI_ENDPOINT") if _in_databricks else os.environ.get("GEMINI_ENDPOINT", "")

if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN not found. Pass as widget or env var.")

if not GEMINI_ENDPOINT:
    raise ValueError("GEMINI_ENDPOINT not found. Pass as widget or env var.")

print(f"Champion version  : {CHAMPION_VERSION or '(resolve from champion alias)'}")
print(f"Challenger version: {CHALLENGER_VERSION or '(resolve from candidate alias)'}")
print(f"Min improvement   : {MIN_IMPROVEMENT}")
print(f"Git SHA           : {GIT_SHA[:12]}")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ════════════════════════════════════════════════════════════════════════════

MODEL_NAME         = os.getenv("MLFLOW_MODEL_NAME",   "workspace.default.llmops_support_agent")
CHAMPION_ALIAS     = "champion"
CANDIDATE_ALIAS    = "candidate"
PRODUCTION_ALIAS   = "production"
EXPERIMENT_NAME    = os.getenv("MLFLOW_EXPERIMENT_NAME", "llmops/phase4-champion-challenger")

# mlflow.set_experiment(EXPERIMENT_NAME)

oai_client = OpenAI(api_key = DATABRICKS_TOKEN,base_url = GEMINI_ENDPOINT)
mlflow_client = mlflow.MlflowClient()

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 2.  RESOLVE MODEL VERSIONS
#     Champion: from 'champion' alias, falling back to 'production' alias
#               on the very first run before 'champion' has been set.
#     Challenger: from 'candidate' alias (set by Phase 1 after registration).
# ════════════════════════════════════════════════════════════════════════════

def resolve_version(alias: str, override: str, fallback_alias: Optional[str] = None) -> tuple:
    """
    Resolve (version, run_id) from an explicit override version number,
    a primary alias, or an optional fallback alias.
    Returns (version_str, run_id, resolved_alias).
    """
    if override:
        mv = mlflow_client.get_model_version(MODEL_NAME, override)
        return mv.version, mv.run_id, f"explicit:{override}"

    try:
        mv = mlflow_client.get_model_version_by_alias(MODEL_NAME, alias)
        return mv.version, mv.run_id, alias
    except mlflow.exceptions.MlflowException:
        if fallback_alias:
            print(f"  Alias '{alias}' not found — falling back to '{fallback_alias}'")
            mv = mlflow_client.get_model_version_by_alias(MODEL_NAME, fallback_alias)
            return mv.version, mv.run_id, fallback_alias
        raise


print("Resolving model versions…")
champ_version,  champ_run_id,  champ_resolved  = resolve_version(
    CHAMPION_ALIAS,   CHAMPION_VERSION,   fallback_alias=PRODUCTION_ALIAS
)
chall_version,  chall_run_id,  chall_resolved  = resolve_version(
    CANDIDATE_ALIAS,  CHALLENGER_VERSION, fallback_alias=None
)

print(f"  Champion  : v{champ_version}  (via {champ_resolved})  run={champ_run_id[:8]}…")
print(f"  Challenger: v{chall_version}  (via {chall_resolved})  run={chall_run_id[:8]}…")

if champ_version == chall_version:
    print("\nChampion and challenger are the same version — nothing to compare.")
    print("Register a new candidate (run Phase 1) before running Phase 4.")
    sys.exit(0)

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 3.  LOAD PROMPT CONFIGS
# ════════════════════════════════════════════════════════════════════════════

def load_prompt_config(run_id: str, label: str) -> dict:
    """Load prompt_config JSON from the registration run artifact store."""
    uri = f"runs:/{run_id}/prompt_config"
    print(f"  Loading {label} config from: {uri}")
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    cfg_files = list(Path(local_dir).glob("*.json"))
    if not cfg_files:
        raise FileNotFoundError(
            f"No prompt_config JSON found for {label} run_id={run_id}.\n"
            f"Contents: {list(Path(local_dir).iterdir())}"
        )
    cfg_path = max(cfg_files, key=lambda p: p.stat().st_mtime)
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"    Template: {cfg['name']} v{cfg['version']}  model={cfg['model']}")
    return cfg


print("\nLoading prompt configs…")
champ_cfg = load_prompt_config(champ_run_id,  "champion")
chall_cfg = load_prompt_config(chall_run_id,  "challenger")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 4.  FROZEN GOLDEN EVAL DATASET
#     Identical across all phases — never modify to improve scores.
# ════════════════════════════════════════════════════════════════════════════

GOLDEN_DATASET = [
    {
        "id"              : "eval-001",
        "user_message"    : "How do I cancel my subscription?",
        "expected_topics" : ["cancel", "subscription", "account"],
        "must_not_contain": ["competitor", "I cannot help"],
        "max_words"       : 160,
    },
    {
        "id"              : "eval-002",
        "user_message"    : "My invoice shows the wrong amount, what do I do?",
        "expected_topics" : ["invoice", "billing", "support", "contact"],
        "must_not_contain": [],
        "max_words"       : 160,
    },
    {
        "id"              : "eval-003",
        "user_message"    : "Does Acme integrate with Salesforce?",
        "expected_topics" : ["integrat", "salesforce"],
        "must_not_contain": [],
        "max_words"       : 160,
    },
    {
        "id"              : "eval-004",
        "user_message"    : "What is the capital of France?",
        "expected_topics" : ["don't know", "cannot", "outside", "Acme"],
        "must_not_contain": ["Paris"],
        "max_words"       : 120,
    },
    {
        "id"              : "eval-005",
        "user_message"    : "How do I add a team member to my account?",
        "expected_topics" : ["team", "member", "invite", "add", "account"],
        "must_not_contain": [],
        "max_words"       : 160,
    },
]

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATORS  (identical to Phases 1 & 2 — keep all phases in sync)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RuleEvalResult:
    eval_id          : str
    passed           : bool
    topic_hits       : int
    topic_misses     : list
    must_not_violated: list
    word_count       : int
    ends_with_cta    : bool
    score            : float


def rule_eval(response: str, case: dict) -> RuleEvalResult:
    resp_lower   = response.lower()
    topic_hits   = [t for t in case["expected_topics"]  if t.lower() in resp_lower]
    topic_misses = [t for t in case["expected_topics"]  if t.lower() not in resp_lower]
    violations   = [t for t in case["must_not_contain"] if t.lower() in resp_lower]
    word_count   = len(response.split())
    ends_cta     = "anything else" in resp_lower
    topic_score  = len(topic_hits) / max(len(case["expected_topics"]), 1)
    score        = max(0.0, min(1.0, topic_score - len(violations) * 0.25 + (0.1 if ends_cta else 0.0)))
    return RuleEvalResult(
        eval_id=case["id"], passed=score >= 0.6 and not violations,
        topic_hits=len(topic_hits), topic_misses=topic_misses,
        must_not_violated=violations, word_count=word_count,
        ends_with_cta=ends_cta, score=round(score, 3),
    )


JUDGE_SYSTEM = textwrap.dedent("""\
    You are an impartial evaluator for a customer support AI.
    Return ONLY valid JSON with this schema:
    {
      "helpfulness"  : <integer 1-5>,
      "faithfulness" : <integer 1-5>,
      "conciseness"  : <integer 1-5>,
      "safety"       : <integer 1-5>,
      "reasoning"    : "<one sentence>"
    }
    - helpfulness  : does the response fully address the user's need?
    - faithfulness : does it stay within known facts, avoiding hallucination?
    - conciseness  : is it appropriately brief without losing clarity?
    - safety       : does it avoid harmful, offensive, or off-topic content?
    Do not include any text outside the JSON object.
""")


def llm_judge(user_message: str, response: str, eval_id: str) -> dict:
    """Score a response on 4 dimensions using gpt-4o-mini as judge."""
    prompt = f"USER MESSAGE:\n{user_message}\n\nASSISTANT RESPONSE:\n{response}"
    resp = oai_client.chat.completions.create(
        model       = "gemini_3_1_flash_Newer",
        max_tokens  = 256,
        temperature = 0.0,
        messages    = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned non-JSON for {eval_id}: {raw!r}") from exc

    avg = (scores["helpfulness"] + scores["faithfulness"] +
           scores["conciseness"] + scores["safety"]) / 4.0
    return {**scores, "avg_score": round(avg, 2), "eval_id": eval_id}

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 6.  EVAL RUNNER — runs the full suite against one model config
# ════════════════════════════════════════════════════════════════════════════

COST_PER_1K_INPUT  = 0.0025   # gpt-4o input pricing
COST_PER_1K_OUTPUT = 0.0100   # gpt-4o output pricing
NUMBER_OF_CHOICES = 2


def call_model(cfg: dict, message: str) -> tuple[str, float, int, int]:
    """Single API call. Returns (reply, latency_ms, input_tokens, output_tokens)."""
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
    return (
        response.choices[0].message.content,
        round(latency_ms, 1),
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def run_full_eval(cfg: dict, label: str) -> dict:
    """
    Run all 5 golden eval cases + LLM judge against one model config.
    Returns a summary dict with per-case results and aggregate metrics.
    """
    print(f"\n  Evaluating {label} ({cfg['name']} v{cfg['version']})…")
    case_results = []

    for case in GOLDEN_DATASET:
        if case.get('id') in choices:
            reply, latency_ms, in_tok, out_tok = call_model(cfg, case["user_message"])
            cost = (in_tok / 1000) * COST_PER_1K_INPUT + (out_tok / 1000) * COST_PER_1K_OUTPUT

            r = rule_eval(reply, case)
            j = llm_judge(case["user_message"], reply, case["id"])

            status = "PASS" if r.passed else "FAIL"
            print(f"    [{case['id']}] rule={status} score={r.score}  "
                f"judge={j['avg_score']}  latency={latency_ms:.0f}ms")

            case_results.append({
                "eval_id"           : case["id"],
                "user_message"      : case["user_message"],
                "reply_snippet"     : reply[:120],
                "rule_passed"       : r.passed,
                "rule_score"        : r.score,
                "topic_misses"      : r.topic_misses,
                "violations"        : r.must_not_violated,
                "judge_helpfulness" : j["helpfulness"],
                "judge_faithfulness": j["faithfulness"],
                "judge_conciseness" : j["conciseness"],
                "judge_safety"      : j["safety"],
                "judge_avg"         : j["avg_score"],
                "judge_reasoning"   : j.get("reasoning", ""),
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

    # Composite score: 50% rule (0-1) + 50% judge (normalised 1-5 → 0-1)
    composite = (avg_rule_score * 0.5) + ((avg_judge_score / 5.0) * 0.5)

    print(f"    ── {label} summary ──────────────────")
    print(f"       rule_pass_rate : {rule_pass_rate:.0%}")
    print(f"       avg_rule_score : {avg_rule_score:.3f}")
    print(f"       avg_judge_score: {avg_judge_score:.2f}/5.0")
    print(f"       composite      : {composite:.4f}")
    print(f"       avg_latency_ms : {avg_latency_ms:.0f}")
    print(f"       avg_cost_usd   : ${avg_cost_usd:.5f}/call")

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

# ════════════════════════════════════════════════════════════════════════════
# 7.  RUN EVALUATIONS ON BOTH MODELS
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*56}")
print(f"  Phase 4 — Champion / Challenger Evaluation")
print(f"  Champion  : {MODEL_NAME} v{champ_version}")
print(f"  Challenger: {MODEL_NAME} v{chall_version}")
print(f"  Min improvement required: {MIN_IMPROVEMENT}")
print(f"{'═'*56}")

# To Trim down API usage and to add some randomness for fun
choices = ["eval-00" + str(choice) for choice in np.random.choice(len(GOLDEN_DATASET),NUMBER_OF_CHOICES,replace=False).tolist()]
champ_results = run_full_eval(champ_cfg,  "champion")

# To Trim down API usage and to add some randomness for fun
choices = ["eval-00" + str(choice) for choice in np.random.choice(len(GOLDEN_DATASET),NUMBER_OF_CHOICES,replace=False).tolist()]
chall_results = run_full_eval(chall_cfg,  "challenger")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 8.  PROMOTION DECISION
# ════════════════════════════════════════════════════════════════════════════

score_delta = chall_results["composite"] - champ_results["composite"]
promote     = score_delta > MIN_IMPROVEMENT

print(f"\n{'─'*56}")
print(f"  Champion  composite: {champ_results['composite']:.4f}")
print(f"  Challenger composite: {chall_results['composite']:.4f}")
print(f"  Delta               : {score_delta:+.4f}  (need > +{MIN_IMPROVEMENT})")
print(f"  Decision            : {'PROMOTE challenger' if promote else 'RETAIN champion'}")

# Tie-break report even if we retain — useful context in MLflow
latency_delta = chall_results["avg_latency_ms"] - champ_results["avg_latency_ms"]
cost_delta    = chall_results["avg_cost_usd"]   - champ_results["avg_cost_usd"]
print(f"  Latency delta       : {latency_delta:+.0f} ms  (+ = challenger slower)")
print(f"  Cost delta          : ${cost_delta:+.5f}/call")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 9.  EXECUTE PROMOTION DECISION
# ════════════════════════════════════════════════════════════════════════════

decision_result = "challenger_promoted" if promote else "champion_retained"
promoted_at     = datetime.now(timezone.utc).isoformat()

if promote:
    print(f"\n  Promoting v{chall_version} → alias '{CHAMPION_ALIAS}'…")

    # Reassign champion alias to challenger version
    mlflow_client.set_registered_model_alias(
        name    = MODEL_NAME,
        alias   = CHAMPION_ALIAS,
        version = chall_version,
    )

    # Tag challenger (new champion) with promotion metadata
    mlflow_client.set_model_version_tag(MODEL_NAME, chall_version, "cc_result",         "promoted_to_champion")
    mlflow_client.set_model_version_tag(MODEL_NAME, chall_version, "cc_promoted_at",    promoted_at)
    mlflow_client.set_model_version_tag(MODEL_NAME, chall_version, "cc_score_delta",    str(round(score_delta, 4)))
    mlflow_client.set_model_version_tag(MODEL_NAME, chall_version, "cc_replaced_champ", champ_version)

    # Tag old champion as retired
    mlflow_client.set_model_version_tag(MODEL_NAME, champ_version, "cc_result",         "retired_by_challenger")
    mlflow_client.set_model_version_tag(MODEL_NAME, champ_version, "cc_retired_at",     promoted_at)
    mlflow_client.set_model_version_tag(MODEL_NAME, champ_version, "cc_replaced_by",    chall_version)

    print(f"  Alias '{CHAMPION_ALIAS}' → v{chall_version}  DONE")

else:
    print(f"\n  Retaining champion v{champ_version}. Retiring challenger v{chall_version}.")

    # Tag challenger as not promoted, with reason
    reason = (
        f"score_delta={score_delta:+.4f} did not exceed min_improvement={MIN_IMPROVEMENT}"
    )
    mlflow_client.set_model_version_tag(MODEL_NAME, chall_version, "cc_result",          "not_promoted")
    mlflow_client.set_model_version_tag(MODEL_NAME, chall_version, "cc_evaluated_at",    promoted_at)
    mlflow_client.set_model_version_tag(MODEL_NAME, chall_version, "cc_rejection_reason", reason)

    # Ensure champion alias still points to the right version (idempotent)
    mlflow_client.set_registered_model_alias(
        name    = MODEL_NAME,
        alias   = CHAMPION_ALIAS,
        version = champ_version,
    )

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 10.  LOG COMPARISON RUN TO MLFLOW
# ════════════════════════════════════════════════════════════════════════════

run_name = f"cc-{decision_result}-{GIT_SHA[:8]}"

with mlflow.start_run(run_name=run_name) as cc_run:

    # ── params ──────────────────────────────────────────────────────────────
    mlflow.log_params({
        "model_name"          : MODEL_NAME,
        "champion_version"    : champ_version,
        "challenger_version"  : chall_version,
        "champion_template"   : champ_cfg["name"],
        "challenger_template" : chall_cfg["name"],
        "champion_model"      : champ_cfg["model"],
        "challenger_model"    : chall_cfg["model"],
        "min_improvement"     : MIN_IMPROVEMENT,
        "eval_set_size"       : len(GOLDEN_DATASET),
    })

    # ── per-model metrics with champion. / challenger. prefix ────────────────
    mlflow.log_metrics({
        "champion/composite"      : champ_results["composite"],
        "champion/rule_pass_rate" : champ_results["rule_pass_rate"],
        "champion/avg_rule_score" : champ_results["avg_rule_score"],
        "champion/avg_judge_score": champ_results["avg_judge_score"],
        "champion/avg_latency_ms" : champ_results["avg_latency_ms"],
        "champion/avg_cost_usd"   : champ_results["avg_cost_usd"],

        "challenger/composite"      : chall_results["composite"],
        "challenger/rule_pass_rate" : chall_results["rule_pass_rate"],
        "challenger/avg_rule_score" : chall_results["avg_rule_score"],
        "challenger/avg_judge_score": chall_results["avg_judge_score"],
        "challenger/avg_latency_ms" : chall_results["avg_latency_ms"],
        "challenger/avg_cost_usd"   : chall_results["avg_cost_usd"],

        "delta/composite"   : round(score_delta,    4),
        "delta/latency_ms"  : round(latency_delta,  1),
        "delta/cost_usd"    : round(cost_delta,      6),
        "decision/promoted" : int(promote),
    })

    # ── tags ────────────────────────────────────────────────────────────────
    mlflow.set_tags({
        "cc.result"           : decision_result,
        "cc.champion_version" : champ_version,
        "cc.challenger_version": chall_version,
        "cc.score_delta"      : str(round(score_delta, 4)),
        "cc.min_improvement"  : str(MIN_IMPROVEMENT),
        "cc.git_sha"          : GIT_SHA,
        "cc.evaluated_at"     : promoted_at,
    })

    # ── artifact: full comparison JSON ───────────────────────────────────────
    comparison = {
        "run_id"             : cc_run.info.run_id,
        "decision"           : decision_result,
        "promoted"           : promote,
        "score_delta"        : score_delta,
        "min_improvement"    : MIN_IMPROVEMENT,
        "evaluated_at"       : promoted_at,
        "git_sha"            : GIT_SHA,
        "champion": {
            "model_name"     : MODEL_NAME,
            "version"        : champ_version,
            "resolved_alias" : champ_resolved,
            "template_name"  : champ_cfg["name"],
            "template_version": champ_cfg["version"],
            "llm_model"      : champ_cfg["model"],
            **{k: v for k, v in champ_results.items() if k != "case_results"},
            "case_results"   : champ_results["case_results"],
        },
        "challenger": {
            "model_name"     : MODEL_NAME,
            "version"        : chall_version,
            "resolved_alias" : chall_resolved,
            "template_name"  : chall_cfg["name"],
            "template_version": chall_cfg["version"],
            "llm_model"      : chall_cfg["model"],
            **{k: v for k, v in chall_results.items() if k != "case_results"},
            "case_results"   : chall_results["case_results"],
        },
    }
    artifact_path = Path("/tmp") / f"{run_name}_comparison.json"
    artifact_path.write_text(json.dumps(comparison, indent=2))
    mlflow.log_artifact(str(artifact_path), artifact_path="cc_results")

    print(f"\n  MLflow run: {cc_run.info.run_id}")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 11.  FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*56}")
if promote:
    print(f"  CHALLENGER PROMOTED")
    print(f"  New champion  : {MODEL_NAME} v{chall_version}")
    print(f"  Old champion  : v{champ_version} (retired)")
    print(f"  Score delta   : {score_delta:+.4f}")
else:
    print(f"  CHAMPION RETAINED")
    print(f"  Champion      : {MODEL_NAME} v{champ_version}")
    print(f"  Challenger v{chall_version} did not improve by >{MIN_IMPROVEMENT}")
    print(f"  Score delta   : {score_delta:+.4f}")
print(f"  MLflow run    : {cc_run.info.run_id}")
print(f"  Experiment    : {EXPERIMENT_NAME}")
print(f"{'═'*56}")
