# Databricks notebook source
# llmops_phase2_ci.py
#
# LLMOps Phase 2 — Continuous Integration notebook
#
# Upload this file to your Databricks Free Edition workspace at:
#   /Shared/llmops/llmops_phase2_ci
#
# This notebook runs on SERVERLESS compute (the only option in Free Edition).
# It is triggered by the GitHub Actions workflow via the Databricks Runs Submit
# API and can also be run manually from the Databricks UI for debugging.
#
# What it does:
#   1. Installs dependencies via %pip (task-level libraries not supported on serverless)
#   2. Reads CI context (git SHA, PR number) from notebook widget parameters
#   3. Loads the registered candidate model from MLflow Model Registry
#   4. Runs the same frozen golden eval dataset used in Phase 1
#   5. Applies three quality gates: rule pass rate, latency, cost
#   6. Logs the full CI run back to MLflow with a ci/ tag namespace
#   7. Exits with sys.exit(1) on any gate failure so the Runs API
#      reports result_state=FAILED, which the GitHub Actions poller
#      translates into a failed PR check.

# COMMAND ----------

# Install dependencies via %pip — this is the only supported install method
# for notebook tasks on serverless compute in Databricks Free Edition.
# %pip must be in the first command cell; it restarts the Python kernel.

%pip install -q openai>=1.30.0 tiktoken>=0.7.0

# COMMAND ----------


# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import os
import sys
import time
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── third-party ───────────────────────────────────────────────────────────────
import mlflow
import mlflow.pyfunc
from openai import OpenAI

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 0.  NOTEBOOK PARAMETERS
#     Read CI context injected by the GitHub Actions workflow.
#     When running manually from the UI, defaults are used.
# ════════════════════════════════════════════════════════════════════════════

# dbutils is available in the Databricks runtime; provide fallbacks for
# running the file locally outside Databricks.
try:
    dbutils  # noqa: F821 — injected by Databricks runtime
    _in_databricks = True
except NameError:
    _in_databricks = False
    class _FakeWidgets:
        def get(self, name, default=""): return default
    class _FakeDbutils:
        widgets = _FakeWidgets()
    dbutils = _FakeDbutils()  # noqa: F841

%python
GIT_SHA = dbutils.widgets.get("git_sha") if _in_databricks else "local"
GIT_REF = dbutils.widgets.get("git_ref") if _in_databricks else "local"
GIT_PR = dbutils.widgets.get("git_pr") if _in_databricks else ""


# OPENAI_API_KEY is passed as a widget parameter (secret) from GitHub Actions.
# In a production workspace you would use Databricks Secrets instead:
# In a production workspace you would use Databricks Secrets instead:
#   DATABRICKS_TOKEN = dbutils.secrets.get(scope="llmops", key="DATABRICKS_TOKEN")
DATABRICKS_TOKEN = dbutils.widgets.get("DATABRICKS_TOKEN") if _in_databricks else os.environ.get("DATABRICKS_TOKEN", "")

if not DATABRICKS_TOKEN:
    raise ValueError(
        "DATABRICKS_TOKEN not found. Pass it as a widget parameter or set the "
        "env var. In production use dbutils.secrets."
    )

print(f"CI context  — SHA={GIT_SHA[:12]}  ref={GIT_REF}  PR={GIT_PR or 'none'}")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 1.  MLFLOW + OPENAI SETUP
# ════════════════════════════════════════════════════════════════════════════

# On Databricks, MLflow is pre-configured to point at the workspace tracking
# server. No MLFLOW_TRACKING_URI needed — mlflow.set_experiment() just works.
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "llmops/phase2-ci")
MODEL_NAME      = os.getenv("MLFLOW_MODEL_NAME",      "llmops/support-agent")
MODEL_ALIAS     = os.getenv("MLFLOW_MODEL_ALIAS",     "candidate")

mlflow.set_experiment(EXPERIMENT_NAME)

client = OpenAI(
    api_key  = DATABRICKS_TOKEN,
    base_url = os.getenv("GEMINI_ENDPOINT"),
)

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 2.  QUALITY GATE THRESHOLDS
#     Mirror these in your team's runbook. Changes here = changes to what
#     "passing CI" means, so treat them like code and version-control them.
# ════════════════════════════════════════════════════════════════════════════

GATE_RULE_PASS_RATE_MIN = float(os.getenv("CI_GATE_RULE_PASS_RATE", "0.80"))  # 80 %
GATE_AVG_LATENCY_MS_MAX = float(os.getenv("CI_GATE_LATENCY_MS",     "8000"))  # 8 s
GATE_AVG_COST_USD_MAX   = float(os.getenv("CI_GATE_COST_USD",       "0.05"))  # $0.05/call

# Approximate cost per 1K tokens for gpt-4o (update when pricing changes)
COST_PER_1K_INPUT_USD  = 0.0025
COST_PER_1K_OUTPUT_USD = 0.0100

print(f"Gate thresholds:")
print(f"  rule_pass_rate >= {GATE_RULE_PASS_RATE_MIN:.0%}")
print(f"  avg_latency_ms <= {GATE_AVG_LATENCY_MS_MAX:.0f} ms")
print(f"  avg_cost_usd   <= ${GATE_AVG_COST_USD_MAX:.4f} / call")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 3.  FROZEN GOLDEN EVAL DATASET
#     Identical to the dataset in Phase 1 — never modify this to chase a
#     passing score.  This is your fixed measurement stick.
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
# 4.  RULE-BASED EVALUATOR  (identical to Phase 1 — keep them in sync)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RuleEvalResult:
    eval_id          : str
    passed           : bool
    topic_hits       : int
    topic_misses     : list
    must_not_violated: list
    word_count       : int
    within_word_limit: bool
    ends_with_cta    : bool
    score            : float


def rule_eval(response: str, case: dict) -> RuleEvalResult:
    resp_lower = response.lower()
    topic_hits   = [t for t in case["expected_topics"]  if t.lower() in resp_lower]
    topic_misses = [t for t in case["expected_topics"]  if t.lower() not in resp_lower]
    violations   = [t for t in case["must_not_contain"] if t.lower() in resp_lower]
    word_count        = len(response.split())
    within_word_limit = word_count <= case["max_words"]
    ends_with_cta     = "anything else" in resp_lower
    topic_score  = len(topic_hits) / max(len(case["expected_topics"]), 1)
    penalty      = len(violations) * 0.25
    format_bonus = 0.1 if ends_with_cta else 0.0
    score        = max(0.0, min(1.0, topic_score - penalty + format_bonus))
    return RuleEvalResult(
        eval_id=case["id"], passed=score >= 0.6 and not violations,
        topic_hits=len(topic_hits), topic_misses=topic_misses,
        must_not_violated=violations, word_count=word_count,
        within_word_limit=within_word_limit, ends_with_cta=ends_with_cta,
        score=round(score, 3),
    )

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 5.  LOAD CANDIDATE MODEL FROM MLFLOW REGISTRY
#     We load the prompt template config from the registered model's artifact
#     rather than using mlflow.pyfunc.load_model() so we can call the OpenAI
#     API directly and capture raw latency + token counts for gate checks.
# ════════════════════════════════════════════════════════════════════════════

def load_candidate_template() -> dict:
    """
    Download the prompt_config artifact from the candidate model version
    and return it as a dict.  We deliberately avoid mlflow.pyfunc.load_model()
    here so we retain fine-grained control over the API call (timing, usage).
    """
    client_mlflow = mlflow.MlflowClient()

    # Resolve the 'candidate' alias to a concrete version
    try:
        mv = client_mlflow.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(
            f"Could not find model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'. "
            f"Run Phase 1 first to register a candidate.\n{exc}"
        ) from exc

    version    = mv.version
    run_id     = mv.run_id
    print(f"Loaded candidate: {MODEL_NAME} v{version}  (run_id={run_id[:8]}…)")

    # Download the prompt_config JSON artifact
    artifact_dir = mlflow.artifacts.download_artifacts(
        run_id        = run_id,
        artifact_path = "prompt_config",
    )
    cfg_files = list(Path(artifact_dir).glob("*.json"))
    if not cfg_files:
        raise FileNotFoundError(
            f"No prompt_config JSON found in artifacts for run {run_id}. "
            "Ensure Phase 1 logged the template artifact correctly."
        )
    with open(cfg_files[0]) as f:
        template_cfg = json.load(f)

    print(f"Template: {template_cfg['name']} v{template_cfg['version']}")
    return template_cfg, version, run_id


template_cfg, candidate_version, candidate_run_id = load_candidate_template()

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 6.  CI EVALUATION LOOP
#     Call the model once per eval case, measure latency + cost, apply rules.
# ════════════════════════════════════════════════════════════════════════════

def call_model(template: dict, user_message: str) -> tuple[str, float, int, int]:
    """
    Single inference call.
    Returns (reply, latency_ms, input_tokens, output_tokens).
    """
    messages = [{"role": "system", "content": template["system"]}]
    messages.extend(template.get("few_shots", []))
    messages.append({"role": "user", "content": user_message})

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model       = template["model"],
        max_tokens  = template["max_tokens"],
        temperature = template["temperature"],
        messages    = messages,
    )
    latency_ms   = (time.perf_counter() - t0) * 1000
    reply        = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens= response.usage.completion_tokens
    return reply, round(latency_ms, 1), input_tokens, output_tokens


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens  / 1000) * COST_PER_1K_INPUT_USD +
        (output_tokens / 1000) * COST_PER_1K_OUTPUT_USD
    )


print(f"\nRunning {len(GOLDEN_DATASET)} eval cases…\n")

results      = []
rule_results = []

for case in GOLDEN_DATASET:
    reply, latency_ms, in_tok, out_tok = call_model(template_cfg, case["user_message"])
    cost_usd = estimate_cost(in_tok, out_tok)
    r        = rule_eval(reply, case)
    rule_results.append(r)

    status = "PASS" if r.passed else "FAIL"
    print(
        f"  [{case['id']}] {status}  score={r.score}  "
        f"latency={latency_ms:.0f}ms  cost=${cost_usd:.5f}"
    )
    if r.topic_misses:
        print(f"           missing topics: {r.topic_misses}")
    if r.must_not_violated:
        print(f"           VIOLATIONS    : {r.must_not_violated}")

    results.append({
        "eval_id"           : case["id"],
        "user_message"      : case["user_message"],
        "assistant_response": reply,
        "rule_passed"       : r.passed,
        "rule_score"        : r.score,
        "topic_misses"      : r.topic_misses,
        "violations"        : r.must_not_violated,
        "latency_ms"        : latency_ms,
        "input_tokens"      : in_tok,
        "output_tokens"     : out_tok,
        "cost_usd"          : round(cost_usd, 6),
    })

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 7.  AGGREGATE METRICS
# ════════════════════════════════════════════════════════════════════════════

n = len(GOLDEN_DATASET)

rule_pass_rate = sum(r.passed for r in rule_results) / n
avg_rule_score = sum(r.score  for r in rule_results) / n
avg_latency_ms = sum(r["latency_ms"]   for r in results) / n
avg_cost_usd   = sum(r["cost_usd"]     for r in results) / n
total_in_tok   = sum(r["input_tokens"] for r in results)
total_out_tok  = sum(r["output_tokens"]for r in results)

print(f"\n{'─'*54}")
print(f"  rule_pass_rate : {rule_pass_rate:.0%}  ({sum(r.passed for r in rule_results)}/{n})")
print(f"  avg_rule_score : {avg_rule_score:.3f}")
print(f"  avg_latency_ms : {avg_latency_ms:.0f} ms")
print(f"  avg_cost_usd   : ${avg_cost_usd:.5f} / call")
print(f"  total_tokens   : {total_in_tok} in / {total_out_tok} out")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 8.  QUALITY GATE CHECKS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GateResult:
    name   : str
    passed : bool
    actual : float
    threshold: float
    direction: str   # ">=" or "<="

    def __str__(self):
        op  = self.direction
        sym = "PASS" if self.passed else "FAIL"
        return (
            f"  [{sym}] {self.name:<28} "
            f"actual={self.actual:.4f}  {op}  threshold={self.threshold:.4f}"
        )


gates = [
    GateResult(
        name      = "rule_pass_rate",
        passed    = rule_pass_rate >= GATE_RULE_PASS_RATE_MIN,
        actual    = rule_pass_rate,
        threshold = GATE_RULE_PASS_RATE_MIN,
        direction = ">=",
    ),
    GateResult(
        name      = "avg_latency_ms",
        passed    = avg_latency_ms <= GATE_AVG_LATENCY_MS_MAX,
        actual    = avg_latency_ms,
        threshold = GATE_AVG_LATENCY_MS_MAX,
        direction = "<=",
    ),
    GateResult(
        name      = "avg_cost_usd",
        passed    = avg_cost_usd <= GATE_AVG_COST_USD_MAX,
        actual    = avg_cost_usd,
        threshold = GATE_AVG_COST_USD_MAX,
        direction = "<=",
    ),
]

all_gates_passed = all(g.passed for g in gates)

print(f"\nQuality gate results:")
for g in gates:
    print(g)
print()

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 9.  LOG CI RUN TO MLFLOW
# ════════════════════════════════════════════════════════════════════════════

ci_run_name = (
    f"ci-{GIT_SHA[:8]}-"
    f"{'pass' if all_gates_passed else 'fail'}"
)

with mlflow.start_run(run_name=ci_run_name) as ci_run:

    # ── params ──────────────────────────────────────────────────────────────
    mlflow.log_params({
        "candidate_model_name"   : MODEL_NAME,
        "candidate_model_version": candidate_version,
        "candidate_run_id"       : candidate_run_id,
        "template_name"          : template_cfg["name"],
        "template_version"       : template_cfg["version"],
        "model"                  : template_cfg["model"],
        "eval_set_size"          : n,
    })

    # ── metrics ─────────────────────────────────────────────────────────────
    mlflow.log_metrics({
        "ci/rule_pass_rate"  : round(rule_pass_rate, 4),
        "ci/avg_rule_score"  : round(avg_rule_score, 4),
        "ci/avg_latency_ms"  : round(avg_latency_ms, 1),
        "ci/avg_cost_usd"    : round(avg_cost_usd,   6),
        "ci/total_in_tokens" : total_in_tok,
        "ci/total_out_tokens": total_out_tok,
        "ci/gates_passed"    : int(all_gates_passed),
        "ci/n_gates_passed"  : sum(g.passed for g in gates),
        "ci/n_gates_total"   : len(gates),
        # individual gate verdicts as 0/1 metrics for easy filtering in UI
        **{f"ci/gate_{g.name}": int(g.passed) for g in gates},
    })

    # ── tags ────────────────────────────────────────────────────────────────
    mlflow.set_tags({
        "ci.result"              : "PASS" if all_gates_passed else "FAIL",
        "ci.git_sha"             : GIT_SHA,
        "ci.git_ref"             : GIT_REF,
        "ci.git_pr"              : GIT_PR or "none",
        "ci.run_date"            : datetime.utcnow().strftime("%Y-%m-%d"),
        "candidate.model_name"   : MODEL_NAME,
        "candidate.model_version": candidate_version,
    })

    # ── artifact: full per-case results ─────────────────────────────────────
    artifact = {
        "ci_run_id"    : ci_run.info.run_id,
        "ci_run_name"  : ci_run_name,
        "git_sha"      : GIT_SHA,
        "git_ref"      : GIT_REF,
        "all_passed"   : all_gates_passed,
        "gates"        : [
            {
                "name"     : g.name,
                "passed"   : g.passed,
                "actual"   : g.actual,
                "threshold": g.threshold,
                "direction": g.direction,
            }
            for g in gates
        ],
        "eval_results" : results,
        "timestamp"    : datetime.utcnow().isoformat(),
    }
    artifact_path = Path("/tmp") / f"{ci_run_name}_ci_results.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    mlflow.log_artifact(str(artifact_path), artifact_path="ci_results")

    print(f"\nMLflow CI run logged: {ci_run.info.run_id}")

# ── If on Databricks, also update the model version tag ─────────────────────
if _in_databricks:
    try:
        mlflow_client = mlflow.MlflowClient()
        mlflow_client.set_model_version_tag(
            name    = MODEL_NAME,
            version = candidate_version,
            key     = "ci.last_result",
            value   = "PASS" if all_gates_passed else "FAIL",
        )
        mlflow_client.set_model_version_tag(
            name    = MODEL_NAME,
            version = candidate_version,
            key     = "ci.last_run_id",
            value   = ci_run.info.run_id,
        )
        print(f"Model version {candidate_version} tagged with ci.last_result="
              f"{'PASS' if all_gates_passed else 'FAIL'}")
    except Exception as exc:
        print(f"Warning: could not tag model version — {exc}")

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════
# 10.  FINAL VERDICT
#      Print CI_RESULT in the format the GitHub Actions poller expects,
#      then exit with a non-zero code on failure so the Databricks Run
#      reports result_state=FAILED.
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*54}")
if all_gates_passed:
    print("CI_RESULT: PASS")
    print("All quality gates passed. PR is unblocked.")
else:
    failed = [g.name for g in gates if not g.passed]
    print(f"CI_RESULT: FAIL  reason=gates_failed:{','.join(failed)}")
    print(f"Failed gates: {failed}")
    print("PR is blocked. Fix the prompt template and re-run Phase 1.")
    sys.exit(1)
