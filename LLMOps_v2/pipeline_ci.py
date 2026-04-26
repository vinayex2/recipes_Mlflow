"""
llmops_core/pipeline_ci.py

Shared Phase 2 CI logic.
The entry-point notebook calls run_ci(cfg, client, ...) and gets back a
CIResult. All MLflow logging, gate evaluation, and model loading live here.
The notebook only handles Databricks widget reading and sys.exit().
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from .evaluators import rule_eval, RuleEvalResult
from .mlflow_helpers import log_ci_run
from .project_config import ProjectConfig


# ── Gate result ───────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    name      : str
    passed    : bool
    actual    : float
    threshold : float
    direction : str   # ">=" or "<="

    def __str__(self) -> str:
        sym = "PASS" if self.passed else "FAIL"
        return (
            f"  [{sym}] {self.name:<28} "
            f"actual={self.actual:.4f}  {self.direction}  threshold={self.threshold:.4f}"
        )


# ── CI result ─────────────────────────────────────────────────────────────────

@dataclass
class CIResult:
    all_gates_passed : bool
    gates            : list[GateResult]
    mlflow_run_id    : str


# ── Main entrypoint ───────────────────────────────────────────────────────────

def run_ci(
    cfg               : ProjectConfig,
    client,
    candidate_version : str,
    candidate_run_id  : str,
    template_cfg      : dict,
    git_sha           : str  = "local",
    git_ref           : str  = "local",
    git_pr            : str  = "",
) -> CIResult:
    """
    Run the full CI evaluation loop against the frozen golden dataset.
    Logs results to MLflow and returns a CIResult.
    """
    dataset = cfg.golden_dataset
    n       = len(dataset)
    print(f"\nRunning {n} eval cases…\n")

    results      : list[dict]           = []
    rule_results : list[RuleEvalResult] = []

    for case in dataset:
        reply, latency_ms, in_tok, out_tok = _call_model(client, template_cfg)
        cost_usd = cfg.cost.estimate(in_tok, out_tok)
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

    # ── aggregate ─────────────────────────────────────────────────────────────
    rule_pass_rate = sum(r.passed for r in rule_results) / n
    avg_rule_score = sum(r.score  for r in rule_results) / n
    avg_latency_ms = sum(r["latency_ms"]  for r in results) / n
    avg_cost_usd   = sum(r["cost_usd"]    for r in results) / n
    total_in_tok   = sum(r["input_tokens"] for r in results)
    total_out_tok  = sum(r["output_tokens"] for r in results)

    print(f"\n{'─'*54}")
    print(f"  rule_pass_rate : {rule_pass_rate:.0%}  ({sum(r.passed for r in rule_results)}/{n})")
    print(f"  avg_rule_score : {avg_rule_score:.3f}")
    print(f"  avg_latency_ms : {avg_latency_ms:.0f} ms")
    print(f"  avg_cost_usd   : ${avg_cost_usd:.5f} / call")

    # ── gates ─────────────────────────────────────────────────────────────────
    g = cfg.gates
    gates = [
        GateResult("rule_pass_rate", rule_pass_rate >= g.ci_rule_pass_rate, rule_pass_rate, g.ci_rule_pass_rate, ">="),
        GateResult("avg_latency_ms", avg_latency_ms <= g.ci_avg_latency_ms, avg_latency_ms, g.ci_avg_latency_ms, "<="),
        GateResult("avg_cost_usd",   avg_cost_usd   <= g.ci_avg_cost_usd,   avg_cost_usd,   g.ci_avg_cost_usd,   "<="),
    ]
    all_gates_passed = all(gate.passed for gate in gates)

    print(f"\nQuality gate results:")
    for gate in gates:
        print(gate)

    # ── MLflow ────────────────────────────────────────────────────────────────
    run_name = f"ci-{git_sha[:8]}-{'pass' if all_gates_passed else 'fail'}"
    ci_metrics = {
        "ci/rule_pass_rate" : round(rule_pass_rate, 4),
        "ci/avg_rule_score" : round(avg_rule_score, 4),
        "ci/avg_latency_ms" : round(avg_latency_ms, 1),
        "ci/avg_cost_usd"   : round(avg_cost_usd,   6),
        "ci/total_in_tokens": total_in_tok,
        "ci/total_out_tokens":total_out_tok,
        "ci/gates_passed"   : int(all_gates_passed),
        "ci/n_gates_passed" : sum(gate.passed for gate in gates),
        "ci/n_gates_total"  : len(gates),
        **{f"ci/gate_{gate.name}": int(gate.passed) for gate in gates},
    }

    with mlflow.start_run(run_name=run_name) as ci_run:
        log_ci_run(
            run               = ci_run,
            run_name          = run_name,
            model_name        = cfg.mlflow_model_name,
            candidate_version = candidate_version,
            candidate_run_id  = candidate_run_id,
            template_cfg      = template_cfg,
            eval_results      = results,
            gates             = gates,
            metrics           = ci_metrics,
            git_sha           = git_sha,
            git_ref           = git_ref,
            git_pr            = git_pr,
            all_gates_passed  = all_gates_passed,
            n_eval_cases      = n,
        )
        run_id = ci_run.info.run_id

    print(f"\nMLflow CI run logged: {run_id}")
    return CIResult(
        all_gates_passed = all_gates_passed,
        gates            = gates,
        mlflow_run_id    = run_id,
    )


# ── Internal model call ───────────────────────────────────────────────────────

def _call_model(
    client       : object,
    template_cfg : dict,
    user_message : str,
) -> tuple[str, float, int, int]:
    """Single inference call. Returns (reply, latency_ms, in_tok, out_tok)."""
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
    latency_ms    = (time.perf_counter() - t0) * 1000
    reply         = response.choices[0].message.content
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return reply, round(latency_ms, 1), input_tokens, output_tokens
