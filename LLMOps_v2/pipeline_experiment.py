"""
llmops_core/pipeline_experiment.py

Shared Phase 1 logic: multi-turn conversation engine, per-template experiment
runner, and model registration.

Entry point: run_all_experiments(cfg, client) → list[ExperimentSummary]
             register_best_candidate(cfg, summaries, client, model_code_path)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from .evaluators     import rule_eval, llm_judge, RuleEvalResult, JudgeResult
from .project_config import ProjectConfig, TemplateConfig


# ── Conversation engine ───────────────────────────────────────────────────────

@dataclass
class Turn:
    user_message       : str
    assistant_response : str
    latency_ms         : float
    input_tokens       : int
    output_tokens      : int


class Conversation:
    """
    Stateful multi-turn conversation manager.

    Prepends the system prompt and any few-shot examples before the live
    conversation history on every call so the model always has full context.

    Usage:
        conv  = Conversation(template, client)
        reply = conv.chat("How do I cancel?")
        conv.reset()   # clear history between eval cases
    """

    def __init__(self, template: TemplateConfig, client):
        self.template = template
        self.client   = client
        self.history  : list[dict] = []
        self.turns    : list[Turn] = []

    def _build_messages(self, user_message: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.template.system}]
        msgs.extend(self.template.few_shots)
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def chat(self, user_message: str) -> str:
        messages = self._build_messages(user_message)
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model       = self.template.model,
            max_tokens  = self.template.max_tokens,
            temperature = self.template.temperature,
            messages    = messages,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        reply = response.choices[0].message.content

        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        self.turns.append(Turn(
            user_message       = user_message,
            assistant_response = reply,
            latency_ms         = round(latency_ms, 1),
            input_tokens       = response.usage.prompt_tokens,
            output_tokens      = response.usage.completion_tokens,
        ))
        return reply

    def reset(self):
        self.history.clear()
        self.turns.clear()

    def total_tokens(self) -> dict:
        return {
            "input" : sum(t.input_tokens  for t in self.turns),
            "output": sum(t.output_tokens for t in self.turns),
            "total" : sum(t.input_tokens + t.output_tokens for t in self.turns),
        }

    def avg_latency_ms(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.latency_ms for t in self.turns) / len(self.turns)


# ── Experiment summary ────────────────────────────────────────────────────────

@dataclass
class ExperimentSummary:
    run_id          : str
    run_name        : str
    template_name   : str
    rule_pass_rate  : float
    avg_rule_score  : float
    avg_latency_ms  : float
    avg_judge_score : float = 0.0   # 0 means judge was not run

    def composite(self) -> float:
        """50% rule score + 50% normalised judge score (or 100% rule if no judge)."""
        if self.avg_judge_score:
            return (self.avg_rule_score * 0.5) + ((self.avg_judge_score / 5.0) * 0.5)
        return self.avg_rule_score


# ── Per-template experiment runner ────────────────────────────────────────────

def run_experiment(
    template   : TemplateConfig,
    cfg        : ProjectConfig,
    client,
    run_judge  : bool = True,
    run_name   : Optional[str] = None,
) -> ExperimentSummary:
    """
    Run one template through the full golden eval dataset.
    Logs params, metrics, and artifacts to the active MLflow experiment.
    Returns an ExperimentSummary for comparison and registration.
    """
    chash    = _config_hash(template)
    run_name = run_name or f"{template.name}-v{template.version}-{chash}"

    print(f"\n{'═'*60}")
    print(f"  Experiment: {run_name}")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "template.name"       : template.name,
            "template.version"    : template.version,
            "template.config_hash": chash,
            "model"               : template.model,
            "temperature"         : template.temperature,
            "max_tokens"          : template.max_tokens,
            "few_shot_count"      : len(template.few_shots),
        })
        mlflow.set_tags({
            "template.name"   : template.name,
            "template.version": template.version,
            "eval_set_size"   : str(len(cfg.golden_dataset)),
            "judge_enabled"   : str(run_judge),
            "run_date"        : datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        })

        rule_results  : list[RuleEvalResult] = []
        judge_results : list[JudgeResult]    = []
        turn_records  : list[dict]           = []

        for case in cfg.golden_dataset:
            print(f"\n  [{case['id']}] {case['user_message'][:60]}…")
            conv  = Conversation(template=template, client=client)
            reply = conv.chat(case["user_message"])
            turn  = conv.turns[0]

            print(f"           → {reply[:80].replace(chr(10), ' ')}…")
            print(f"             latency={turn.latency_ms:.0f}ms  "
                  f"tokens={turn.input_tokens}in/{turn.output_tokens}out")

            r = rule_eval(reply, case)
            rule_results.append(r)
            print(f"             rule_eval: {'PASS' if r.passed else 'FAIL'}  score={r.score}")
            if r.topic_misses:
                print(f"             missing: {r.topic_misses}")

            record: dict = {
                "eval_id"           : case["id"],
                "user_message"      : case["user_message"],
                "assistant_response": reply,
                "latency_ms"        : turn.latency_ms,
                "input_tokens"      : turn.input_tokens,
                "output_tokens"     : turn.output_tokens,
                "rule_score"        : r.score,
                "rule_passed"       : r.passed,
                "topic_misses"      : r.topic_misses,
                "violations"        : r.must_not_violated,
            }

            if run_judge:
                j = llm_judge(
                    user_message = case["user_message"],
                    response     = reply,
                    eval_id      = case["id"],
                    client       = client,
                    judge_model  = cfg.judge_model,
                )
                judge_results.append(j)
                print(f"             llm_judge: avg={j.avg_score}  {j.reasoning}")
                record.update({
                    "judge_helpfulness" : j.helpfulness,
                    "judge_faithfulness": j.faithfulness,
                    "judge_conciseness" : j.conciseness,
                    "judge_safety"      : j.safety,
                    "judge_avg"         : j.avg_score,
                    "judge_reasoning"   : j.reasoning,
                })

            turn_records.append(record)

        # ── aggregate metrics ──────────────────────────────────────────────────
        n              = len(cfg.golden_dataset)
        rule_pass_rate = sum(r.passed for r in rule_results) / n
        avg_rule_score = sum(r.score  for r in rule_results) / n
        avg_latency    = sum(r["latency_ms"]    for r in turn_records) / n
        avg_input_tok  = sum(r["input_tokens"]  for r in turn_records) / n
        avg_output_tok = sum(r["output_tokens"] for r in turn_records) / n

        metrics = {
            "eval/rule_pass_rate"   : round(rule_pass_rate, 3),
            "eval/avg_rule_score"   : round(avg_rule_score, 3),
            "perf/avg_latency_ms"   : round(avg_latency,    1),
            "perf/avg_input_tokens" : round(avg_input_tok,  1),
            "perf/avg_output_tokens": round(avg_output_tok, 1),
        }

        avg_judge = 0.0
        if run_judge and judge_results:
            avg_judge = sum(j.avg_score for j in judge_results) / n
            metrics.update({
                "eval/judge_avg_score"    : round(avg_judge, 2),
                "eval/judge_helpfulness"  : round(sum(j.helpfulness  for j in judge_results) / n, 2),
                "eval/judge_faithfulness" : round(sum(j.faithfulness for j in judge_results) / n, 2),
                "eval/judge_conciseness"  : round(sum(j.conciseness  for j in judge_results) / n, 2),
                "eval/judge_safety"       : round(sum(j.safety       for j in judge_results) / n, 2),
            })

        mlflow.log_metrics(metrics)

        # ── artifacts ─────────────────────────────────────────────────────────
        template_dict = _template_to_dict(template)
        _write_artifact(
            {"run_id": run.info.run_id, "run_name": run_name, "template": template_dict,
             "timestamp": datetime.now(timezone.utc).isoformat(), "turn_records": turn_records},
            f"{run_name}_results.json", "eval_results",
        )
        _write_artifact(template_dict, f"{run_name}_template.json", "prompt_config")

        print(f"\n  Run ID    : {run.info.run_id}")
        print(f"  Rule pass : {rule_pass_rate:.0%}  ({sum(r.passed for r in rule_results)}/{n})")
        if avg_judge:
            print(f"  Judge avg : {avg_judge:.2f}/5.0")
        print(f"  Latency   : {avg_latency:.0f} ms")

        return ExperimentSummary(
            run_id         = run.info.run_id,
            run_name       = run_name,
            template_name  = template.name,
            rule_pass_rate = rule_pass_rate,
            avg_rule_score = avg_rule_score,
            avg_latency_ms = avg_latency,
            avg_judge_score= avg_judge,
        )


# ── Run all templates ─────────────────────────────────────────────────────────

def run_all_experiments(
    cfg       : ProjectConfig,
    client,
    run_judge : bool = True,
) -> list[ExperimentSummary]:
    """Evaluate every template in cfg.templates and return all summaries."""
    summaries = []
    for template in cfg.templates:
        summary = run_experiment(template, cfg, client, run_judge=run_judge)
        summaries.append(summary)

    # ── comparison table ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {'Template':<28}  {'Rule%':>6}  {'Rule↑':>6}  {'Judge':>6}  {'ms':>6}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for s in summaries:
        judge_str = f"{s.avg_judge_score:>5.2f}" if s.avg_judge_score else "  N/A"
        print(
            f"  {s.template_name:<28}  "
            f"{s.rule_pass_rate:>5.0%}  "
            f"{s.avg_rule_score:>6.3f}  "
            f"{judge_str}  "
            f"{s.avg_latency_ms:>5.0f}"
        )

    return summaries


# ── Registration ──────────────────────────────────────────────────────────────

def register_best_candidate(
    cfg             : ProjectConfig,
    summaries       : list[ExperimentSummary],
    model_code_path : str | Path,
) -> Optional[str]:
    """
    Pick the highest-scoring template, check it against cfg.score_threshold,
    and register it in the MLflow Model Registry with alias 'candidate'.

    model_code_path must point to a prompt_model.py that is fully self-contained
    (MLflow loads it in a fresh interpreter at serve time).

    Returns the registered model URI, or None if threshold not met.
    """
    if not summaries:
        print("No summaries to register.")
        return None

    ranked      = sorted(summaries, key=lambda s: s.composite(), reverse=True)
    best        = ranked[0]
    best_score  = best.composite()
    best_template = next(t for t in cfg.templates if t.name == best.template_name)

    print(f"\n{'═'*60}")
    print(f"  Best template  : {best.template_name}")
    print(f"  Composite score: {best_score:.3f}  (threshold: {cfg.score_threshold})")

    if best_score < cfg.score_threshold:
        print("  Below threshold — not registering.")
        return None

    print("  Registering in MLflow Model Registry…")

    model_code_path = Path(model_code_path)
    if not model_code_path.exists():
        raise FileNotFoundError(
            f"prompt_model.py not found at {model_code_path}. "
            "It must be a fully self-contained pyfunc model file."
        )

    chash        = _config_hash(best_template)
    cfg_path     = Path("/tmp") / f"prompt_config_{chash}.json"
    template_dict = _template_to_dict(best_template)
    cfg_path.write_text(json.dumps(template_dict, indent=2))

    with mlflow.start_run(run_name=f"register-{best.template_name}") as reg_run:
        mlflow.log_params({
            "template.name"       : best_template.name,
            "template.version"    : best_template.version,
            "template.config_hash": chash,
            "model"               : best_template.model,
            "temperature"         : best_template.temperature,
            "max_tokens"          : best_template.max_tokens,
            "few_shot_count"      : len(best_template.few_shots),
        })
        mlflow.log_metrics({
            "composite_score": round(best_score, 3),
            "rule_pass_rate" : round(best.rule_pass_rate, 3),
        })
        mlflow.set_tags({
            "stage"                : "candidate",
            "template.name"        : best_template.name,
            "template.config_hash" : chash,
        })

        # log the config as a run artifact so Phase 2 can download it via runs:/
        mlflow.log_artifact(str(cfg_path), artifact_path="prompt_config")
        print(f"  Logged prompt_config to run {reg_run.info.run_id[:8]}…")

        model_info = mlflow.pyfunc.log_model(
            name                  = "prompt_model",
            python_model          = str(model_code_path),
            artifacts             = {"prompt_config": str(cfg_path)},
            registered_model_name = cfg.mlflow_model_name,
            pip_requirements      = ["openai", "mlflow"],
            signature             = ModelSignature(
                inputs  = Schema([ColSpec(type="string", name="message")]),
                outputs = Schema([ColSpec(type="string", name="response")]),
            ),
        )

    mlflow_client = mlflow.client.MlflowClient()
    latest = mlflow_client.search_model_versions(
        f"name='{cfg.mlflow_model_name}'"
    )[0].version
    mlflow_client.set_registered_model_alias(cfg.mlflow_model_name, "candidate", latest)

    uri = model_info.model_uri
    print(f"  Registered : {uri}")
    print(f"  Model name : {cfg.mlflow_model_name}")
    print(f"  Alias      : candidate → v{latest}")
    return uri


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config_hash(template: TemplateConfig) -> str:
    payload = json.dumps(_template_to_dict(template), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _template_to_dict(template: TemplateConfig) -> dict:
    return {
        "name"       : template.name,
        "version"    : template.version,
        "system"     : template.system,
        "model"      : template.model,
        "temperature": template.temperature,
        "max_tokens" : template.max_tokens,
        "few_shots"  : template.few_shots,
    }


def _write_artifact(data: dict, filename: str, artifact_path: str) -> None:
    p = Path("/tmp") / filename
    p.write_text(json.dumps(data, indent=2))
    mlflow.log_artifact(str(p), artifact_path=artifact_path)
