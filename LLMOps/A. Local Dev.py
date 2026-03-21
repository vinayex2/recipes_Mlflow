"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LLMOps Phase 1 — Local Experiment Notebook                                 ║
║  Conversation with an LLM + MLflow tracking                                  ║
║                                                                              ║
║  Workflow:                                                                   ║
║    1. Define prompt templates (system + few-shot examples)                   ║
║    2. Run a multi-turn conversation with an LLM (Anthropic Claude)           ║
║    3. Log every experiment run to MLflow (params, metrics, artifacts)        ║
║    4. Evaluate responses using an LLM-as-judge rubric                        ║
║    5. Register the best-performing prompt config in MLflow Model Registry    ║
║                                                                              ║
║  Dependencies:                                                               ║
║    pip install anthropic mlflow python-dotenv tiktoken                       ║
║                                                                              ║
║  Environment:                                                                ║
║    OPEN_AI_KEY=<your key>          (or set in .env)                    ║
║    MLFLOW_TRACKING_URI=<uri>             (default: local ./mlruns)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import os
import time
import hashlib
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── third-party ─────────────────────────────────────────────────────────────
import anthropic
import mlflow
import mlflow.pyfunc
import tiktoken
from dotenv import load_dotenv


# ════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

load_dotenv()

# ── MLflow tracking setup ───────────────────────────────────────────────────
# For Databricks: set MLFLOW_TRACKING_URI="databricks" and configure
# DATABRICKS_HOST / DATABRICKS_TOKEN in your env.
# For local dev: leave as-is; runs land in ./mlruns
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "llmops/phase1-local-dev")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── LLM client ──────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Token counter (approximation for non-OpenAI models) ─────────────────────
try:
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:          # fallback: word heuristic
        return int(len(text.split()) * 1.3)


# ════════════════════════════════════════════════════════════════════════════
# 1.  PROMPT TEMPLATES
#     Each template is a versioned, self-contained config.
#     Treat these like code: version-controlled, tested in isolation.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PromptTemplate:
    """
    A versioned prompt configuration.

    Attributes
    ----------
    name        : short slug used in MLflow run tags
    version     : semver string — bump on any edit
    system      : the system prompt sent to the model
    few_shots   : list of {"role": ..., "content": ...} example turns
    temperature : sampling temperature
    max_tokens  : token budget for the completion
    model       : model identifier string
    """
    name        : str
    version     : str
    system      : str
    few_shots   : list[dict]  = field(default_factory=list)
    temperature : float       = 0.3
    max_tokens  : int         = 1024
    model       : str         = "claude-sonnet-4-20250514"

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def config_hash(self) -> str:
        """Stable hash of the prompt config — useful for deduplication."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def to_mlflow_params(self) -> dict:
        """Flat dict suitable for mlflow.log_params()."""
        return {
            "template.name"        : self.name,
            "template.version"     : self.version,
            "template.config_hash" : self.config_hash,
            "model"                : self.model,
            "temperature"          : self.temperature,
            "max_tokens"           : self.max_tokens,
            "few_shot_count"       : len(self.few_shots),
            "system_token_count"   : count_tokens(self.system),
        }


# ── Template A: zero-shot customer support agent ─────────────────────────────
TEMPLATE_ZERO_SHOT = PromptTemplate(
    name    = "zero-shot-support",
    version = "1.0.0",
    system  = textwrap.dedent("""\
        You are a helpful, concise customer support agent for Acme SaaS.
        Rules:
        - Answer only questions about Acme products.
        - If you don't know, say so clearly — never fabricate.
        - Keep replies under 150 words unless detail is essential.
        - Always end with: "Is there anything else I can help with?"
    """),
    temperature = 0.2,
)

# ── Template B: few-shot customer support agent ───────────────────────────────
TEMPLATE_FEW_SHOT = PromptTemplate(
    name    = "few-shot-support",
    version = "1.0.0",
    system  = TEMPLATE_ZERO_SHOT.system,
    few_shots = [
        {
            "role"   : "user",
            "content": "How do I reset my password?",
        },
        {
            "role"   : "assistant",
            "content": (
                "To reset your password: visit the login page, click "
                "'Forgot password', and enter your email. You'll receive "
                "a reset link within 2 minutes. Check your spam folder if "
                "it doesn't arrive.\n\nIs there anything else I can help with?"
            ),
        },
        {
            "role"   : "user",
            "content": "What payment methods do you accept?",
        },
        {
            "role"   : "assistant",
            "content": (
                "Acme accepts Visa, Mastercard, American Express, and PayPal. "
                "Annual plans can also be paid by bank transfer — contact "
                "billing@acme.example for details.\n\n"
                "Is there anything else I can help with?"
            ),
        },
    ],
    temperature = 0.2,
)

# ── Template C: chain-of-thought variant ────────────────────────────────────
TEMPLATE_COT = PromptTemplate(
    name    = "cot-support",
    version = "1.0.0",
    system  = textwrap.dedent("""\
        You are a helpful customer support agent for Acme SaaS.
        Before answering, briefly think through the customer's need in
        a <thinking> block (not shown to the customer), then provide your
        final answer in an <answer> block.

        Rules:
        - Answer only questions about Acme products.
        - If you don't know, say so clearly — never fabricate.
        - Keep the <answer> under 150 words.
        - Always end the <answer> with: "Is there anything else I can help with?"
    """),
    temperature = 0.4,
)

TEMPLATES = [TEMPLATE_ZERO_SHOT, TEMPLATE_FEW_SHOT, TEMPLATE_COT]


# ════════════════════════════════════════════════════════════════════════════
# 2.  CONVERSATION ENGINE
#     Manages multi-turn history and calls the LLM.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """One exchange in a conversation."""
    user_message      : str
    assistant_response: str
    latency_ms        : float
    input_tokens      : int
    output_tokens     : int
    timestamp         : str = field(default_factory=lambda: datetime.utcnow().isoformat())


class Conversation:
    """
    Multi-turn conversation manager.

    Usage
    -----
    >>> conv = Conversation(template=TEMPLATE_FEW_SHOT)
    >>> reply = conv.chat("How do I cancel my subscription?")
    >>> print(reply)
    """

    def __init__(self, template: PromptTemplate):
        self.template = template
        self.history  : list[dict] = []    # running message list (no system)
        self.turns    : list[Turn] = []    # detailed turn records

    # ── seed with few-shot examples ──────────────────────────────────────────
    def _build_messages(self, user_message: str) -> list[dict]:
        """Construct the messages array: few-shots + history + new user turn."""
        messages = list(self.template.few_shots)  # copy few-shots first
        messages.extend(self.history)             # then real conversation
        messages.append({"role": "user", "content": user_message})
        return messages

    # ── single chat turn ─────────────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        """Send a message and return the assistant reply."""
        messages = self._build_messages(user_message)

        t0 = time.perf_counter()
        response = client.messages.create(
            model      = self.template.model,
            max_tokens = self.template.max_tokens,
            temperature= self.template.temperature,
            system     = self.template.system,
            messages   = messages,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        reply = response.content[0].text

        # ── persist history ──────────────────────────────────────────────────
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant", "content": reply})

        # ── record the turn ──────────────────────────────────────────────────
        self.turns.append(Turn(
            user_message       = user_message,
            assistant_response = reply,
            latency_ms         = round(latency_ms, 1),
            input_tokens       = response.usage.input_tokens,
            output_tokens      = response.usage.output_tokens,
        ))

        return reply

    # ── helpers ──────────────────────────────────────────────────────────────
    def reset(self):
        """Clear history (but keep template)."""
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


# ════════════════════════════════════════════════════════════════════════════
# 3.  GOLDEN EVAL DATASET
#     A small frozen set of (user_message, expected_criteria) pairs.
#     Never tune prompts against this set — it exists only for measurement.
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
        "user_message"    : "What is the capital of France?",   # out-of-scope
        "expected_topics" : ["don't know", "cannot", "outside", "Acme"],
        "must_not_contain": ["Paris"],   # should NOT answer off-topic questions
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


# ════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATORS
# ════════════════════════════════════════════════════════════════════════════

# ── 4a. Rule-based evaluator (fast, free, deterministic) ────────────────────

@dataclass
class RuleEvalResult:
    eval_id          : str
    passed           : bool
    topic_hits       : int
    topic_misses     : list[str]
    must_not_violated: list[str]
    word_count       : int
    within_word_limit: bool
    ends_with_cta    : bool
    score            : float   # 0.0 – 1.0


def rule_eval(response: str, case: dict) -> RuleEvalResult:
    """Fast, deterministic rule checks against a golden eval case."""
    resp_lower = response.lower()

    topic_hits   = [t for t in case["expected_topics"]  if t.lower() in resp_lower]
    topic_misses = [t for t in case["expected_topics"]  if t.lower() not in resp_lower]
    violations   = [t for t in case["must_not_contain"] if t.lower() in resp_lower]

    word_count        = len(response.split())
    within_word_limit = word_count <= case["max_words"]
    ends_with_cta     = "anything else" in resp_lower

    # weighted score
    topic_score  = len(topic_hits) / max(len(case["expected_topics"]), 1)
    penalty      = len(violations) * 0.25
    format_bonus = 0.1 if ends_with_cta else 0.0
    score        = max(0.0, min(1.0, topic_score - penalty + format_bonus))

    return RuleEvalResult(
        eval_id           = case["id"],
        passed            = score >= 0.6 and not violations,
        topic_hits        = len(topic_hits),
        topic_misses      = topic_misses,
        must_not_violated = violations,
        word_count        = word_count,
        within_word_limit = within_word_limit,
        ends_with_cta     = ends_with_cta,
        score             = round(score, 3),
    )


# ── 4b. LLM-as-judge evaluator ───────────────────────────────────────────────

JUDGE_SYSTEM = textwrap.dedent("""\
    You are an impartial evaluator for a customer support AI.
    You will receive a user message and the assistant's response.
    Return ONLY valid JSON with this schema:
    {
      "helpfulness"  : <integer 1-5>,
      "faithfulness" : <integer 1-5>,
      "conciseness"  : <integer 1-5>,
      "safety"       : <integer 1-5>,
      "reasoning"    : "<one sentence>"
    }
    Definitions:
    - helpfulness  : does the response fully address the user's need?
    - faithfulness : does it stay within known facts, avoiding hallucination?
    - conciseness  : is it appropriately brief without losing clarity?
    - safety       : does it avoid harmful, offensive, or off-topic content?
    Do not include any text outside the JSON object.
""")


@dataclass
class JudgeResult:
    eval_id     : str
    helpfulness : int
    faithfulness: int
    conciseness : int
    safety      : int
    reasoning   : str
    avg_score   : float


def llm_judge(user_message: str, response: str, eval_id: str,
              judge_model: str = "claude-haiku-4-5-20251001") -> JudgeResult:
    """
    Use a separate LLM to score a response on four rubric dimensions.
    Cheaper models (Haiku) work well as judges for structured scoring.
    """
    prompt = (
        f"USER MESSAGE:\n{user_message}\n\n"
        f"ASSISTANT RESPONSE:\n{response}"
    )
    judge_response = client.messages.create(
        model      = judge_model,
        max_tokens = 256,
        temperature= 0.0,   # fully deterministic for reproducibility
        system     = JUDGE_SYSTEM,
        messages   = [{"role": "user", "content": prompt}],
    )
    raw = judge_response.content[0].text.strip()

    # strip markdown fences if model wraps with ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    scores = json.loads(raw)

    avg = (scores["helpfulness"] + scores["faithfulness"] +
           scores["conciseness"] + scores["safety"]) / 4.0

    return JudgeResult(
        eval_id      = eval_id,
        helpfulness  = scores["helpfulness"],
        faithfulness = scores["faithfulness"],
        conciseness  = scores["conciseness"],
        safety       = scores["safety"],
        reasoning    = scores.get("reasoning", ""),
        avg_score    = round(avg, 2),
    )


# ════════════════════════════════════════════════════════════════════════════
# 5.  MLflow EXPERIMENT RUNNER
#     Wraps a full evaluation pass in a single MLflow run.
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(
    template      : PromptTemplate,
    eval_dataset  : list[dict] = GOLDEN_DATASET,
    run_judge     : bool       = True,
    run_name      : Optional[str] = None,
) -> dict:
    """
    Execute one full experiment pass:
      - Run each eval case through the conversation engine
      - Apply rule-based evals
      - Optionally apply LLM-as-judge evals
      - Log everything to MLflow
      - Return a summary dict

    Parameters
    ----------
    template     : PromptTemplate to evaluate
    eval_dataset : list of golden eval cases
    run_judge    : whether to run the LLM-as-judge (costs tokens)
    run_name     : human-readable MLflow run name (auto-generated if None)
    """
    if run_name is None:
        run_name = f"{template.name}-v{template.version}-{template.config_hash}"

    print(f"\n{'═'*60}")
    print(f"  Starting experiment: {run_name}")
    print(f"{'═'*60}")

    with mlflow.start_run(run_name=run_name) as run:

        # ── log template config ───────────────────────────────────────────────
        mlflow.log_params(template.to_mlflow_params())
        mlflow.set_tags({
            "template.name"   : template.name,
            "template.version": template.version,
            "eval_set_size"   : str(len(eval_dataset)),
            "judge_enabled"   : str(run_judge),
            "run_date"        : datetime.utcnow().strftime("%Y-%m-%d"),
        })

        # ── per-case results ──────────────────────────────────────────────────
        rule_results  : list[RuleEvalResult] = []
        judge_results : list[JudgeResult]    = []
        turn_records  : list[dict]           = []

        for case in eval_dataset:
            print(f"\n  [{case['id']}] {case['user_message'][:60]}…")

            # fresh conversation for each eval case (stateless eval)
            conv  = Conversation(template=template)
            reply = conv.chat(case["user_message"])
            turn  = conv.turns[0]

            print(f"         → {reply[:80].replace(chr(10), ' ')}…")
            print(f"           latency={turn.latency_ms:.0f}ms  "
                  f"tokens={turn.input_tokens}in/{turn.output_tokens}out")

            # ── rule eval ─────────────────────────────────────────────────────
            r = rule_eval(reply, case)
            rule_results.append(r)
            status = "✓ PASS" if r.passed else "✗ FAIL"
            print(f"           rule_eval: {status}  score={r.score}")

            # ── llm judge ─────────────────────────────────────────────────────
            if run_judge:
                j = llm_judge(case["user_message"], reply, case["id"])
                judge_results.append(j)
                print(f"           llm_judge: avg={j.avg_score}  | {j.reasoning}")

            # ── accumulate for artifact ───────────────────────────────────────
            record = {
                "eval_id"          : case["id"],
                "user_message"     : case["user_message"],
                "assistant_response": reply,
                "latency_ms"       : turn.latency_ms,
                "input_tokens"     : turn.input_tokens,
                "output_tokens"    : turn.output_tokens,
                "rule_score"       : r.score,
                "rule_passed"      : r.passed,
                "topic_misses"     : r.topic_misses,
                "violations"       : r.must_not_violated,
            }
            if run_judge:
                record.update({
                    "judge_helpfulness" : j.helpfulness,
                    "judge_faithfulness": j.faithfulness,
                    "judge_conciseness" : j.conciseness,
                    "judge_safety"      : j.safety,
                    "judge_avg"         : j.avg_score,
                    "judge_reasoning"   : j.reasoning,
                })
            turn_records.append(record)

        # ── aggregate metrics ─────────────────────────────────────────────────
        n = len(eval_dataset)

        rule_pass_rate  = sum(r.passed for r in rule_results) / n
        avg_rule_score  = sum(r.score for r in rule_results)  / n
        avg_latency     = sum(r["latency_ms"] for r in turn_records) / n
        avg_input_tok   = sum(r["input_tokens"] for r in turn_records) / n
        avg_output_tok  = sum(r["output_tokens"] for r in turn_records) / n

        mlflow.log_metrics({
            "eval/rule_pass_rate"  : round(rule_pass_rate,  3),
            "eval/avg_rule_score"  : round(avg_rule_score,  3),
            "perf/avg_latency_ms"  : round(avg_latency,     1),
            "perf/avg_input_tokens": round(avg_input_tok,   1),
            "perf/avg_output_tokens":round(avg_output_tok,  1),
        })

        if run_judge and judge_results:
            avg_judge          = sum(j.avg_score     for j in judge_results) / n
            avg_helpfulness    = sum(j.helpfulness   for j in judge_results) / n
            avg_faithfulness   = sum(j.faithfulness  for j in judge_results) / n
            avg_conciseness    = sum(j.conciseness   for j in judge_results) / n
            avg_safety         = sum(j.safety        for j in judge_results) / n

            mlflow.log_metrics({
                "eval/judge_avg_score"      : round(avg_judge,        2),
                "eval/judge_helpfulness"    : round(avg_helpfulness,  2),
                "eval/judge_faithfulness"   : round(avg_faithfulness, 2),
                "eval/judge_conciseness"    : round(avg_conciseness,  2),
                "eval/judge_safety"         : round(avg_safety,       2),
            })

        # ── log artifact: full results JSON ───────────────────────────────────
        artifact_path = Path("/tmp") / f"{run_name}_results.json"
        artifact_data = {
            "run_id"     : run.info.run_id,
            "run_name"   : run_name,
            "template"   : asdict(template),
            "timestamp"  : datetime.utcnow().isoformat(),
            "turn_records": turn_records,
        }
        artifact_path.write_text(json.dumps(artifact_data, indent=2))
        mlflow.log_artifact(str(artifact_path), artifact_path="eval_results")

        # ── log artifact: prompt template ─────────────────────────────────────
        template_path = Path("/tmp") / f"{run_name}_template.json"
        template_path.write_text(json.dumps(asdict(template), indent=2))
        mlflow.log_artifact(str(template_path), artifact_path="prompt_config")

        # ── summary ──────────────────────────────────────────────────────────
        summary = {
            "run_id"         : run.info.run_id,
            "run_name"       : run_name,
            "rule_pass_rate" : rule_pass_rate,
            "avg_rule_score" : avg_rule_score,
            "avg_latency_ms" : avg_latency,
        }
        if run_judge and judge_results:
            summary["avg_judge_score"] = avg_judge

        print(f"\n  {'─'*54}")
        print(f"  Run ID       : {run.info.run_id}")
        print(f"  Rule pass    : {rule_pass_rate:.0%}  ({sum(r.passed for r in rule_results)}/{n})")
        print(f"  Avg rule score: {avg_rule_score:.3f}")
        if run_judge and judge_results:
            print(f"  Avg judge score: {avg_judge:.2f}/5.0")
        print(f"  Avg latency  : {avg_latency:.0f} ms")

        return summary


# ════════════════════════════════════════════════════════════════════════════
# 6.  MODEL REGISTRY — register best candidate
#     After evaluating multiple templates, register the winner.
# ════════════════════════════════════════════════════════════════════════════

def register_best_candidate(
    summaries       : list[dict],
    templates       : list[PromptTemplate],
    model_name      : str = "llmops/support-agent",
    score_threshold : float = 0.70,
) -> Optional[str]:
    """
    Compare experiment summaries, pick the best by composite score,
    and register it in the MLflow Model Registry if it clears the threshold.

    The registered "model" is the prompt config wrapped in a pyfunc
    — not weights, just the versioned prompt + call logic.

    Returns the MLflow model version URI, or None if threshold not met.
    """

    # ── pick winner ───────────────────────────────────────────────────────────
    def composite(s: dict) -> float:
        rule  = s.get("avg_rule_score",  0.0)
        judge = s.get("avg_judge_score", 0.0) / 5.0   # normalise 1-5 → 0-1
        return (rule * 0.5) + (judge * 0.5) if judge else rule

    ranked  = sorted(zip(summaries, templates), key=lambda x: composite(x[0]), reverse=True)
    best_summary, best_template = ranked[0]
    best_score = composite(best_summary)

    print(f"\n{'═'*60}")
    print(f"  Best template : {best_template.name} v{best_template.version}")
    print(f"  Composite score: {best_score:.3f}  (threshold: {score_threshold})")

    if best_score < score_threshold:
        print(f"  ✗ Below threshold — not registering.")
        return None

    print(f"  ✓ Above threshold — registering in MLflow Model Registry…")

    # ── wrap prompt config as a pyfunc model ─────────────────────────────────
    class PromptConfigModel(mlflow.pyfunc.PythonModel):
        """
        A pyfunc wrapper that stores the prompt template as model artifacts
        and exposes a predict() interface: input df with 'message' column,
        output list of LLM responses.

        In production this class gets loaded by mlflow.pyfunc.load_model()
        and called via .predict(). The model is the *prompt config*, not weights.
        """

        def load_context(self, context):
            cfg_path = context.artifacts["prompt_config"]
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.template = PromptTemplate(**cfg)
            self._client  = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        def predict(self, context, model_input):
            messages = (
                model_input["message"].tolist()
                if hasattr(model_input, "tolist")
                else list(model_input["message"])
            )
            results = []
            for msg in messages:
                conv  = Conversation(template=self.template)
                reply = conv.chat(msg)
                results.append(reply)
            return results

    # ── persist template as artifact file ────────────────────────────────────
    cfg_path = Path("/tmp") / f"prompt_config_{best_template.config_hash}.json"
    cfg_path.write_text(json.dumps(asdict(best_template), indent=2))

    artifacts = {"prompt_config": str(cfg_path)}

    # ── log and register ──────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"register-{best_template.name}"):
        mlflow.log_params(best_template.to_mlflow_params())
        mlflow.log_metrics({
            "composite_score"  : round(best_score, 3),
            "rule_pass_rate"   : round(best_summary["rule_pass_rate"], 3),
        })
        mlflow.set_tags({
            "stage"                : "candidate",
            "template.name"       : best_template.name,
            "template.config_hash": best_template.config_hash,
        })

        model_info = mlflow.pyfunc.log_model(
            artifact_path   = "prompt_model",
            python_model    = PromptConfigModel(),
            artifacts       = artifacts,
            registered_model_name = model_name,
            pip_requirements= ["anthropic", "mlflow"],
        )

    uri = model_info.model_uri
    print(f"  Registered: {uri}")
    print(f"  Model name: {model_name}")
    return uri


# ════════════════════════════════════════════════════════════════════════════
# 7.  INTERACTIVE DEMO — run a live multi-turn conversation
# ════════════════════════════════════════════════════════════════════════════

def interactive_demo(template: PromptTemplate = TEMPLATE_FEW_SHOT):
    """
    Drop into a REPL-style conversation with the LLM.
    Type 'quit' or 'exit' to stop. Type 'reset' to clear history.
    All turns are logged to a single MLflow run.
    """
    print(f"\n{'═'*60}")
    print(f"  Interactive Demo — {template.name} v{template.version}")
    print(f"  Type 'quit' to exit, 'reset' to clear conversation history.")
    print(f"{'═'*60}\n")

    conv = Conversation(template=template)

    with mlflow.start_run(run_name=f"interactive-{template.name}"):
        mlflow.log_params(template.to_mlflow_params())
        mlflow.set_tag("mode", "interactive")

        turn_idx = 0
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[session ended]")
                break

            if user_input.lower() in {"quit", "exit"}:
                break
            if user_input.lower() == "reset":
                conv.reset()
                print("[conversation reset]\n")
                continue
            if not user_input:
                continue

            reply = conv.chat(user_input)
            turn  = conv.turns[-1]

            print(f"\nAssistant: {reply}\n")

            # log each turn as a step metric
            mlflow.log_metrics({
                "turn/latency_ms"   : turn.latency_ms,
                "turn/input_tokens" : turn.input_tokens,
                "turn/output_tokens": turn.output_tokens,
            }, step=turn_idx)
            turn_idx += 1

        # log session summary
        tok = conv.total_tokens()
        mlflow.log_metrics({
            "session/total_turns"       : len(conv.turns),
            "session/total_input_tokens": tok["input"],
            "session/total_output_tokens":tok["output"],
            "session/avg_latency_ms"    : conv.avg_latency_ms(),
        })


# ════════════════════════════════════════════════════════════════════════════
# 8.  MAIN — run all experiments, compare, register best
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\nLLMOps Phase 1 — Local Experiment Runner")
    print(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Experiment          : {EXPERIMENT_NAME}")
    print(f"Templates to test   : {[t.name for t in TEMPLATES]}")
    print(f"Eval cases          : {len(GOLDEN_DATASET)}")

    # ── run each template through the eval suite ──────────────────────────────
    summaries: list[dict] = []
    for template in TEMPLATES:
        summary = run_experiment(
            template   = template,
            run_judge  = True,         # set False to skip judge and save tokens
        )
        summaries.append(summary)

    # ── print comparison table ────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {'Template':<28}  {'Rule%':>6}  {'Rule↑':>6}  {'Judge':>6}  {'ms':>6}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for s, t in zip(summaries, TEMPLATES):
        judge_str = f"{s.get('avg_judge_score', 0):>5.2f}" if 'avg_judge_score' in s else "  N/A"
        print(
            f"  {t.name:<28}  "
            f"{s['rule_pass_rate']:>5.0%}  "
            f"{s['avg_rule_score']:>6.3f}  "
            f"{judge_str}  "
            f"{s['avg_latency_ms']:>5.0f}"
        )

    # ── register best candidate ───────────────────────────────────────────────
    uri = register_best_candidate(
        summaries       = summaries,
        templates       = TEMPLATES,
        model_name      = "llmops/support-agent",
        score_threshold = 0.60,
    )
    if uri:
        print(f"\n✓ Candidate registered: {uri}")
        print("  Next step: run CI pipeline (Phase 2) against this registered model.")
    else:
        print("\n✗ No candidate met the threshold. Iterate on your prompt templates.")

    print(f"\n  View results: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # run interactive demo instead:  python llmops_phase1_experiment.py demo
        interactive_demo(template=TEMPLATE_FEW_SHOT)
    else:
        main()