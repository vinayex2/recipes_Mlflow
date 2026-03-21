# LLMOps Phase 1 — Local Experiment Notebook

Local model development and prompt tuning with MLflow tracking.
This is **Phase 1** of the full LLMOps pipeline.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY=sk-ant-...

# 3. Run all experiments and register the best candidate
python llmops_phase1_experiment.py

# 4. Drop into an interactive conversation
python llmops_phase1_experiment.py demo

# 5. View results in MLflow UI
mlflow ui --backend-store-uri mlruns
# open http://localhost:5000
```

---

## What this notebook does

### 1. Prompt templates (`PromptTemplate`)
Three versioned prompt configs are defined and compared:

| Template | Strategy | Description |
|---|---|---|
| `zero-shot-support` | Zero-shot | System prompt only, no examples |
| `few-shot-support`  | Few-shot  | 2 worked Q&A examples inline |
| `cot-support`       | Chain-of-thought | Reasoning in `<thinking>` block |

Each template is a dataclass with a stable `config_hash` — treat it like code.
Version it, test changes in isolation, never edit casually.

### 2. Conversation engine (`Conversation`)
Multi-turn conversation manager that:
- Prepends few-shot examples before the live conversation history
- Tracks latency, input tokens, output tokens per turn
- Keeps running history for multi-turn coherence
- Supports `reset()` to clear state between eval cases

### 3. Golden eval dataset
A frozen set of 5 labeled eval cases covering:
- Normal support queries (cancel subscription, invoice, team members)
- Integration query (Salesforce)
- Out-of-scope query (capital of France) — model should refuse, not answer

**Do not tune prompts against this set.** It exists only for measurement.

### 4. Evaluators

#### Rule-based (fast, free, deterministic)
Checks each response against:
- `expected_topics` — keywords that should appear
- `must_not_contain` — terms that should never appear (e.g. off-topic answers)
- `max_words` — word count limit
- CTA check — does it end with "Is there anything else I can help with?"

Runs on every CI push. Zero cost.

#### LLM-as-judge (richer, costs tokens)
Uses `claude-haiku` to score responses on a 1–5 rubric:
- **Helpfulness** — does it address the user's need?
- **Faithfulness** — does it avoid hallucination?
- **Conciseness** — appropriately brief?
- **Safety** — no harmful or off-topic content?

Returns structured JSON scores + one-sentence reasoning.

### 5. MLflow tracking
Every experiment run logs:
- **Params**: template name, version, config hash, model, temperature, token counts
- **Metrics**: rule pass rate, avg rule score, judge scores (all 4 dimensions), latency, token usage
- **Artifacts**: full results JSON per run, prompt template JSON
- **Tags**: template name/version, eval set size, run date

### 6. Model Registry
After all templates are evaluated, `register_best_candidate()`:
1. Ranks templates by composite score (50% rule + 50% judge, normalised)
2. Checks against a configurable threshold (default 0.60)
3. Wraps the winning prompt config in an `mlflow.pyfunc` model
4. Registers it in the MLflow Model Registry with alias `candidate`

The registered "model" is the **prompt config + call logic** — not weights.
This is the artifact that flows into Phase 2 (CI) and Phase 3 (CD).

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow server URI (use `databricks` on Databricks) |
| `MLFLOW_EXPERIMENT_NAME` | `llmops/phase1-local-dev` | Experiment name |

For Databricks, also set:
```
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=dapi...
```

---

## File structure

```
llmops_phase1_experiment.py   ← this file
requirements.txt
.env                          ← your keys (gitignored)
mlruns/                       ← local MLflow tracking store
```

---

## Next steps (Phase 2 — CI)

Once a candidate is registered, the CI pipeline will:
1. Pull the registered model from MLflow Model Registry
2. Run the same golden eval dataset as a quality gate
3. Check latency, cost, and safety thresholds
4. Block the PR if any gate fails
5. Log the CI run back to MLflow

See `llmops_phase2_ci.py` (coming next).
