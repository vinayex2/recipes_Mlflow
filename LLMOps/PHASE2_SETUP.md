# LLMOps Phase 2 — CI Setup Guide
# GitHub Actions + Databricks Community Edition

---

## File layout

```
your_repo/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                  ← GitHub Actions workflow
│   └── scripts/
│       └── submit_and_poll.py      ← submits notebook run, polls result
├── llmops_phase1_experiment.py     ← Phase 1 (already built)
├── llmops_phase2_ci.py             ← upload this to Databricks
├── prompt_model.py
└── requirements.txt
```

---

## Step 1 — Add GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions → New secret.

| Secret name       | Value                                         |
|-------------------|-----------------------------------------------|
| `DATABRICKS_HOST` | `https://community.cloud.databricks.com`      |
| `DATABRICKS_TOKEN`| Your Databricks personal access token (below) |
| `OPENAI_API_KEY`  | Your OpenAI API key                           |

### Get your Databricks personal access token
1. Log in to Databricks Community Edition
2. Click your username (top right) → User Settings
3. Access tokens → Generate new token
4. Copy it — you only see it once

---

## Step 2 — Upload the CI notebook to Databricks

1. In Databricks, go to Workspace → Shared
2. Create a folder called `llmops`
3. Import `llmops_phase2_ci.py` as a notebook:
   - Click the `llmops` folder → Import
   - Select `llmops_phase2_ci.py`
   - Format: Python file
4. The notebook will be available at `/Shared/llmops/llmops_phase2_ci`

---

## Step 3 — Register a candidate from Phase 1

The CI notebook needs a registered model to test.  Run Phase 1 first:

```bash
python llmops_phase1_experiment.py
```

Confirm the model appears in your Databricks MLflow UI:
- Experiments → llmops/phase1-local-dev → check runs exist
- Models → llmops/support-agent → check a version with alias `candidate` exists

---

## Step 4 — Push and trigger CI

```bash
git add .
git commit -m "feat: add LLMOps Phase 2 CI"
git push origin main
```

GitHub Actions will:
1. Trigger `ci.yml` on push
2. Run `submit_and_poll.py`, which calls the Databricks Runs Submit API
3. A transient cluster starts in Databricks CE (~3–5 min)
4. The CI notebook runs, evaluates the candidate, logs results to MLflow
5. The poller fetches the result and exits 0 (pass) or 1 (fail)
6. The PR check turns green or red

---

## Step 5 — View results

- GitHub: Actions tab → LLMOps CI → job logs
- Databricks: Workspace → Clusters → (transient run cluster) → Spark UI
- MLflow: Experiments → llmops/phase2-ci → latest run

---

## Quality gate thresholds

Thresholds are set as environment variables so you can override without
editing code.  Set them as GitHub repo variables or in the notebook:

| Variable                  | Default  | Meaning                        |
|---------------------------|----------|--------------------------------|
| `CI_GATE_RULE_PASS_RATE`  | `0.80`   | Min fraction of eval cases passing rule eval |
| `CI_GATE_LATENCY_MS`      | `8000`   | Max avg latency per call (ms)  |
| `CI_GATE_COST_USD`        | `0.05`   | Max avg cost per call (USD)    |

---

## Databricks Community Edition constraints

| Constraint                     | How we handle it                               |
|-------------------------------|------------------------------------------------|
| No persistent clusters         | `runs/submit` spins up a transient cluster     |
| No Jobs API job definitions    | We use `runs/submit` (no job ID needed)        |
| Single-node only               | `num_workers=0` + `spark.master=local[*]`      |
| No Secrets API (CE)            | API key passed as notebook widget parameter    |
| Cluster startup ~3–5 min       | Poller timeout set to 30 min                   |

### Upgrading to a full Databricks workspace
When you move off Community Edition, replace the widget-based secret passing
with Databricks Secrets:
```python
OPENAI_API_KEY = dbutils.secrets.get(scope="llmops", key="openai_api_key")
```
And replace the `node_type_id` in `submit_and_poll.py` with your workspace's
available instance types.
