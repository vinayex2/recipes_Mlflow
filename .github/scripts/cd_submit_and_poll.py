"""
.github/scripts/cd_submit_and_poll.py

Submits the CD notebook to Databricks Free Edition (serverless) and polls
for completion.  Used for both the staging and production deployment steps —
DEPLOYMENT_STAGE env var tells the notebook which path to execute.

Behaviour:
  DEPLOYMENT_STAGE=staging    → deploy to staging endpoint + smoke test
  DEPLOYMENT_STAGE=production → promote candidate alias to production alias
                                 + log audit trail to MLflow
"""

import os
import sys
import time

import requests

HOST  = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type" : "application/json",
}

NOTEBOOK_PATH    = os.environ.get("NOTEBOOK_PATH", "/Workspace/Users/vasaicrow@gmail.com/recipes_Mlflow/LLMOps/C. llmops_phase3_cd")
DEPLOYMENT_STAGE = os.environ.get("DEPLOYMENT_STAGE", "staging")
POLL_INTERVAL_S  = 20
MAX_WAIT_S       = 1800


def submit_run() -> str:
    sha   = os.environ.get("GIT_SHA", "")[:8]
    stage = DEPLOYMENT_STAGE

    payload = {
        "run_name": f"llmops-cd-{stage}-{sha}",
        "tasks": [
            {
                "task_key": f"cd_{stage}",
                "notebook_task": {
                    "notebook_path": NOTEBOOK_PATH,
                    "base_parameters": {
                        "deployment_stage": DEPLOYMENT_STAGE,
                        "git_sha"         : os.environ.get("GIT_SHA",       ""),
                        "git_ref"         : os.environ.get("GIT_REF",       ""),
                        "approved_by"     : os.environ.get("APPROVED_BY",   ""),
                        "model_version"   : os.environ.get("MODEL_VERSION", ""),
                        "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN", ""),
                        "GEMINI_ENDPOINT": os.environ.get("GEMINI_ENDPOINT", ""),
                    },
                    "source": "WORKSPACE",
                },
            }
        ],
    }

    resp = requests.post(
        f"{HOST}/api/2.1/jobs/runs/submit",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    if not resp.ok:
        print(f"[CD] ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        resp.raise_for_status()

    run_id = resp.json()["run_id"]
    print(f"[CD:{DEPLOYMENT_STAGE}] Submitted run_id={run_id}")
    print(f"[CD:{DEPLOYMENT_STAGE}] View at: {HOST}/jobs/runs/{run_id}")
    return str(run_id)


def poll_run(run_id: str) -> dict:
    elapsed = 0
    while elapsed < MAX_WAIT_S:
        resp = requests.get(
            f"{HOST}/api/2.1/jobs/runs/get",
            headers=HEADERS,
            params={"run_id": run_id},
            timeout=30,
        )
        if not resp.ok:
            print(f"[CD] Poll error {resp.status_code}: {resp.text}", file=sys.stderr)
            resp.raise_for_status()

        run        = resp.json()
        life_cycle = run["state"]["life_cycle_state"]
        result     = run["state"].get("result_state", "")
        task_states = [
            f"{t['task_key']}={t['state'].get('result_state', t['state']['life_cycle_state'])}"
            for t in run.get("tasks", [])
        ]
        task_str = "  tasks=[" + ", ".join(task_states) + "]" if task_states else ""
        print(f"[CD:{DEPLOYMENT_STAGE}] [{elapsed:>4}s] {life_cycle}  {result or '—'}{task_str}")

        if life_cycle in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            return run

        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S

    raise TimeoutError(f"Run {run_id} did not complete within {MAX_WAIT_S}s")


def main():
    print(f"[CD] Phase 3 — {DEPLOYMENT_STAGE} deployment")
    print(f"[CD] Host    : {HOST}")
    print(f"[CD] SHA     : {os.environ.get('GIT_SHA', 'local')[:12]}")

    run_id       = submit_run()
    run          = poll_run(run_id)
    result_state = run["state"].get("result_state", "UNKNOWN")
    passed       = result_state == "SUCCESS"

    msg = run.get("state", {}).get("state_message", "")
    print(f"\n[CD:{DEPLOYMENT_STAGE}] {'PASSED' if passed else 'FAILED'} — result={result_state}  {msg}")

    if not passed:
        print(f"[CD] {DEPLOYMENT_STAGE} deployment failed. Check notebook output in Databricks.")
        sys.exit(1)

    print(f"[CD] {DEPLOYMENT_STAGE} deployment succeeded.")
    sys.exit(0)


if __name__ == "__main__":
    main()
