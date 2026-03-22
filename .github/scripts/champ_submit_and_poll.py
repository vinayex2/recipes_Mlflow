"""
.github/scripts/champ_submit_and_poll.py

Submits the champion/challenger evaluation notebook to Databricks Free Edition
(serverless) and polls for completion.
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

NOTEBOOK_PATH  = os.environ.get("NOTEBOOK_PATH", "/Workspace/Users/vasaicrow@gmail.com/recipes_Mlflow/LLMOps/D. llmops_phase4_champion_challenger")
POLL_INTERVAL_S = 20
MAX_WAIT_S      = 2700   # 45 min — two full eval suites with judge calls


def submit_run() -> str:
    sha = os.environ.get("GIT_SHA", "")[:8]

    payload = {
        "run_name": f"llmops-champ-challenger-{sha}",
        "tasks": [
            {
                "task_key": "champion_challenger_eval",
                "notebook_task": {
                    "notebook_path": NOTEBOOK_PATH,
                    "base_parameters": {
                        "champion_version"  : os.environ.get("CHAMPION_VERSION",   ""),
                        "challenger_version": os.environ.get("CHALLENGER_VERSION", ""),
                        "min_improvement"   : os.environ.get("MIN_IMPROVEMENT",    "0.02"),
                        "git_sha"           : os.environ.get("GIT_SHA",            ""),
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
        print(f"[CC] ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        resp.raise_for_status()

    run_id = resp.json()["run_id"]
    print(f"[CC] Submitted run_id={run_id}")
    print(f"[CC] View at: {HOST}/jobs/runs/{run_id}")
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
            print(f"[CC] Poll error {resp.status_code}: {resp.text}", file=sys.stderr)
            resp.raise_for_status()

        run        = resp.json()
        life_cycle = run["state"]["life_cycle_state"]
        result     = run["state"].get("result_state", "")
        task_states = [
            f"{t['task_key']}={t['state'].get('result_state', t['state']['life_cycle_state'])}"
            for t in run.get("tasks", [])
        ]
        task_str = "  [" + ", ".join(task_states) + "]" if task_states else ""
        print(f"[CC] [{elapsed:>4}s] {life_cycle}  {result or '—'}{task_str}")

        if life_cycle in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            return run

        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S

    raise TimeoutError(f"Run {run_id} did not complete within {MAX_WAIT_S}s")


def main():
    print("[CC] Phase 4 — Champion/Challenger evaluation")
    print(f"[CC] Champion version  : {os.environ.get('CHAMPION_VERSION') or '(from champion alias)'}")
    print(f"[CC] Challenger version: {os.environ.get('CHALLENGER_VERSION') or '(from candidate alias)'}")
    print(f"[CC] Min improvement   : {os.environ.get('MIN_IMPROVEMENT', '0.02')}")

    run_id       = submit_run()
    run          = poll_run(run_id)
    result_state = run["state"].get("result_state", "UNKNOWN")
    passed       = result_state == "SUCCESS"

    msg = run.get("state", {}).get("state_message", "")
    print(f"\n[CC] {'COMPLETE' if passed else 'FAILED'} — result={result_state}  {msg}")

    if not passed:
        print("[CC] Champion/challenger notebook failed. Check Databricks for details.")
        sys.exit(1)

    print("[CC] Evaluation complete. Check MLflow for promotion decision.")
    sys.exit(0)


if __name__ == "__main__":
    main()
