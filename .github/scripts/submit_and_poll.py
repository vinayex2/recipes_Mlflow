"""
.github/scripts/submit_and_poll.py

Submits a one-off notebook run to Databricks Free Edition using the
Runs Submit API, then polls until the run completes and exits with a
non-zero code on failure so GitHub Actions marks the check as failed.

Databricks Free Edition constraints :
  - Serverless compute ONLY — no custom cluster specs (node_type_id,
    spark_version, num_workers, spark_conf) are accepted.  Sending any
    of those fields causes a 400 Bad Request.
  - To use serverless, omit new_cluster/existing_cluster_id entirely.
    The API treats a task with no cluster spec as serverless by default.
  - The payload must use the tasks[] array format (API 2.1 multi-task).
  - Task-level libraries are not supported for notebook tasks on serverless.
    Install packages inside the notebook using %pip instead.
  - Environment variables are not supported — pass secrets as notebook
    widget parameters (base_parameters) instead.
  - Outbound internet is restricted to trusted domains; api.openai.com
    is accessible from Free Edition workspaces.
"""

import os
import sys
import time

import requests

# ── Config from environment ───────────────────────────────────────────────────
HOST  = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type" : "application/json",
}

# Path of the CI notebook inside your Databricks workspace.
# Upload llmops_phase2_ci.py to this path via the Databricks UI or CLI.
NOTEBOOK_PATH = "/Workspace/Users/vasaicrow@gmail.com/recipes_Mlflow/LLMOps/B. llmops_phase2_ci"

POLL_INTERVAL_S = 20    # seconds between status checks
MAX_WAIT_S      = 1800  # 30 minutes


def submit_run() -> str:
    """
    Submit the notebook as a one-off serverless run and return the run_id.

    Key differences from a classic cluster payload:
      - No new_cluster block at all (serverless = omit cluster spec entirely)
      - No top-level libraries field (not supported for serverless notebook tasks)
      - Tasks wrapped in tasks[] array (required by API 2.1)
      - Secrets passed as base_parameters widgets, not env vars
    """
    payload = {
        "run_name": f"llmops-ci-{os.environ.get('GIT_SHA', 'local')[:8]}",
        "tasks": [
            {
                "task_key": "ci_quality_gates",
                # No new_cluster / existing_cluster_id / job_cluster_key here.
                # Omitting all cluster fields tells Databricks to use serverless.
                "notebook_task": {
                    "notebook_path": NOTEBOOK_PATH,
                    # Pass CI context and the OpenAI key as notebook widget
                    # parameters.  The notebook reads them via dbutils.widgets.get().
                    # We cannot use environment variables on serverless compute.
                    "base_parameters": {
                        "git_sha"       : os.environ.get("GIT_SHA",  ""),
                        "git_ref"       : os.environ.get("GIT_REF",  ""),
                        "git_pr"        : os.environ.get("GIT_PR",   ""),
                        "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN", ""),
                "GEMINI_ENDPOINT": os.environ.get("GEMINI_ENDPOINT", ""),
                    },
                    "source": "WORKSPACE",
                },
                # Libraries are installed inside the notebook via %pip.
                # Task-level library installs are not supported for serverless
                # notebook tasks — omit the libraries field entirely.
            }
        ],
    }

    resp = requests.post(
        f"{HOST}/api/2.1/jobs/runs/submit",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    # Print the full response body on errors — the default raise_for_status()
    # message hides the Databricks error_code and message that explain *why*
    # the request was rejected (e.g. invalid cluster spec, missing fields).
    if not resp.ok:
        print(f"[CI] ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        resp.raise_for_status()

    run_id = resp.json()["run_id"]
    print(f"[CI] Submitted run_id={run_id}")
    print(f"[CI] View at: {HOST}/jobs/runs/{run_id}")
    return str(run_id)


def poll_run(run_id: str) -> dict:
    """Poll until the run reaches a terminal state. Returns the final run object."""
    url     = f"{HOST}/api/2.1/jobs/runs/get"
    elapsed = 0

    while elapsed < MAX_WAIT_S:
        resp = requests.get(
            url,
            headers=HEADERS,
            params={"run_id": run_id},
            timeout=30,
        )
        if not resp.ok:
            print(f"[CI] Poll error {resp.status_code}: {resp.text}", file=sys.stderr)
            resp.raise_for_status()

        run = resp.json()

        # Top-level run state (PENDING, RUNNING, TERMINATED, etc.)
        life_cycle = run["state"]["life_cycle_state"]
        result     = run["state"].get("result_state", "")

        # For multi-task runs, also show individual task states
        task_states = [
            f"{t['task_key']}={t['state'].get('result_state', t['state']['life_cycle_state'])}"
            for t in run.get("tasks", [])
        ]
        task_str = "  tasks=[" + ", ".join(task_states) + "]" if task_states else ""

        print(f"[CI] [{elapsed:>4}s] life_cycle={life_cycle}  result={result or '—'}{task_str}")

        if life_cycle in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            return run

        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S

    raise TimeoutError(f"Run {run_id} did not complete within {MAX_WAIT_S}s")


def parse_notebook_output(run: dict) -> tuple[bool, str]:
    """
    Extract the CI pass/fail result from the notebook's output cell.

    The CI notebook writes its final verdict as the last cell output in the form:
        CI_RESULT: PASS
    or
        CI_RESULT: FAIL  reason=...

    We extract that line from the notebook output to get a human-readable message
    in addition to the overall run result state.
    """
    result_state = run["state"].get("result_state", "UNKNOWN")
    passed       = result_state == "SUCCESS"

    # Extract notebook output message if available
    output = run.get("state", {}).get("state_message", "")
    tasks  = run.get("tasks", [])
    if tasks:
        output = tasks[0].get("state", {}).get("state_message", output)

    return passed, f"result_state={result_state}  message={output}"


def main() -> None:
    print("[CI] Phase 2 — Databricks CI quality gates")
    print(f"[CI] Notebook : {NOTEBOOK_PATH}")
    print(f"[CI] Host     : {HOST}")
    print(f"[CI] SHA      : {os.environ.get('GIT_SHA', 'local')[:12]}")

    run_id          = submit_run()
    run             = poll_run(run_id)
    passed, message = parse_notebook_output(run)

    print(f"\n[CI] {'PASSED' if passed else 'FAILED'} — {message}")

    if not passed:
        print("[CI] One or more quality gates failed. PR is blocked.")
        sys.exit(1)

    print("[CI] All quality gates passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
