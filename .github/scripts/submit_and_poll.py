"""
.github/scripts/submit_and_poll.py

Submits a one-off notebook run to Databricks Community Edition using the
Runs Submit API (no persistent cluster or Job definition needed), then polls
until the run completes and exits with a non-zero code on failure so GitHub
Actions marks the check as failed.

Databricks Community Edition constraints handled here:
  - No Jobs API v2.1 "create job then run" — we use runs/submit instead,
    which spins up a transient single-node cluster for one run and tears it
    down automatically.
  - Community Edition clusters are single-node (no worker nodes).
  - The notebook path must already exist in the workspace.
"""

import json
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
NOTEBOOK_PATH = "/Workspace/Users/vasaicrow@gmail.com/recipes_Mlflow/LLMOps/llmops_phase2_ci"

# Community Edition: single-node cluster, smallest available node type.
# "Runtime 14.3 LTS ML" includes MLflow pre-installed.
CLUSTER_SPEC = {
    "spark_version"  : "14.3.x-scala2.12",
    "node_type_id"   : "i3.xlarge",      # CE default; adjust if yours differs
    "num_workers"    : 0,                 # single-node (driver only)
    "spark_conf"     : {
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master"                    : "local[*]",
    },
    "custom_tags"    : {"ResourceClass": "SingleNode"},
}

POLL_INTERVAL_S = 20    # seconds between status checks
MAX_WAIT_S      = 1800  # 30 minutes — CE clusters take a few minutes to start


def submit_run() -> str:
    """Submit the notebook as a one-off run and return the run_id."""
    payload = {
        "run_name": (
            f"llmops-ci-{os.environ.get('GIT_SHA', 'local')[:8]}"
        ),
        "new_cluster": CLUSTER_SPEC,
        "notebook_task": {
            "notebook_path": NOTEBOOK_PATH,
            # Pass CI context and secrets into the notebook as parameters.
            # The notebook reads these via dbutils.widgets.get().
            "base_parameters": {
                "git_sha"       : os.environ.get("GIT_SHA",  ""),
                "git_ref"       : os.environ.get("GIT_REF",  ""),
                "git_pr"        : os.environ.get("GIT_PR",   ""),
                "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN", ""),
                "GEMINI_ENDPOINT": os.environ.get("GEMINI_ENDPOINT", ""),
            },
        },
        "libraries": [
            {"pypi": {"package": "openai>=1.30.0"}},
            {"pypi": {"package": "tiktoken>=0.7.0"}},
        ],
    }

    resp = requests.post(
        f"{HOST}/api/2.1/jobs/runs/submit",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    print(f"[CI] Submitted run_id={run_id}")
    print(f"[CI] View at: {HOST}/#job/0/run/{run_id}")
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
        resp.raise_for_status()
        run = resp.json()

        life_cycle = run["state"]["life_cycle_state"]
        result     = run["state"].get("result_state", "")
        print(f"[CI] [{elapsed:>4}s] life_cycle={life_cycle}  result={result or '—'}")

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
