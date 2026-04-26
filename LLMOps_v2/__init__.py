"""
llmops_core — shared pipeline infrastructure.

Import from here in all phase entry points:

    from llmops_core.project_config      import load_project_config
    from llmops_core.evaluators          import rule_eval, llm_judge
    from llmops_core.pipeline_experiment import run_all_experiments, register_best_candidate
    from llmops_core.pipeline_ci         import run_ci
    from llmops_core.pipeline_cd         import run_staging, run_production
    from llmops_core.mlflow_helpers      import log_cc_run
"""
