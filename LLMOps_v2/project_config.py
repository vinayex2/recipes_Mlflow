"""
llmops_core/project_config.py

Loads a project.yaml file and exposes a typed ProjectConfig dataclass.
Every pipeline phase calls load_project_config(path) to get its settings.
Nothing in llmops_core hardcodes project names, model names, thresholds,
templates, or datasets — all of that lives in the YAML.

YAML structure (see projects/support_agent/project.yaml for a full example):

  project:
    name: support-agent
    mlflow_experiment: llmops/support-agent
    mlflow_model_name: workspace.default.support_agent
    judge_model: gemini_3_1_flash_Newer

  gates:
    ci_rule_pass_rate: 0.80
    ci_avg_latency_ms: 8000
    ci_avg_cost_usd:   0.05
    cc_min_improvement: 0.02

  cost:
    per_1k_input_usd:  0.0025
    per_1k_output_usd: 0.0100

  templates:
    - name: zero-shot-support
      version: "1.0.0"
      model: gemini_3_1_flash_Newer
      temperature: 0.2
      max_tokens: 1024
      system: |
        You are a helpful customer support agent for Acme SaaS.
        ...
      few_shots: []

  golden_dataset:
    - id: eval-001
      user_message: "How do I cancel my subscription?"
      expected_topics: [cancel, subscription, account]
      must_not_contain: [competitor, "I cannot help"]
      max_words: 160
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ── Template dataclass (mirrors prompt_model.PromptTemplate) ──────────────────

@dataclass
class TemplateConfig:
    name        : str
    version     : str
    system      : str
    model       : str
    temperature : float        = 0.3
    max_tokens  : int          = 1024
    few_shots   : list[dict]   = field(default_factory=list)


# ── Gate thresholds ───────────────────────────────────────────────────────────

@dataclass
class GateConfig:
    ci_rule_pass_rate  : float = 0.80
    ci_avg_latency_ms  : float = 8000.0
    ci_avg_cost_usd    : float = 0.05
    cc_min_improvement : float = 0.02


# ── Token cost ────────────────────────────────────────────────────────────────

@dataclass
class CostConfig:
    per_1k_input_usd  : float = 0.0025
    per_1k_output_usd : float = 0.0100

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens  / 1000) * self.per_1k_input_usd +
            (output_tokens / 1000) * self.per_1k_output_usd
        )


# ── Top-level project config ──────────────────────────────────────────────────

@dataclass
class ProjectConfig:
    # identity
    name               : str
    mlflow_experiment  : str
    mlflow_model_name  : str
    judge_model        : str

    # pipeline config
    gates              : GateConfig
    cost               : CostConfig

    # per-project data
    templates          : list[TemplateConfig]
    golden_dataset     : list[dict]

    # optional — set by caller, not from YAML
    score_threshold    : float = 0.60


def load_project_config(yaml_path: str | Path) -> ProjectConfig:
    """
    Load and validate a project.yaml file.
    Raises FileNotFoundError or ValueError on bad input.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"project.yaml not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    proj = raw.get("project", {})
    name = proj.get("name")
    if not name:
        raise ValueError("project.yaml must have a non-empty project.name")

    gates_raw = raw.get("gates", {})
    gates = GateConfig(
        ci_rule_pass_rate  = gates_raw.get("ci_rule_pass_rate",   0.80),
        ci_avg_latency_ms  = gates_raw.get("ci_avg_latency_ms",   8000.0),
        ci_avg_cost_usd    = gates_raw.get("ci_avg_cost_usd",     0.05),
        cc_min_improvement = gates_raw.get("cc_min_improvement",  0.02),
    )

    cost_raw = raw.get("cost", {})
    cost = CostConfig(
        per_1k_input_usd  = cost_raw.get("per_1k_input_usd",  0.0025),
        per_1k_output_usd = cost_raw.get("per_1k_output_usd", 0.0100),
    )

    templates = [
        TemplateConfig(
            name        = t["name"],
            version     = str(t.get("version", "1.0.0")),
            system      = t["system"],
            model       = t.get("model", "gemini_3_1_flash_Newer"),
            temperature = float(t.get("temperature", 0.3)),
            max_tokens  = int(t.get("max_tokens", 1024)),
            few_shots   = t.get("few_shots", []),
        )
        for t in raw.get("templates", [])
    ]
    if not templates:
        raise ValueError("project.yaml must define at least one template under 'templates:'")

    golden_dataset = raw.get("golden_dataset", [])
    if not golden_dataset:
        raise ValueError("project.yaml must define at least one case under 'golden_dataset:'")

    return ProjectConfig(
        name              = name,
        mlflow_experiment = proj.get("mlflow_experiment", f"llmops/{name}"),
        mlflow_model_name = proj.get("mlflow_model_name", f"workspace.default.{name}"),
        judge_model       = proj.get("judge_model", "gemini_3_1_flash_Newer"),
        gates             = gates,
        cost              = cost,
        templates         = templates,
        golden_dataset    = golden_dataset,
        score_threshold   = float(proj.get("score_threshold", 0.60)),
    )
