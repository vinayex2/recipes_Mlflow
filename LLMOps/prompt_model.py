"""
prompt_model.py — MLflow code-based pyfunc model for LLMOps Phase 1.

This file is intentionally separate from the experiment notebook so that
MLflow can serialize it via code-based logging (mlflow.pyfunc.log_model
with model_code_path=) rather than pickling a class defined inside a function.

MLflow loads this file at serve/inference time by importing it and calling
mlflow.pyfunc.load_model(), which in turn calls load_context() then predict().

Loaded artifacts
----------------
  prompt_config  →  JSON file containing a serialized PromptTemplate

predict() interface
-------------------
  Input  : pandas DataFrame with a "message" column (list of user strings)
  Output : list[str]  —  one assistant reply per input message
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any
import hashlib

import mlflow.pyfunc
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# ── PromptTemplate must be redefined here so this file is fully self-contained
#    (MLflow loads it in a fresh interpreter with no imports from the notebook)

@dataclass
class PromptTemplate:
    name        : str
    version     : str
    system      : str
    few_shots   : list  = field(default_factory=list)
    temperature : float = 0.3
    max_tokens  : int   = 1024
    model       : str   = "gemini_3_1_flash_Newer"

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
            # "system_token_count"   : count_tokens(self.system),
        }


# ── Thin conversation helper (no MLflow / eval dependencies needed here) ──────

def _chat_once(template: PromptTemplate, user_message: str, oai_client: Any) -> str:
    """Single-turn chat using the OpenAI Chat Completions API."""
    messages = [{"role": "system", "content": template.system}]
    messages.extend(template.few_shots)
    messages.append({"role": "user", "content": user_message})

    response = oai_client.chat.completions.create(
        model       = template.model,
        max_tokens  = template.max_tokens,
        temperature = template.temperature,
        messages    = messages,
    )
    return response.choices[0].message.content


# ── The pyfunc model class — must be at module level for MLflow to import it ──

class PromptConfigModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper around a versioned prompt template.

    The "model" here is the prompt config (system prompt, few-shots,
    sampling parameters) — not neural-network weights.  This lets us
    version, register, and deploy prompt configs through the standard
    MLflow Model Registry workflow.

    Usage after loading:
        model = mlflow.pyfunc.load_model("models:/llmops/support-agent/1")
        import pandas as pd
        preds = model.predict(pd.DataFrame({"message": ["How do I cancel?"]}))
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        cfg_path = context.artifacts["prompt_config"]
        with open(cfg_path) as f:
            cfg = json.load(f)

        # reconstruct the dataclass — few_shots may have been stored as a list
        self.template = PromptTemplate(**cfg)

        self._client = OpenAI(
            api_key = DATABRICKS_TOKEN,
            base_url = os.getenv("GEMINI_ENDPOINT"),   # None → defaults to api.openai.com
        )

    def predict(
        self,
        context    : mlflow.pyfunc.PythonModelContext,
        model_input: Any,
    ) -> list[str]:
        # accept both pandas DataFrame and plain dict/list
        if hasattr(model_input, "__getitem__"):
            raw = model_input["message"]
            messages = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        else:
            messages = list(model_input)

        return [
            _chat_once(self.template, msg, self._client)
            for msg in messages
        ]

# ── Required for MLflow code-based logging ────────────────────────────────────
# When python_model is a file path (not an instance), MLflow executes this file
# and looks for mlflow.models.set_model() to know which class to register.
# Without this line MLflow raises:
#   MlflowException: If the model is logged as code, ensure the model is set
#   using mlflow.models.set_model() within the code file.
mlflow.models.set_model(PromptConfigModel())    
