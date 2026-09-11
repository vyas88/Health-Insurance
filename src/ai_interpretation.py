"""Optional OpenAI explanation layer for already-computed statistical results."""

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # The rest of the statistical application still works without this optional package.
    def load_dotenv(*_args, **_kwargs):
        return False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.6-luna"

SYSTEM_PROMPT = """
You are the optional AI interpretation layer of a university-built multivariate health-insurance risk-profiling application.
Explain only the supplied, already-computed results in plain, professional language for a reader with no statistics background.

The Python statistical model is authoritative. Never recalculate, replace, change, or invent a risk tier, probability, cost range, test result, or statistical conclusion. Do not infer causation, make medical claims, predict a person's future health, give an underwriting decision, or quote an insurance premium.

Refer to the cost output as an "observed annual medical-cost range" from the historical dataset, never as a premium. Describe class probabilities as model support for a risk-group classification, not as a probability of illness, a claim, or a future cost.

Explain Mahalanobis distance as how typical the combined profile is within its predicted group. If QDA is selected because covariance patterns differ, state that QDA permits group-specific covariance structures; do not claim it is automatically more accurate. If assumptions are not fully supported, acknowledge this as a reason for cautious, empirical interpretation without declaring the analysis invalid.

Return only these Markdown sections when enough information is supplied:
### Overall assessment
### What the model found
### What this means in practical terms
### Why the model made this assessment
### What the cost range means
### Important limitation

Use short paragraphs, no equations, no unsupported feature-level causes, and no more than about 250 words.
""".strip()


class AIInterpretationError(Exception):
    """A safe, user-facing failure for the optional explanation service."""


def get_ai_settings():
    """Load the one local .env file and return only usable API settings."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not api_key or api_key == "your_api_key_here":
        return None, model
    return api_key, model


def build_ai_context(profile, assessment, results):
    """Create the small, case-specific contract sent to the interpretation layer."""
    classification = results["classification"]
    box_m = results["box_m"]
    return {
        "applicant_profile": profile,
        "statistical_result": {
            "risk_tier": assessment["tier"],
            "class_probabilities": assessment["probabilities"],
            "classifier": classification["method"],
            "observed_cost_range": assessment["cost_range"],
            "mahalanobis_distance": round(assessment["distance"], 2),
            "mahalanobis_cutoff": round(assessment["cutoff"], 2),
            "profile_status": assessment["status"],
        },
        "model_context": {
            "cross_validation_accuracy": classification["overall_accuracy"],
            "lda_accuracy": classification["comparison"]["lda_cv_mean"],
            "qda_accuracy": classification["comparison"]["qda_cv_mean"],
            "covariance_diagnostic": {
                "conclusion": "Covariance homogeneity is not supported.",
                "box_m_p_value": box_m["p_value"],
            },
            "relevant_key_findings": [
                "The classifier uses all eight standardized features.",
                "The observed medical-cost range is not an insurance premium quote.",
                "Mardia diagnostics do not support exact multivariate normality for the continuous variables.",
            ],
        },
    }


def create_client(api_key):
    """Create the official OpenAI client only when the user requests an explanation."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise AIInterpretationError("The optional OpenAI package is not installed.") from error
    return OpenAI(api_key=api_key)


def generate_ai_interpretation(client, model, context):
    """Request a concise explanation without giving OpenAI any decision-making role."""
    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=json.dumps(context, indent=2),
            max_output_tokens=600,
            store=False,
        )
        if not response.output_text:
            raise AIInterpretationError("The interpretation service returned no text.")
        return response.output_text
    except AIInterpretationError:
        raise
    except Exception as error:
        raise AIInterpretationError("AI interpretation is currently unavailable.") from error
