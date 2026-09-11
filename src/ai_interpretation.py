"""Optional OpenAI explanation layer for already-computed statistical results."""

import json
import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # The rest of the statistical application still works without this optional package.
    def load_dotenv(*_args, **_kwargs):
        return False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.6-luna"
LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the interpretation assistant for a university-built multivariate health-insurance risk-profiling application. A separate Python statistical model has already calculated every supplied result. Treat those results as authoritative and only interpret them.

Never recalculate or change the risk tier, probabilities, cost range, or statistical conclusions. Never invent statistics, causal explanations, medical diagnoses, future illness or claim predictions, a premium, or an underwriting decision. The reader has no statistics background.

Write strictly about 100 to 150 words in concise plain English, using one short sentence under each heading. Use the supplied probabilities only as "model support for the classification", never as probabilities of illness, claims, health, or future costs. Describe typicality without requiring the reader to understand Mahalanobis distance. The observed annual medical-cost range is historical dataset context, not a premium quote or guaranteed future cost.

Return exactly these Markdown headings in this order. Write one short sentence under each heading and do not omit any heading:
### Your result
### What this means
### How strong is the model's support?
### Profile check
### Business perspective
### Cost context
### Bottom line

In Business perspective, use cautious insurance language such as may, could, or could warrant. Do not recommend a price or approve or reject an applicant. End by noting that the assessment is based on historical data and is not medical advice or an insurance quote. Do not include equations, p-values, degrees of freedom, PCA, LDA, QDA, covariance, or other technical jargon unless specifically asked.
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
        "risk_result": {
            "risk_tier": assessment["tier"],
            "class_probabilities": assessment["probabilities"],
            "classifier": classification["method"],
            "observed_cost_range": assessment["cost_range"],
            "mahalanobis_distance": round(assessment["distance"], 2),
            "mahalanobis_cutoff": round(assessment["cutoff"], 2),
            "profile_status": assessment["status"],
        },
        "model_context": {
            "cv_accuracy": classification["overall_accuracy"],
            "lda_accuracy": classification["comparison"]["lda_cv_mean"],
            "qda_accuracy": classification["comparison"]["qda_cv_mean"],
            "box_m_p_value": box_m["p_value"],
            "key_findings": [
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
            max_output_tokens=360,
            reasoning={"effort": "none"},
            store=False,
        )
        if not response.output_text:
            error = getattr(response, "error", None)
            incomplete = getattr(response, "incomplete_details", None)
            LOGGER.warning(
                "OpenAI interpretation returned no text: status=%s, error_code=%s, incomplete_reason=%s",
                getattr(response, "status", None),
                getattr(error, "code", None),
                getattr(incomplete, "reason", None),
            )
            raise AIInterpretationError("The interpretation service returned no text.")
        return response.output_text
    except AIInterpretationError:
        raise
    except Exception as error:
        # Keep the UI safe while retaining a non-sensitive diagnostic in Cloud logs.
        LOGGER.warning("OpenAI interpretation request failed: %s", type(error).__name__)
        raise AIInterpretationError("AI interpretation is currently unavailable.") from error
