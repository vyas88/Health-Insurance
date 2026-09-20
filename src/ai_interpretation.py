"""Optional OpenAI explanation layer for already-computed statistical results."""

import json
import logging
import os
from pathlib import Path

# dotenv is optional: it loads local configuration, not the model or data.
# If unavailable, normal environment variables and Python classification still work.
try:
    from dotenv import load_dotenv
except ImportError:  # The rest of the statistical application still works without this optional package.
    # This no-op fallback keeps the same callable interface when dotenv is absent.
    # It does not create a key or pretend that configuration was loaded.
    def load_dotenv(*_args, **_kwargs):
        return False


ROOT = Path(__file__).resolve().parents[1]
# Preserve the existing model setting; OPENAI_MODEL can override it.
# A configured name does not guarantee availability for a particular account.
DEFAULT_MODEL = "gpt-5.6-luna"
LOGGER = logging.getLogger(__name__)

# These are instructions to the language model, not executable Python comments.
# Keep computed facts authoritative and restrict the role to explanation.
# A prompt guides wording but cannot mathematically guarantee faithful output.
SYSTEM_PROMPT = """
Explain only the supplied Python-computed facts in 120-180 words of plain English.
Never recalculate, override or invent the class, support, costs or metrics. Similarity
is not causality. Do not diagnose illness, predict future claims or expenditure,
recommend premiums or applicant acceptance/rejection, or imply underwriting readiness.
Support is the supplied uniform fraction or normalized distance-weighted vote, not
calibrated confidence. Even 100% support is not proof of correctness. Historical group
summaries are not personalized intervals. Mention uncertain individual outcomes.
Use exactly these headings, with concise prose under each:
### Your result
### What the similar profiles show
### How to read the model support
### Business perspective
### Cost context and limitations
Business perspective may discuss exploratory historical portfolio analysis or further
review only. Avoid equations, statistical jargon and p-values. Do not use em dashes.
""".strip()


# A dedicated exception lets the UI handle optional-service failures without
# confusing them with a failed Python classification.
class AIInterpretationError(Exception):
    """A safe, user-facing failure for the optional explanation service."""


# Read local configuration without making a network request. Return None for
# an absent or placeholder key so the caller can show an availability message.
def get_ai_settings():
    """Load the one local .env file and return only usable API settings."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not api_key or api_key == "your_api_key_here":
        return None, model
    return api_key, model


# Build a compact explanation contract from already-computed results.
# Only the submitted profile and aggregate neighbour facts are included,
# not the entire training dataset or the preview of individual neighbours.
def build_ai_context(profile, assessment, results):
    return {
        "submitted_profile": profile,
        "run_id": results["run_id"],
        "predicted_observed_cost_group": assessment["tier"],
        "neighbour_support": assessment["support"],
        "voting": assessment["weights"],
        "k": assessment["k"],
        "distance_metric": assessment["metric"],
        "neighbour_counts": assessment["neighbour_counts"],
        "outside_training_range": assessment["outside_training_range"],
        "frozen_thresholds": results["thresholds"],
        "boundary_rule": results["boundary_rule"],
        "historical_development_group_summary": results["historical_development_costs"][assessment["tier"]],
        "held_out_test_metrics": results["models"]["k-NN"]["test"],
        "limitations": "Previously explored historical sample; held out during this refactor, not external validation. Missing clinical and utilization information; individual outcomes uncertain. No causal or future-cost interpretation."
    }


# Hash a canonical JSON representation of the complete context AND model.
# Sorted keys prevent dictionary insertion order from changing the cache key.
# Different profiles, metrics or run IDs must not reuse stale prose.
# This fingerprint is not anonymization of the context sent to the service.
def explanation_key(context, model):
    import hashlib
    return hashlib.sha256(json.dumps([context, model], sort_keys=True).encode()).hexdigest()


# Construct the client lazily, inside the explicitly requested AI path.
# The timeout and bounded SDK retries limit request waiting; a retry can
# make total elapsed time longer than the timeout of a single attempt.
def create_client(api_key):
    """Create the official OpenAI client only when the user requests an explanation."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise AIInterpretationError("The optional OpenAI package is not installed.") from error
    return OpenAI(api_key=api_key, timeout=20.0, max_retries=1)


# Return successful text or raise the neutral service exception.
# The fitted classifier is never called or changed by the language model.
def generate_ai_interpretation(client, model, context):
    """Request a concise explanation without giving OpenAI any decision-making role."""
    try:
        # Send the existing configured model, system instructions and compact JSON.
        # The output budget bounds generated tokens; store=False requests that this
        # response not be stored for later API retrieval. It does not mean no data
        # is transmitted. Preserve the existing reasoning setting for this integration.
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=json.dumps(context, indent=2),
            max_output_tokens=650,
            reasoning={"effort": "none"},
            store=False,
        )
        # A partial response can contain text, so nonempty text alone is insufficient.
        # Require completed status and non-whitespace output. getattr permits missing
        # optional metadata without causing another exception in the failure handler.
        if getattr(response, "status", None) != "completed" or not response.output_text or not response.output_text.strip():
            error = getattr(response, "error", None)
            incomplete = getattr(response, "incomplete_details", None)
            # Log only status/codes/reasons, not API keys, profiles or response contents.
            # These diagnostics identify service failure without dumping sensitive context.
            LOGGER.warning(
                "OpenAI interpretation returned no text: status=%s, error_code=%s, incomplete_reason=%s",
                getattr(response, "status", None),
                getattr(error, "code", None),
                getattr(incomplete, "reason", None),
            )
            raise AIInterpretationError("The interpretation service returned no text.")
        return response.output_text
    # Re-raise our own safe exception unchanged. Unexpected SDK/network errors
    # are converted below so the app remains usable with its Python result.
    except AIInterpretationError:
        raise
    except Exception as error:
        # Keep the UI safe while retaining a non-sensitive diagnostic in Cloud logs.
        LOGGER.warning("OpenAI interpretation request failed: %s", type(error).__name__)
        raise AIInterpretationError("AI interpretation is currently unavailable.") from error
