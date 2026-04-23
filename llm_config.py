from __future__ import annotations


# model behavior settings

default_max_output_tokens = 400
default_temperature = 0.0
# low temperature is better here because we want stable, structured JSON output instead of creative writing.
default_top_p = 1.0

# Prompting settings
# these control how the prompt is built.

use_few_shot_prompting = False
default_few_shot_example_count = 3

# If few-shot prompting is turned on, keep the number small so the prompt stays short and easy to manage.
maximum_few_shot_example_count = 5


# output settings
# These define the structure we expect back from the model.

required_output_keys = [
    "classification",
    "scam_subtype",
    "suspicious_phrases",
    "short_explanation",
]

allowed_classification_labels = {
    "safe",
    "scam",
}

default_safe_scam_subtype = "none"
default_unknown_classification = "unknown"
default_general_scam_subtype = "general_scam"


# fallback messages
# these are used when the model output is missing, broken, or does not follow the expected format.

default_invalid_output_explanation = "The model did not return a valid result."
default_missing_explanation = "The model did not provide a short explanation."


# demo settings for now
# These are just for local testing with the demo callable.

demo_trigger_keywords = [
    "urgent",
    "pay now",
    "verify",
    "gift card",
    "click the link",
]

demo_safe_explanation = "This message appears safe."
demo_scam_explanation = "This message is risky because it uses urgent scam-like language."


#Optional backend labels
# These are not required for the pipeline logic, they just make it easier to stay organized :)
supported_backend_types = {
    "demo",
    "openai",
    "huggingface",
    "local",
}

default_backend_type = "demo"