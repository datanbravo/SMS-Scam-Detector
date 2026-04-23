from __future__ import annotations

import json
from typing import Any, Callable

from prompt_builder import (
    build_few_shot_example,
    build_inference_prompt_from_message,
)


# This is the default empty result when the model output is missing 0r cannot be parsed into the expected JSON structure.
default_empty_result = {
    "classification": "unknown",
    "scam_subtype": "none",
    "suspicious_phrases": [],
    "short_explanation": "The model did not return a valid result.",
}


def parse_generated_response_json_text(generated_response_text: str) -> dict[str, Any]:
    # Try to parse the model output as JSON, if the model adds extra text before or after the JSON, we try to recover the JSON object from inside the response.
    stripped_response_text = generated_response_text.strip()

    if not stripped_response_text:
        return {}

    try:
        parsed_response = json.loads(stripped_response_text)
    except json.JSONDecodeError:
        first_curly_brace_index = stripped_response_text.find("{")
        last_curly_brace_index = stripped_response_text.rfind("}")

        if first_curly_brace_index == -1 or last_curly_brace_index == -1:
            return {}

        candidate_json_text = stripped_response_text[first_curly_brace_index:last_curly_brace_index + 1]

        try:
            parsed_response = json.loads(candidate_json_text)
        except json.JSONDecodeError:
            return {}

    if not isinstance(parsed_response, dict):
        return {}

    return parsed_response


def clean_suspicious_phrase_list(suspicious_phrase_list: Any) -> list[dict[str, Any]]:
    # Clean the suspicious phrase field so it always becomes a list of dictionaries with the expected keys.
    if not isinstance(suspicious_phrase_list, list):
        return []

    cleaned_phrase_list: list[dict[str, Any]] = []

    for suspicious_phrase in suspicious_phrase_list:
        if not isinstance(suspicious_phrase, dict):
            continue

        phrase_text = str(suspicious_phrase.get("phrase_text", "")).strip()
        risk_category = str(suspicious_phrase.get("risk_category", "")).strip()
        risk_explanation = str(suspicious_phrase.get("risk_explanation", "")).strip()

        if not phrase_text:
            continue

        if not risk_category:
            continue

        cleaned_phrase_list.append(
            {
                "phrase_text": phrase_text,
                "risk_category": risk_category,
                "risk_explanation": risk_explanation,
            }
        )

    return cleaned_phrase_list


def normalize_prediction_dictionary(prediction_dictionary: dict[str, Any]) -> dict[str, Any]:
    # Normalize the model output into the exact structure we want.
    classification = str(prediction_dictionary.get("classification", "unknown")).strip().lower()
    scam_subtype = str(prediction_dictionary.get("scam_subtype", "none")).strip()
    short_explanation = str(prediction_dictionary.get("short_explanation", "")).strip()

    if classification not in {"safe", "scam"}:
        classification = "unknown"

    if classification == "safe":
        scam_subtype = "none"
    elif not scam_subtype:
        scam_subtype = "general_scam"

    suspicious_phrases = clean_suspicious_phrase_list(
        prediction_dictionary.get("suspicious_phrases", [])
    )

    if not short_explanation:
        short_explanation = "The model did not provide a short explanation."

    return {
        "classification": classification,
        "scam_subtype": scam_subtype,
        "suspicious_phrases": suspicious_phrases,
        "short_explanation": short_explanation,
    }


def build_few_shot_prompt(
    message_text: str,
    example_rows: list[dict[str, Any]],
) -> str:
    # Build one prompt that includes a few worked examples before the real message to analyze.
    prompt_parts: list[str] = []

    for example_index, example_row in enumerate(example_rows, start=1):
        few_shot_example = build_few_shot_example(example_row)

        prompt_parts.append(f"Example {example_index} Input:\n{few_shot_example['input']}")
        prompt_parts.append(f"Example {example_index} Output:\n{few_shot_example['output']}")

    prompt_parts.append("Now analyze a new SMS message.")
    prompt_parts.append(build_inference_prompt_from_message(message_text))

    return "\n\n".join(prompt_parts)


def build_prompt(
    message_text: str,
    example_rows: list[dict[str, Any]] | None = None,
) -> str:
    # Build the final prompt. If examples are provided, use few-shot prompting. Otherwise, use a simple zero-shot prompt.
    if example_rows:
        return build_few_shot_prompt(
            message_text=message_text,
            example_rows=example_rows,
        )

    return build_inference_prompt_from_message(message_text)


def run_model_on_prompt(
    prompt_text: str,
    model_callable: Callable[[str], str],
) -> str:
    # Run the model backend, because it's easier rn.
    return model_callable(prompt_text)


def predict_message(
    message_text: str,
    model_callable: Callable[[str], str],
    example_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Main public function for predicting one SMS message. It builds the prompt, calls the model, parses the output, and returns a normalized result.
    prompt_text = build_prompt(
        message_text=message_text,
        example_rows=example_rows,
    )

    generated_response_text = run_model_on_prompt(
        prompt_text=prompt_text,
        model_callable=model_callable,
    )

    parsed_response = parse_generated_response_json_text(generated_response_text)

    if not parsed_response:
        return dict(default_empty_result)

    return normalize_prediction_dictionary(parsed_response)


def predict_messages(
    message_list: list[str],
    model_callable: Callable[[str], str],
    example_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # Run prediction for multiple messages.
    prediction_list: list[dict[str, Any]] = []

    for message_text in message_list:
        prediction_list.append(
            predict_message(
                message_text=message_text,
                model_callable=model_callable,
                example_rows=example_rows,
            )
        )

    return prediction_list


def build_demo_model_callable() -> Callable[[str], str]:
    # This is a placeholder model backend for testing the pipeline before connecting a real LLM. Add local model later...
    def demo_model_callable(prompt_text: str) -> str:
        prompt_text_lower = prompt_text.lower()

        if any(keyword in prompt_text_lower for keyword in ["urgent", "pay now", "verify", "gift card", "click the link"]):
            return json.dumps(
                {
                    "classification": "scam",
                    "scam_subtype": "general_scam",
                    "suspicious_phrases": [
                        {
                            "phrase_text": "urgent",
                            "risk_category": "urgency",
                            "risk_explanation": "This phrase creates pressure to act quickly.",
                        }
                    ],
                    "short_explanation": "This message is risky because it uses urgent scam-like language.",
                }
            )

        return json.dumps(
            {
                "classification": "safe",
                "scam_subtype": "none",
                "suspicious_phrases": [],
                "short_explanation": "This message appears safe.",
            }
        )

    return demo_model_callable


def main() -> None:

    demo_model_callable = build_demo_model_callable()

    # Example to see if it works.
    example_message = "Urgent: verify your account now and click the link to avoid suspension."

    prediction = predict_message(
        message_text=example_message,
        model_callable=demo_model_callable,
        example_rows=None,
    )

    print(json.dumps(prediction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()