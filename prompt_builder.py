from __future__ import annotations

import json
from typing import Any


# This is the system instruction for the LLM, it tells the model exactly what role it should play.
default_system_instruction = (
    "You analyze SMS messages for scam risk. "
    "You classify each message as safe or scam. "
    "You highlight suspicious phrases when they exist. "
    "You explain briefly why each flagged phrase is risky. "
    "Always return valid JSON."
)


def parse_suspicious_phrases_json(suspicious_phrases_json_text: Any) -> list[dict[str, Any]]:
    # Turn the stored json text into a clean Python list.
    # If the field is empty or broken, return an empty list.
    if suspicious_phrases_json_text is None:
        return []

    suspicious_phrases_text = str(suspicious_phrases_json_text).strip()

    if not suspicious_phrases_text:
        return []

    try:
        parsed_value = json.loads(suspicious_phrases_text)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed_value, list):
        return []

    cleaned_annotation_list: list[dict[str, Any]] = []

    for annotation in parsed_value:
        if not isinstance(annotation, dict):
            continue

        phrase_text = str(annotation.get("phrase_text", "")).strip()
        risk_category = str(annotation.get("risk_category", "")).strip()
        risk_explanation = str(annotation.get("risk_explanation", "")).strip()

        if not phrase_text or not risk_category:
            continue

        cleaned_annotation_list.append(
            {
                "phrase_text": phrase_text,
                "risk_category": risk_category,
                "risk_explanation": risk_explanation,
            }
        )

    return cleaned_annotation_list


def format_risk_category_name(risk_category: str) -> str:
    # Make category names more readable in explanations.
    return risk_category.replace("_", " ")


def build_short_explanation_text(
    label_name: str,
    scam_subtype: str,
    suspicious_phrase_list: list[dict[str, Any]],
) -> str:
    # Build one short explanation for the whole message. Keep it simple and easy for the model to learn from.
    unique_risk_category_list: list[str] = []
    seen_risk_categories: set[str] = set()

    for suspicious_phrase in suspicious_phrase_list:
        risk_category = str(suspicious_phrase.get("risk_category", "")).strip()

        if not risk_category:
            continue

        if risk_category in seen_risk_categories:
            continue

        seen_risk_categories.add(risk_category)
        unique_risk_category_list.append(format_risk_category_name(risk_category))

    if label_name == "scam":
        if unique_risk_category_list:
            risk_summary_text = ", ".join(unique_risk_category_list)

            if scam_subtype and scam_subtype not in {"", "none", "general_scam"}:
                readable_subtype_text = scam_subtype.replace("_", " ")
                return (
                    "This message is risky because it includes "
                    f"{risk_summary_text} language and resembles a {readable_subtype_text} scam."
                )

            return f"This message is risky because it includes {risk_summary_text} language."

        if scam_subtype and scam_subtype not in {"", "none"}:
            readable_subtype_text = scam_subtype.replace("_", " ")
            return f"This message is risky because the dataset labels it as a {readable_subtype_text} scam."

        return "This message is risky because the dataset labels it as a scam."

    if unique_risk_category_list:
        risk_summary_text = ", ".join(unique_risk_category_list)
        return (
            "The dataset labels this message as safe, "
            f"but it still contains phrases that may look risky in isolation, such as {risk_summary_text} language."
        )

    return "The dataset labels this message as safe and no suspicious phrases were flagged."


def build_response_dictionary(dataset_row: dict[str, Any]) -> dict[str, Any]:
    #build the expected JSON style answer for one dataset row.
    label_name = str(dataset_row.get("label_name", "safe")).strip() or "safe"
    scam_subtype = str(dataset_row.get("scam_subtype", "none")).strip() or "none"

    suspicious_phrase_list = parse_suspicious_phrases_json(
        dataset_row.get("suspicious_phrases_json")
    )

    if label_name == "safe":
        normalized_scam_subtype = "none"
    else:
        normalized_scam_subtype = scam_subtype if scam_subtype not in {"", "none"} else "general_scam"

    return {
        "classification": label_name,
        "scam_subtype": normalized_scam_subtype,
        "suspicious_phrases": suspicious_phrase_list,
        "short_explanation": build_short_explanation_text(
            label_name=label_name,
            scam_subtype=normalized_scam_subtype,
            suspicious_phrase_list=suspicious_phrase_list,
        ),
    }


def build_user_message_text(message_text: str) -> str:
    # Build the user facing task prompt for one SMS message.
    return (
        "Analyze the following SMS message.\n"
        "Classify it as safe or scam.\n"
        'Use "none" for scam_subtype when the message is safe.\n'
        "Use an empty list for suspicious_phrases when no suspicious phrase applies.\n"
        "Each suspicious phrase must include phrase_text, risk_category, and risk_explanation.\n"
        "Return valid JSON with these keys: classification, scam_subtype, suspicious_phrases, short_explanation.\n\n"
        f"SMS message:\n{message_text}"
    )


def build_prompt_parts(dataset_row: dict[str, Any]) -> dict[str, str]:
    # Build the basic prompt parts.
    # IMPORTANT: This is useful if another file wants system/user text separately.
    message_text = str(dataset_row.get("message_text", "")).strip()

    return {
        "system_instruction": default_system_instruction,
        "user_message": build_user_message_text(message_text),
    }


def build_prompt_text(dataset_row: dict[str, Any]) -> str:
    # Build one plain-text prompt, easiest format for prompt based inference.
    prompt_parts = build_prompt_parts(dataset_row)

    return (
        f"System:\n{prompt_parts['system_instruction']}\n\n"
        f"User:\n{prompt_parts['user_message']}\n\n"
        "Assistant:\n"
    )


def build_example_output_text(dataset_row: dict[str, Any]) -> str:
    # Build the target output text as pretty JSON.
    response_dictionary = build_response_dictionary(dataset_row)

    return json.dumps(
        response_dictionary,
        indent=2,
        ensure_ascii=False,
    )


def build_full_example_text(dataset_row: dict[str, Any]) -> str:
    # Build a full example with prompt + answer. Useful for few-shot examples, idk. 
    prompt_text = build_prompt_text(dataset_row)
    output_text = build_example_output_text(dataset_row)

    return prompt_text + output_text


def build_inference_prompt_from_message(message_text: str) -> str:
    # Build a prompt when you only have raw message text and do not have a full dataset row.
    dataset_row = {
        "message_text": message_text,
    }

    return build_prompt_text(dataset_row)


def build_few_shot_example(dataset_row: dict[str, Any]) -> dict[str, str]:
    # Return one example in structured form.
    return {
        "input": build_prompt_text(dataset_row),
        "output": build_example_output_text(dataset_row),
    }