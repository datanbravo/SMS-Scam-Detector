from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import pandas as pd

from config import (
    default_vectorizer_configuration,
    risk_category_explanation_templates,
    risk_category_output_delimiter,
    risk_category_regex_patterns,
)


# ------------------------------------------------------------
# basic stopwords
# ------------------------------------------------------------
# Only used if stopword removal is turned on.

default_stopwords = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


# ------------------------------------------------------------
# regex patterns
# ------------------------------------------------------------

phone_number_pattern = re.compile(r"\b(?:\+?\d[\d\-\s()]{7,}\d)\b")
url_pattern = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
email_address_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)


# ------------------------------------------------------------
# subtype rules
# ------------------------------------------------------------
# These are simple keyword-based subtype assignments for scam rows.

scam_subtype_patterns = [
    ("delivery_scam", ["delivery", "package", "parcel", "postage", "tracking", "usps", "ups", "fedex", "dhl"]),
    ("account_verification", ["account", "confirm", "login", "password", "profile", "security", "verify"]),
    ("payment_request", ["balance", "charge", "fee", "fine", "invoice", "overdue", "pay", "payment", "toll"]),
    ("prize_scam", ["bonus", "coupon", "gift", "lottery", "points", "prize", "reward", "winner"]),
    ("threat", ["court", "enforcement", "final notice", "judgment", "legal", "penalty", "suspend"]),
    ("urgency", ["act now", "expires", "final reminder", "immediately", "now", "today", "urgent"]),
    ("impersonation", ["amazon", "bank", "boss", "court", "dmv", "irs", "manager", "usps"]),
]


# ------------------------------------------------------------
# annotation ranking rules
# ------------------------------------------------------------
# Some categories are broad, and some are more specific.
# When overlap happens, we usually keep the more specific one.

general_risk_categories = {
    "urgency",
    "deadline_pressure",
    "link_request",
    "contact_request",
}

risk_category_specificity_rank = {
    "account_verification": 5,
    "payment_request": 5,
    "delivery_scam": 5,
    "prize_scam": 5,
    "threat": 5,
    "impersonation": 4,
    "link_request": 3,
    "contact_request": 3,
    "deadline_pressure": 2,
    "urgency": 1,
}


# Compile regex once so matching stays faster and cleaner.
compiled_risk_category_regex_patterns = {
    risk_category: tuple(
        re.compile(pattern_text, flags=re.IGNORECASE)
        for pattern_text in pattern_text_list
    )
    for risk_category, pattern_text_list in risk_category_regex_patterns.items()
}


# ------------------------------------------------------------
# text cleaning
# ------------------------------------------------------------

def clean_message_text(message_text: str, remove_stopwords: bool = False) -> str:
    # Normalize message text for ML.
    # Main steps:
    # - normalize unicode
    # - replace urls/emails/phones with stable tokens
    # - lowercase
    # - remove most punctuation
    # - normalize spaces
    normalized_text = unicodedata.normalize("NFKC", message_text)
    normalized_text = normalized_text.replace("\u00a0", " ")

    normalized_text = url_pattern.sub(" url_token ", normalized_text)
    normalized_text = email_address_pattern.sub(" email_token ", normalized_text)
    normalized_text = phone_number_pattern.sub(" phone_token ", normalized_text)

    normalized_text = normalized_text.lower()

    # Keep letters, numbers, underscores, apostrophes, and spaces.
    normalized_text = re.sub(r"[^a-z0-9_'\s]", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    if remove_stopwords:
        filtered_tokens = [
            token
            for token in normalized_text.split()
            if token not in default_stopwords
        ]
        normalized_text = " ".join(filtered_tokens)

    return normalized_text


def create_unigram_bigram_ready_text(cleaned_message_text: str) -> str:
    # Build one text field that includes:
    # - original tokens
    # - adjacent token bigrams joined with underscores
    token_list = cleaned_message_text.split()

    unigram_list = list(token_list)
    bigram_list: list[str] = []

    for token_index in range(len(token_list) - 1):
        first_token = token_list[token_index]
        second_token = token_list[token_index + 1]
        bigram_list.append(f"{first_token}_{second_token}")

    unigram_and_bigram_list = unigram_list + bigram_list
    return " ".join(unigram_and_bigram_list)


# ------------------------------------------------------------
# annotation building
# ------------------------------------------------------------

def build_suspicious_phrase_annotation(
    message_text: str,
    start_index: int,
    end_index: int,
    risk_category: str,
) -> dict[str, Any]:
    # Build one suspicious phrase annotation.
    phrase_text = message_text[start_index:end_index]
    risk_explanation = risk_category_explanation_templates[risk_category]

    return {
        "phrase_text": phrase_text,
        "start_index": start_index,
        "end_index": end_index,
        "risk_category": risk_category,
        "risk_explanation": risk_explanation,
    }


def extract_annotation_candidates_from_message_text(message_text: str) -> list[dict[str, Any]]:
    # Extract all raw regex matches before overlap cleanup.
    candidate_annotation_list: list[dict[str, Any]] = []
    seen_annotation_keys: set[tuple[int, int, str]] = set()

    for risk_category, compiled_pattern_list in compiled_risk_category_regex_patterns.items():
        for compiled_pattern in compiled_pattern_list:
            for match in compiled_pattern.finditer(message_text):
                start_index = match.start()
                end_index = match.end()
                annotation_key = (start_index, end_index, risk_category)

                if annotation_key in seen_annotation_keys:
                    continue

                if not message_text[start_index:end_index].strip():
                    continue

                candidate_annotation_list.append(
                    build_suspicious_phrase_annotation(
                        message_text=message_text,
                        start_index=start_index,
                        end_index=end_index,
                        risk_category=risk_category,
                    )
                )
                seen_annotation_keys.add(annotation_key)

    return candidate_annotation_list


# ------------------------------------------------------------
# annotation overlap helpers
# ------------------------------------------------------------

def get_annotation_length(annotation: dict[str, Any]) -> int:
    # Return character length of one annotation.
    return int(annotation["end_index"]) - int(annotation["start_index"])


def get_annotation_word_count(annotation: dict[str, Any]) -> int:
    # Return rough word count of one annotation.
    return len(str(annotation["phrase_text"]).split())


def get_annotation_specificity_rank(annotation: dict[str, Any]) -> int:
    # Return the configured specificity rank for a category.
    risk_category = str(annotation["risk_category"])
    return risk_category_specificity_rank.get(risk_category, 0)


def annotation_category_is_general(annotation: dict[str, Any]) -> bool:
    # Check whether the category is one of the general categories.
    return str(annotation["risk_category"]) in general_risk_categories


def annotations_overlap(
    first_annotation: dict[str, Any],
    second_annotation: dict[str, Any],
) -> bool:
    # Check if two annotation spans overlap.
    return (
        int(first_annotation["start_index"]) < int(second_annotation["end_index"])
        and int(second_annotation["start_index"]) < int(first_annotation["end_index"])
    )


def calculate_overlap_character_count(
    first_annotation: dict[str, Any],
    second_annotation: dict[str, Any],
) -> int:
    # Return how many characters two spans share.
    overlap_start_index = max(int(first_annotation["start_index"]), int(second_annotation["start_index"]))
    overlap_end_index = min(int(first_annotation["end_index"]), int(second_annotation["end_index"]))
    return max(0, overlap_end_index - overlap_start_index)


def annotation_contains_other(
    container_annotation: dict[str, Any],
    contained_annotation: dict[str, Any],
) -> bool:
    # Check if one annotation fully contains another.
    return (
        int(container_annotation["start_index"]) <= int(contained_annotation["start_index"])
        and int(container_annotation["end_index"]) >= int(contained_annotation["end_index"])
    )


def annotations_overlap_heavily(
    first_annotation: dict[str, Any],
    second_annotation: dict[str, Any],
) -> bool:
    # Decide if overlap is strong enough to matter.
    # We use 80% of the shorter span as the threshold.
    overlap_character_count = calculate_overlap_character_count(first_annotation, second_annotation)
    shorter_annotation_length = min(get_annotation_length(first_annotation), get_annotation_length(second_annotation))

    if shorter_annotation_length <= 0:
        return False

    return overlap_character_count / shorter_annotation_length >= 0.8


def annotations_are_distinct_enough(
    first_annotation: dict[str, Any],
    second_annotation: dict[str, Any],
) -> bool:
    # Decide whether two overlapping annotations should both stay.
    if not annotations_overlap(first_annotation, second_annotation):
        return True

    if not annotations_overlap_heavily(first_annotation, second_annotation):
        return True

    if (
        not annotation_contains_other(first_annotation, second_annotation)
        and not annotation_contains_other(second_annotation, first_annotation)
    ):
        return True

    if first_annotation["risk_category"] == second_annotation["risk_category"]:
        return False

    first_annotation_is_general = annotation_category_is_general(first_annotation)
    second_annotation_is_general = annotation_category_is_general(second_annotation)

    if first_annotation_is_general != second_annotation_is_general:
        return False

    if first_annotation_is_general and second_annotation_is_general:
        return False

    return True


def choose_preferred_overlapping_annotation(
    first_annotation: dict[str, Any],
    second_annotation: dict[str, Any],
) -> dict[str, Any]:
    # Pick the better annotation when two overlapping ones should not both stay.
    first_annotation_is_general = annotation_category_is_general(first_annotation)
    second_annotation_is_general = annotation_category_is_general(second_annotation)

    if first_annotation_is_general != second_annotation_is_general:
        return second_annotation if first_annotation_is_general else first_annotation

    first_annotation_word_count = get_annotation_word_count(first_annotation)
    second_annotation_word_count = get_annotation_word_count(second_annotation)

    first_annotation_length = get_annotation_length(first_annotation)
    second_annotation_length = get_annotation_length(second_annotation)

    first_annotation_specificity_rank = get_annotation_specificity_rank(first_annotation)
    second_annotation_specificity_rank = get_annotation_specificity_rank(second_annotation)

    if annotation_contains_other(first_annotation, second_annotation) or annotation_contains_other(second_annotation, first_annotation):
        if first_annotation_word_count != second_annotation_word_count:
            return first_annotation if first_annotation_word_count > second_annotation_word_count else second_annotation

        if first_annotation_length != second_annotation_length:
            return first_annotation if first_annotation_length > second_annotation_length else second_annotation

    if first_annotation_specificity_rank != second_annotation_specificity_rank:
        return first_annotation if first_annotation_specificity_rank > second_annotation_specificity_rank else second_annotation

    if first_annotation_word_count != second_annotation_word_count:
        return first_annotation if first_annotation_word_count > second_annotation_word_count else second_annotation

    if first_annotation_length != second_annotation_length:
        return first_annotation if first_annotation_length > second_annotation_length else second_annotation

    return first_annotation


def merge_overlapping_annotations(annotation_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Clean overlap conflicts so the final annotation list stays readable.
    sorted_annotation_list = sorted(
        annotation_list,
        key=lambda annotation: (
            int(annotation["start_index"]),
            -get_annotation_length(annotation),
            -get_annotation_specificity_rank(annotation),
            str(annotation["risk_category"]),
            str(annotation["phrase_text"]).lower(),
        ),
    )

    cleaned_annotation_list: list[dict[str, Any]] = []

    for candidate_annotation in sorted_annotation_list:
        keep_candidate_annotation = True
        existing_indices_to_remove: list[int] = []

        overlapping_annotation_indices = [
            existing_index
            for existing_index, existing_annotation in enumerate(cleaned_annotation_list)
            if annotations_overlap(existing_annotation, candidate_annotation)
        ]

        for existing_index in overlapping_annotation_indices:
            existing_annotation = cleaned_annotation_list[existing_index]

            if annotations_are_distinct_enough(existing_annotation, candidate_annotation):
                continue

            preferred_annotation = choose_preferred_overlapping_annotation(
                first_annotation=existing_annotation,
                second_annotation=candidate_annotation,
            )

            if preferred_annotation is existing_annotation:
                keep_candidate_annotation = False
                break

            existing_indices_to_remove.append(existing_index)

        if not keep_candidate_annotation:
            continue

        for existing_index in reversed(sorted(set(existing_indices_to_remove))):
            cleaned_annotation_list.pop(existing_index)

        cleaned_annotation_list.append(candidate_annotation)

    cleaned_annotation_list.sort(
        key=lambda annotation: (
            int(annotation["start_index"]),
            int(annotation["end_index"]),
            str(annotation["risk_category"]),
        )
    )

    return cleaned_annotation_list


def extract_suspicious_phrase_annotations(message_text: str) -> list[dict[str, Any]]:
    # Extract and clean suspicious phrase annotations from one message.
    if not message_text.strip():
        return []

    candidate_annotation_list = extract_annotation_candidates_from_message_text(message_text)
    return merge_overlapping_annotations(candidate_annotation_list)


def build_risk_categories_present_text(annotation_list: list[dict[str, Any]]) -> str:
    # Build a compact category summary string for one row.
    ordered_risk_category_list: list[str] = []
    seen_risk_categories: set[str] = set()

    for annotation in annotation_list:
        risk_category = str(annotation["risk_category"])
        if risk_category in seen_risk_categories:
            continue

        seen_risk_categories.add(risk_category)
        ordered_risk_category_list.append(risk_category)

    return risk_category_output_delimiter.join(ordered_risk_category_list)


def convert_annotation_list_to_row_fields(annotation_list: list[dict[str, Any]]) -> dict[str, Any]:
    # Convert annotation list into row-friendly columns.
    return {
        "annotation_count": len(annotation_list),
        "risk_categories_present": build_risk_categories_present_text(annotation_list),
        "suspicious_phrases_json": json.dumps(annotation_list, ensure_ascii=False),
    }


# ------------------------------------------------------------
# subtype and lightweight features
# ------------------------------------------------------------

def assign_optional_scam_subtype(cleaned_message_text: str, label: int) -> str:
    # Assign a simple scam subtype using keyword rules.
    if label == 0:
        return "none"

    for subtype_name, keyword_list in scam_subtype_patterns:
        if any(keyword in cleaned_message_text for keyword in keyword_list):
            return subtype_name

    return "general_scam"


def create_basic_text_features(message_text: str, cleaned_message_text: str) -> dict[str, int]:
    # Create a few lightweight numeric helper features.
    return {
        "message_length_characters": len(message_text),
        "message_length_words": len(message_text.split()),
        "cleaned_token_count": len(cleaned_message_text.split()),
        "digit_count": sum(character.isdigit() for character in message_text),
        "exclamation_mark_count": message_text.count("!"),
        "contains_url_like_text": int(bool(url_pattern.search(message_text))),
        "contains_phone_number_like_text": int(bool(phone_number_pattern.search(message_text))),
    }


def build_recommended_vectorizer_configuration() -> dict[str, Any]:
    # Return the recommended traditional vectorizer configuration.
    return dict(default_vectorizer_configuration)


# ------------------------------------------------------------
# main dataset preparation
# ------------------------------------------------------------

def prepare_dataset_for_machine_learning(
    raw_dataset: pd.DataFrame,
    minimum_message_length: int,
    remove_stopwords: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # Turn the raw rows into a processed dataset ready for ML.
    skipped_rows: list[dict[str, Any]] = []
    prepared_records: list[dict[str, Any]] = []

    for row in raw_dataset.to_dict(orient="records"):
        message_text_value = row.get("message_text")
        message_text = "" if message_text_value is None else str(message_text_value).strip()

        if not message_text:
            skipped_rows.append(
                {
                    "source_name": row.get("source_name"),
                    "source_url": row.get("source_url"),
                    "reason": "Empty message text.",
                    "message_preview": "",
                }
            )
            continue

        if len(message_text) < minimum_message_length:
            skipped_rows.append(
                {
                    "source_name": row.get("source_name"),
                    "source_url": row.get("source_url"),
                    "reason": "Message text was shorter than the minimum allowed length.",
                    "message_preview": message_text[:120],
                }
            )
            continue

        cleaned_message_text = clean_message_text(
            message_text=message_text,
            remove_stopwords=remove_stopwords,
        )

        if not cleaned_message_text:
            skipped_rows.append(
                {
                    "source_name": row.get("source_name"),
                    "source_url": row.get("source_url"),
                    "reason": "Message text became empty after cleaning.",
                    "message_preview": message_text[:120],
                }
            )
            continue

        label = int(row.get("label", 0))
        label_name = "scam" if label == 1 else "safe"

        existing_subtype = str(row.get("scam_subtype", "") or "").strip()
        scam_subtype = existing_subtype or assign_optional_scam_subtype(cleaned_message_text, label)

        unigram_bigram_ready_text = create_unigram_bigram_ready_text(cleaned_message_text)

        annotation_list = extract_suspicious_phrase_annotations(message_text)
        annotation_row_fields = convert_annotation_list_to_row_fields(annotation_list)

        prepared_record = dict(row)
        prepared_record["message_text"] = message_text
        prepared_record["cleaned_message_text"] = cleaned_message_text
        prepared_record["unigram_bigram_ready_text"] = unigram_bigram_ready_text
        prepared_record["label"] = label
        prepared_record["label_name"] = label_name
        prepared_record["scam_subtype"] = scam_subtype
        prepared_record.update(annotation_row_fields)
        prepared_record.update(create_basic_text_features(message_text, cleaned_message_text))

        prepared_records.append(prepared_record)

    prepared_dataset = pd.DataFrame(prepared_records)

    if prepared_dataset.empty:
        return prepared_dataset, skipped_rows

    # If the same cleaned text appears with both labels,
    # keeping both would teach the model contradictory information.
    conflicting_rows = prepared_dataset[
        prepared_dataset.groupby("cleaned_message_text")["label"].transform("nunique") > 1
    ]

    if not conflicting_rows.empty:
        for _, conflicting_row in conflicting_rows.iterrows():
            skipped_rows.append(
                {
                    "source_name": conflicting_row.get("source_name"),
                    "source_url": conflicting_row.get("source_url"),
                    "reason": "Conflicting duplicate after cleaning.",
                    "message_preview": str(conflicting_row.get("message_text", ""))[:120],
                }
            )

        prepared_dataset = prepared_dataset[
            prepared_dataset.groupby("cleaned_message_text")["label"].transform("nunique") == 1
        ].copy()

    # Drop exact duplicates after cleaning, keeping the first.
    duplicate_rows = prepared_dataset[
        prepared_dataset.duplicated(subset=["cleaned_message_text"], keep="first")
    ]

    if not duplicate_rows.empty:
        for _, duplicate_row in duplicate_rows.iterrows():
            skipped_rows.append(
                {
                    "source_name": duplicate_row.get("source_name"),
                    "source_url": duplicate_row.get("source_url"),
                    "reason": "Duplicate message after cleaning.",
                    "message_preview": str(duplicate_row.get("message_text", ""))[:120],
                }
            )

        prepared_dataset = prepared_dataset.drop_duplicates(
            subset=["cleaned_message_text"],
            keep="first",
        ).copy()

    prepared_dataset = prepared_dataset.reset_index(drop=True)

    return prepared_dataset, skipped_rows