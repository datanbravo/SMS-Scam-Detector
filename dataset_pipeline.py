from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

from config import (
    data_directory,
    downloads_directory,
    final_dataset_column_order,
    full_dataset_file_name,
    logs_directory,
    message_filter_stop_contains,
    message_filter_stop_prefixes,
    metadata_directory,
    metadata_file_name,
    minimum_message_length,
    processed_data_directory,
    random_seed,
    raw_data_directory,
    raw_dataset_file_name,
    request_timeout_seconds,
    required_dataset_columns,
    scam_action_words,
    scam_example_cue_words,
    skipped_rows_log_file_name,
    source_manifest,
    source_validation_log_file_name,
    synthetic_scam_message_count,
    test_dataset_file_name,
    test_split_ratio,
    train_dataset_file_name,
    train_split_ratio,
    use_stopword_removal,
    SourceDefinition,
)
from preprocessing import (
    build_recommended_vectorizer_configuration,
    build_risk_categories_present_text,
    prepare_dataset_for_machine_learning,
)
from source_validation import (
    classify_request_failure,
    create_default_request_session,
    validate_download_source,
    validate_source_definition,
    validate_web_page_target,
)
from synthetic_data import generate_synthetic_scam_messages


#Thee are failure types that usually mean the internet/source was unavailable, not that the code was wrong.
network_failure_categories = {
    "dns_or_host_resolution_failure",
    "http_error",
    "network_connection_failure",
    "network_request_failure",
    "network_timeout",
}


# basic file and folder helpers

def ensure_output_directories_exist() -> None:
    #Create all folders the pipeline needs.
    directory_list = [
        data_directory,
        raw_data_directory,
        downloads_directory,
        processed_data_directory,
        metadata_directory,
        logs_directory,
    ]

    for directory in directory_list:
        directory.mkdir(parents=True, exist_ok=True)


def get_current_timestamp_in_universal_time() -> str:
    #Return a UTC timestamp for logs and metadata.
    return datetime.now(timezone.utc).isoformat()


def save_dataframe_as_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    #Save a dataframe as a csv file.
    dataframe.to_csv(output_path, index=False)


def save_json_to_file(content_value: dict[str, Any] | list[dict[str, Any]], output_path: Path) -> None:
    # Save a dictionary or list to json.
    output_path.write_text(json.dumps(content_value, indent=2), encoding="utf-8")


def sanitize_file_name(value: str) -> str:
    # Make a safe file name for downloaded files.
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()



# row builders

def build_dataset_record(
    message_text: str,
    label: int,
    source_name: str,
    source_url: str,
    data_origin_type: str,
    is_synthetic: bool,
    scam_subtype: str = "",
) -> dict[str, Any]:
    # Build one dataset row in a consistent shape.
    return {
        "message_text": message_text,
        "label": int(label),
        "label_name": "scam" if int(label) == 1 else "safe",
        "source_name": source_name,
        "source_url": source_url,
        "data_origin_type": data_origin_type,
        "is_synthetic": bool(is_synthetic),
        "split": "",
        "scam_subtype": scam_subtype,
    }


def build_skipped_row(
    source_name: str,
    source_url: str,
    reason: str,
    failure_category: str = "",
    message_preview: str = "",
) -> dict[str, str]:
    #Build one skipped-row log entry.
    return {
        "source_name": source_name,
        "source_url": source_url,
        "reason": reason,
        "failure_category": failure_category,
        "message_preview": message_preview,
    }


def write_processing_logs(
    validation_log: list[dict[str, Any]],
    skipped_rows_log: list[dict[str, Any]],
) -> None:
    # Write the source validation and skippedrow logs.
    save_json_to_file(validation_log, logs_directory / source_validation_log_file_name)
    save_json_to_file(skipped_rows_log, logs_directory / skipped_rows_log_file_name)


# uci dataset extraction

def extract_records_from_uc_irvine_sms_spam_collection(
    request_session: requests.Session,
    source_definition: SourceDefinition,
    validation_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Download and parse the UCI SMS Spam Collection.
    #This is the main real labeled source for safe/scam messages.
    collected_records: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    validation_result = validate_download_source(request_session, source_definition)
    validation_log.append(validation_result.to_dictionary())

    target_web_address = source_definition.downloadable_file_web_address or source_definition.source_url

    if not validation_result.is_allowed:
        skipped_rows.append(
            build_skipped_row(
                source_name=source_definition.source_name,
                source_url=target_web_address,
                reason=validation_result.reason,
                failure_category=validation_result.failure_category or "",
            )
        )
        return collected_records, skipped_rows

    try:
        response = request_session.get(
            target_web_address,
            timeout=request_timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        failure_category, reason = classify_request_failure(error)
        skipped_rows.append(
            build_skipped_row(
                source_name=source_definition.source_name,
                source_url=target_web_address,
                reason=f"Download failed: {reason}",
                failure_category=failure_category,
            )
        )
        return collected_records, skipped_rows

    download_file_name = f"{sanitize_file_name(source_definition.source_name)}.zip"
    download_file_path = downloads_directory / download_file_name
    download_file_path.write_bytes(response.content)

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive_file:
            archive_member_names = archive_file.namelist()

            # The UCI zip normally contains a file called SMSSpamCollection.
            message_file_name = next(
                (
                    archive_member_name
                    for archive_member_name in archive_member_names
                    if Path(archive_member_name).name.lower() == "smsspamcollection"
                ),
                None,
            )

            if message_file_name is None:
                skipped_rows.append(
                    build_skipped_row(
                        source_name=source_definition.source_name,
                        source_url=target_web_address,
                        reason="Could not find the expected SMSSpamCollection file in the archive.",
                        failure_category="download_parse_error",
                    )
                )
                return collected_records, skipped_rows

            message_file_bytes = archive_file.read(message_file_name)

    except (zipfile.BadZipFile, OSError, KeyError) as error:
        skipped_rows.append(
            build_skipped_row(
                source_name=source_definition.source_name,
                source_url=target_web_address,
                reason=f"Downloaded file could not be read as the expected archive: {error}",
                failure_category="download_parse_error",
            )
        )
        return collected_records, skipped_rows

    for raw_line in message_file_bytes.decode("utf-8", errors="ignore").splitlines():
        line_text = raw_line.strip()

        if not line_text:
            continue

        line_parts = line_text.split("\t", maxsplit=1)
        if len(line_parts) != 2:
            skipped_rows.append(
                build_skipped_row(
                    source_name=source_definition.source_name,
                    source_url=source_definition.source_url,
                    reason="Unexpected line format in the UCI source file.",
                    failure_category="source_file_parse_error",
                    message_preview=line_text[:120],
                )
            )
            continue

        source_label, message_text = line_parts
        normalized_label = source_label.strip().lower()

        if normalized_label not in {"ham", "spam"}:
            skipped_rows.append(
                build_skipped_row(
                    source_name=source_definition.source_name,
                    source_url=source_definition.source_url,
                    reason="Unexpected label in the UCI source file.",
                    failure_category="source_file_parse_error",
                    message_preview=line_text[:120],
                )
            )
            continue

        collected_records.append(
            build_dataset_record(
                message_text=message_text.strip(),
                label=1 if normalized_label == "spam" else 0,
                source_name=source_definition.source_name,
                source_url=source_definition.source_url,
                data_origin_type="downloadable_dataset",
                is_synthetic=False,
            )
        )

    return collected_records, skipped_rows


# public web page extraction

def split_text_into_sentences(text_value: str) -> list[str]:
    #Split visible page text into rough sentence units.
    sentence_candidates = re.split(r"(?<=[.!?])\s+", text_value)
    return [sentence.strip() for sentence in sentence_candidates if sentence.strip()]


def normalize_extracted_message_candidate(message_candidate: str) -> str:
    # Clean up extracted scam example text before keepng it.
    normalized_candidate = re.sub(r"\(blurred link\)", "", message_candidate, flags=re.IGNORECASE)
    normalized_candidate = normalized_candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    normalized_candidate = normalized_candidate.strip(" :;,-")
    normalized_candidate = normalized_candidate.strip('"')
    normalized_candidate = re.sub(r"\s+", " ", normalized_candidate).strip()
    return normalized_candidate


def extract_quoted_message_candidates(visible_text: str) -> list[str]:
    # Find quoted message like snippets inside page text.
    quoted_candidates: list[str] = []

    for match in re.finditer(r"[“\"]([^\"”]{6,240})[”\"]", visible_text):
        quoted_candidates.append(match.group(1).strip())

    return quoted_candidates


def extract_message_candidates_from_alt_text(image_alternative_text: str) -> list[str]:
    #Try to pull scam-message examples from image alt text. Some public pages store example text inside screenshot alt text... (may be taken off if not useful later).
    normalized_alternative_text = " ".join(image_alternative_text.split())
    extracted_candidates = extract_quoted_message_candidates(normalized_alternative_text)

    pattern_list = [
        r"(?:text blurb|text message|message|screenshot|image)[^:]{0,80}:\s*(.+)$",
        r"(?:text blurb|text message|message|screenshot|image)[^:]{0,80}\bsays?\s*:?\s*(.+)$",
    ]

    for pattern in pattern_list:
        match = re.search(pattern, normalized_alternative_text, flags=re.IGNORECASE)
        if match:
            extracted_candidates.append(match.group(1).strip())

    return extracted_candidates


def extract_message_candidates_from_visible_text(visible_text: str) -> list[str]:
    # Pull message like examples from page paragraphs and lists. We only want obvious message examples, not the whole article.
    extracted_candidates = extract_quoted_message_candidates(visible_text)

    pattern_list = [
        r"(?:text(?: message)?|message)\s+(?:says?|said|reading|reads)\s*[\"“]?(.+?)[\"”]?$",
        r"(?:text(?: message)?|message)\s+(?:warning|warns)\s+that\s+(.+?)$",
        r"(?:text(?: message)?|message)\s+(?:claiming|claims)\s+(.+?)$",
        r"(?:unexpected text|text(?: message)?|message)\s+looks like it'?s from .+?,\s+claiming\s+(.+?)$",
        r"(?:scammers|they)\s+say you need to\s+(.+?)$",
        r"(?:it|the text)\s+offers\s+(.+?)$",
        r"(?:it|the text)\s+claims\s+(.+?)$",
        r"saying\s+(.+?)$",
        r"promising\s+(.+?)$",
        r"offering\s+(.+?)$",
    ]

    for sentence in split_text_into_sentences(visible_text):
        normalized_sentence = " ".join(sentence.split())

        for pattern in pattern_list:
            match = re.search(pattern, normalized_sentence, flags=re.IGNORECASE)
            if match:
                extracted_candidates.append(match.group(1).strip())

    return extracted_candidates


def looks_like_explicit_message_example(message_candidate: str) -> bool:
    # Decide whether extracted text actually looks like a scam message, and keeps the dataset from filling up with article advice text (it has tons of it).

    normalized_candidate = normalize_extracted_message_candidate(message_candidate)
    normalized_candidate_lower = normalized_candidate.lower()
    word_list = normalized_candidate_lower.split()

    if len(word_list) < 5 or len(word_list) > 45:
        return False

    if any(normalized_candidate_lower.startswith(prefix) for prefix in message_filter_stop_prefixes):
        return False

    if any(stop_phrase in normalized_candidate_lower for stop_phrase in message_filter_stop_contains):
        return False

    if "scammer" in normalized_candidate_lower or "scammers" in normalized_candidate_lower:
        return False

    has_scam_cue_word = any(keyword in normalized_candidate_lower for keyword in scam_example_cue_words)
    has_action_word = any(keyword in normalized_candidate_lower for keyword in scam_action_words)
    has_second_person_language = any(word in {"you", "your", "we", "our"} for word in word_list)
    has_link_like_reference = "link" in normalized_candidate_lower or "http" in normalized_candidate_lower

    if not has_scam_cue_word:
        return False

    if not (has_action_word or has_link_like_reference):
        return False

    if not (has_second_person_language or has_action_word):
        return False

    return True


def extract_message_examples_from_page_markup(page_markup_text: str) -> list[str]:
    # Extract explicit scam-message examples from one page. We collect from: image alt text, paragraphs, lists, adn blockquotes.

    parsed_document = BeautifulSoup(page_markup_text, "html.parser")

    candidate_list: list[str] = []
    normalized_candidates: list[str] = []
    seen_candidates: set[str] = set()

    for image_tag in parsed_document.find_all("img"):
        image_alternative_text = (image_tag.get("alt") or "").strip()
        if image_alternative_text:
            candidate_list.extend(extract_message_candidates_from_alt_text(image_alternative_text))

    for content_tag in parsed_document.find_all(["p", "li", "blockquote"]):
        visible_text = content_tag.get_text(" ", strip=True)
        if visible_text:
            candidate_list.extend(extract_message_candidates_from_visible_text(visible_text))

    for candidate in candidate_list:
        normalized_candidate = normalize_extracted_message_candidate(candidate)
        normalized_candidate_lower = normalized_candidate.lower()

        if not normalized_candidate:
            continue

        if not looks_like_explicit_message_example(normalized_candidate):
            continue

        if normalized_candidate_lower in seen_candidates:
            continue

        seen_candidates.add(normalized_candidate_lower)
        normalized_candidates.append(normalized_candidate)

    return normalized_candidates


def extract_records_from_public_web_pages(
    request_session: requests.Session,
    source_definition: SourceDefinition,
    validation_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Collect scam-message examples from public alert pages. ONLY USED IF: the source is allowed, robots.txt allows access, and the page actually contains message like examples.
    
    collected_records: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for page_web_address in source_definition.page_web_addresses:
        page_validation = validate_web_page_target(
            request_session=request_session,
            source_definition=source_definition,
            page_web_address=page_web_address,
        )
        validation_log.append(page_validation.to_dictionary())

        if not page_validation.is_allowed:
            skipped_rows.append(
                build_skipped_row(
                    source_name=source_definition.source_name,
                    source_url=page_web_address,
                    reason=page_validation.reason,
                    failure_category=page_validation.failure_category or "",
                )
            )
            continue

        try:
            response = request_session.get(
                page_web_address,
                timeout=request_timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            failure_category, reason = classify_request_failure(error)
            skipped_rows.append(
                build_skipped_row(
                    source_name=source_definition.source_name,
                    source_url=page_web_address,
                    reason=f"Page download failed: {reason}",
                    failure_category=failure_category,
                )
            )
            continue

        message_examples = extract_message_examples_from_page_markup(response.text)

        if not message_examples:
            skipped_rows.append(
                build_skipped_row(
                    source_name=source_definition.source_name,
                    source_url=page_web_address,
                    reason="No explicit message examples were extracted from the page.",
                    failure_category="empty_extraction_result",
                )
            )
            continue

        for message_text in message_examples:
            collected_records.append(
                build_dataset_record(
                    message_text=message_text,
                    label=1,
                    source_name=source_definition.source_name,
                    source_url=page_web_address,
                    data_origin_type="public_web_example",
                    is_synthetic=False,
                )
            )

    return collected_records, skipped_rows


# source collection

def collect_records_from_source(
    request_session: requests.Session,
    source_definition: SourceDefinition,
    validation_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # Collect records from one source and build a short sumarry.
    collected_records: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    source_summary = {
        "source_name": source_definition.source_name,
        "source_type": source_definition.source_type,
        "records_collected": 0,
        "used_for_dataset": False,
        "is_synthetic_source": False,
        "failure_categories": [],
        "skip_reason": "",
    }

    source_definition_result = validate_source_definition(source_definition)
    validation_log.append(source_definition_result.to_dictionary())

    if not source_definition_result.is_allowed:
        source_summary["skip_reason"] = source_definition_result.reason
        return collected_records, skipped_rows, source_summary

    if source_definition.source_type == "downloadable_dataset":
        collected_records, skipped_rows = extract_records_from_uc_irvine_sms_spam_collection(
            request_session=request_session,
            source_definition=source_definition,
            validation_log=validation_log,
        )
    elif source_definition.source_type == "web_page_examples":
        collected_records, skipped_rows = extract_records_from_public_web_pages(
            request_session=request_session,
            source_definition=source_definition,
            validation_log=validation_log,
        )
    else:
        skipped_rows.append(
            build_skipped_row(
                source_name=source_definition.source_name,
                source_url=source_definition.source_url,
                reason=f"Unsupported source type '{source_definition.source_type}'.",
            )
        )

    source_summary["records_collected"] = len(collected_records)
    source_summary["used_for_dataset"] = len(collected_records) > 0
    source_summary["failure_categories"] = sorted(
        {
            skipped_row["failure_category"]
            for skipped_row in skipped_rows
            if skipped_row.get("failure_category")
        }
    )

    if skipped_rows and not source_summary["skip_reason"] and not source_summary["used_for_dataset"]:
        source_summary["skip_reason"] = skipped_rows[0]["reason"]

    return collected_records, skipped_rows, source_summary


# splitting and ordering

def split_dataset(prepared_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Split data into train and test dats.

    if prepared_dataset.empty:
        raise ValueError("Cannot split an empty dataset.")

    stratify_values = prepared_dataset["label"] if prepared_dataset["label"].nunique() > 1 else None

    try:
        train_dataset, test_dataset = train_test_split(
            prepared_dataset,
            test_size=test_split_ratio,
            random_state=random_seed,
            stratify=stratify_values,
        )
    except ValueError:
        # This fallback keeps the pipeline from crashing if the dataset is too small for stratification.
        train_dataset, test_dataset = train_test_split(
            prepared_dataset,
            test_size=test_split_ratio,
            random_state=random_seed,
            stratify=None,
        )

    train_dataset = train_dataset.copy().reset_index(drop=True)
    test_dataset = test_dataset.copy().reset_index(drop=True)
    train_dataset["split"] = "train"
    test_dataset["split"] = "test"

    return train_dataset, test_dataset


def order_final_dataset_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    # Put the most useful columns first in the final csv files (easier for us and the system to work with).
    ordered_column_names: list[str] = []

    for column_name in final_dataset_column_order:
        if column_name in dataset.columns:
            ordered_column_names.append(column_name)

    for column_name in dataset.columns:
        if column_name not in ordered_column_names:
            ordered_column_names.append(column_name)

    return dataset.loc[:, ordered_column_names]


# summaries and validation

def build_source_composition_summary(full_dataset: pd.DataFrame) -> dict[str, Any]:
    # Summarize how much of the dataset is real vs synthetic.
    real_dataset = full_dataset[~full_dataset["is_synthetic"]].copy()
    synthetic_dataset = full_dataset[full_dataset["is_synthetic"]].copy()

    real_label_values = set(real_dataset["label"].dropna().astype(int).tolist()) if not real_dataset.empty else set()
    final_label_values = set(full_dataset["label"].dropna().astype(int).tolist()) if not full_dataset.empty else set()
    non_synthetic_source_names = sorted(real_dataset["source_name"].dropna().astype(str).unique().tolist())

    return {
        "real_sample_count": int(len(real_dataset)),
        "synthetic_sample_count": int(len(synthetic_dataset)),
        "real_safe_sample_count": int((real_dataset["label"] == 0).sum()) if not real_dataset.empty else 0,
        "real_scam_sample_count": int((real_dataset["label"] == 1).sum()) if not real_dataset.empty else 0,
        "has_non_synthetic_source_success": len(non_synthetic_source_names) > 0,
        "non_synthetic_source_names": non_synthetic_source_names,
        "dataset_is_synthetic_only": len(real_dataset) == 0 and len(synthetic_dataset) > 0,
        "label_coverage_depends_on_synthetic_rows": bool(final_label_values - real_label_values),
    }


def build_annotation_summary(full_dataset: pd.DataFrame) -> dict[str, Any]:
    # Summarize how many phrase annotations were created.
    total_annotated_rows = int((full_dataset["annotation_count"] > 0).sum())
    total_annotation_count = int(full_dataset["annotation_count"].sum())
    annotation_count_by_risk_category: dict[str, int] = {}

    for suspicious_phrases_json_text in full_dataset["suspicious_phrases_json"].fillna("[]").astype(str).tolist():
        annotation_list = json.loads(suspicious_phrases_json_text)

        for annotation in annotation_list:
            risk_category = str(annotation.get("risk_category", "")).strip()
            if not risk_category:
                continue

            annotation_count_by_risk_category[risk_category] = annotation_count_by_risk_category.get(risk_category, 0) + 1

    return {
        "total_annotated_rows": total_annotated_rows,
        "total_annotation_count": total_annotation_count,
        "annotation_count_by_risk_category": annotation_count_by_risk_category,
    }


def validate_real_source_collection(
    source_summaries: list[dict[str, Any]],
    source_composition_summary: dict[str, Any],
) -> None:
    # IMPORTANT: Make sure the dataset is not pretending to be real if all public sources failed.
    real_source_summaries = [
        source_summary
        for source_summary in source_summaries
        if not source_summary.get("is_synthetic_source", False)
    ]

    if source_composition_summary["real_sample_count"] > 0:
        return

    real_source_failure_categories = sorted(
        {
            failure_category
            for source_summary in real_source_summaries
            for failure_category in source_summary.get("failure_categories", [])
            if failure_category
        }
    )

    if real_source_failure_categories and set(real_source_failure_categories).issubset(network_failure_categories):
        if "dns_or_host_resolution_failure" in real_source_failure_categories:
            raise RuntimeError(
                "No real dataset rows were collected because public sources were unreachable during DNS or host resolution. "
                "Check internet access or DNS and rerun."
            )

        raise RuntimeError(
            "No real dataset rows were collected because public sources were unreachable or returned network errors. "
            "Check internet access and rerun."
        )

    raise RuntimeError(
        "No real dataset rows were collected from public sources. "
        "Check network access, robots rules, and skipped log files for details."
    )


def validate_annotation_columns(full_dataset: pd.DataFrame) -> list[str]:
    # Validate annotation structure so later highlighting logic can trust the saved spans and categories.
    validation_errors: list[str] = []

    for row_number, row in full_dataset.reset_index(drop=True).iterrows():
        annotation_count_value = row.get("annotation_count", 0)
        message_text = "" if pd.isna(row.get("message_text")) else str(row.get("message_text"))
        risk_categories_present_value = row.get("risk_categories_present", "")
        suspicious_phrases_json_value = row.get("suspicious_phrases_json", "[]")

        try:
            annotation_count = int(annotation_count_value)
        except (TypeError, ValueError):
            validation_errors.append(f"Row {row_number} has a non-integer annotation_count.")
            continue

        if annotation_count < 0:
            validation_errors.append(f"Row {row_number} has a negative annotation_count.")
            continue

        suspicious_phrases_json_text = "[]" if pd.isna(suspicious_phrases_json_value) else str(suspicious_phrases_json_value)

        try:
            annotation_list = json.loads(suspicious_phrases_json_text)
        except json.JSONDecodeError:
            validation_errors.append(f"Row {row_number} has invalid suspicious_phrases_json.")
            continue

        if not isinstance(annotation_list, list):
            validation_errors.append(f"Row {row_number} suspicious_phrases_json must decode to a list.")
            continue

        if annotation_count != len(annotation_list):
            validation_errors.append(f"Row {row_number} annotation_count does not match suspicious_phrases_json length.")

        risk_categories_present = "" if pd.isna(risk_categories_present_value) else str(risk_categories_present_value)
        annotation_list_for_category_summary = [
            {"risk_category": str(annotation.get("risk_category", "")).strip()}
            for annotation in annotation_list
            if isinstance(annotation, dict) and str(annotation.get("risk_category", "")).strip()
        ]
        expected_risk_categories_present = build_risk_categories_present_text(annotation_list_for_category_summary)

        if annotation_count > 0 and not risk_categories_present.strip():
            validation_errors.append(f"Row {row_number} is missing risk_categories_present even though annotations exist.")

        if annotation_count == 0 and risk_categories_present.strip():
            validation_errors.append(f"Row {row_number} has risk_categories_present even though annotation_count is zero.")

        if risk_categories_present != expected_risk_categories_present:
            validation_errors.append(f"Row {row_number} risk_categories_present does not match the annotation categories.")

        for annotation_number, annotation in enumerate(annotation_list):
            if not isinstance(annotation, dict):
                validation_errors.append(f"Row {row_number} annotation {annotation_number} is not a dictionary.")
                continue

            required_annotation_keys = {
                "phrase_text",
                "start_index",
                "end_index",
                "risk_category",
                "risk_explanation",
            }
            missing_annotation_keys = [
                required_annotation_key
                for required_annotation_key in required_annotation_keys
                if required_annotation_key not in annotation
            ]

            if missing_annotation_keys:
                validation_errors.append(
                    f"Row {row_number} annotation {annotation_number} is missing keys: {missing_annotation_keys}"
                )
                continue

            phrase_text = str(annotation.get("phrase_text", ""))
            risk_category = str(annotation.get("risk_category", "")).strip()
            risk_explanation = str(annotation.get("risk_explanation", "")).strip()

            try:
                start_index = int(annotation.get("start_index"))
                end_index = int(annotation.get("end_index"))
            except (TypeError, ValueError):
                validation_errors.append(f"Row {row_number} annotation {annotation_number} has invalid span indexes.")
                continue

            if start_index < 0 or end_index <= start_index:
                validation_errors.append(f"Row {row_number} annotation {annotation_number} has an invalid span range.")
                continue

            if end_index > len(message_text):
                validation_errors.append(f"Row {row_number} annotation {annotation_number} ends outside message_text.")
                continue

            if not phrase_text.strip():
                validation_errors.append(f"Row {row_number} annotation {annotation_number} has empty phrase_text.")

            if not risk_category:
                validation_errors.append(f"Row {row_number} annotation {annotation_number} has empty risk_category.")

            if not risk_explanation:
                validation_errors.append(f"Row {row_number} annotation {annotation_number} has empty risk_explanation.")

            if message_text[start_index:end_index] != phrase_text:
                validation_errors.append(
                    f"Row {row_number} annotation {annotation_number} does not match the original message_text span."
                )

    return validation_errors


def validate_dataset_quality(
    full_dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    source_composition_summary: dict[str, Any],
) -> dict[str, Any]:
    # Run a simple quality gate before final savin just to be sure.
    validation_errors: list[str] = []

    dataset_map = {
        "full_dataset": full_dataset,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }

    for dataset_name, dataset in dataset_map.items():
        missing_columns = [
            required_column_name
            for required_column_name in required_dataset_columns
            if required_column_name not in dataset.columns
        ]

        if missing_columns:
            validation_errors.append(f"{dataset_name} is missing required columns: {missing_columns}")

    if train_dataset.empty:
        validation_errors.append("train_dataset is empty.")

    if test_dataset.empty:
        validation_errors.append("test_dataset is empty.")

    if full_dataset.empty:
        validation_errors.append("full_dataset is empty.")

    if source_composition_summary["real_sample_count"] == 0:
        validation_errors.append("full_dataset does not contain any non-synthetic rows.")

    if source_composition_summary["dataset_is_synthetic_only"]:
        validation_errors.append("full_dataset is composed entirely of synthetic rows.")

    if source_composition_summary["real_safe_sample_count"] == 0:
        validation_errors.append("No real safe messages were collected. The pipeline requires at least one real safe source.")

    if not full_dataset.empty:
        present_labels = set(full_dataset["label"].dropna().astype(int).tolist())
        if present_labels != {0, 1}:
            validation_errors.append("full_dataset must contain both label 0 and label 1.")

        valid_split_rows = full_dataset["split"].fillna("").isin({"train", "test"})
        if not valid_split_rows.all():
            validation_errors.append("full_dataset contains invalid or empty split values.")

        empty_cleaned_messages = full_dataset["cleaned_message_text"].fillna("").str.strip().eq("")
        if empty_cleaned_messages.any():
            validation_errors.append("full_dataset contains empty cleaned_message_text values.")

        validation_errors.extend(validate_annotation_columns(full_dataset))

    if not train_dataset.empty and not train_dataset["split"].eq("train").all():
        validation_errors.append("train_dataset split column must be 'train'.")

    if not test_dataset.empty and not test_dataset["split"].eq("test").all():
        validation_errors.append("test_dataset split column must be 'test'.")

    if validation_errors:
        raise RuntimeError("Dataset validation failed:\n- " + "\n- ".join(validation_errors))

    return {
        "required_columns_present": True,
        "full_dataset_contains_both_labels": True,
        "non_synthetic_rows_exist": True,
        "real_safe_rows_exist": True,
        "train_dataset_is_not_empty": True,
        "test_dataset_is_not_empty": True,
        "split_column_is_filled_correctly": True,
        "cleaned_messages_are_not_empty": True,
        "annotation_fields_are_valid": True,
    }


def validate_saved_output_files(output_file_paths: list[Path]) -> None:
    # To make sure the expected output files were actually written.
    validation_errors: list[str] = []

    for output_file_path in output_file_paths:
        if not output_file_path.exists():
            validation_errors.append(f"Missing output file: {output_file_path.name}")
            continue

        if output_file_path.stat().st_size == 0:
            validation_errors.append(f"Output file is empty: {output_file_path.name}")

    if validation_errors:
        raise RuntimeError("Saved file validation failed:\n- " + "\n- ".join(validation_errors))


def build_metadata_payload(
    raw_record_count: int,
    full_dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    source_summaries: list[dict[str, Any]],
    source_composition_summary: dict[str, Any],
    dataset_validation_summary: dict[str, Any],
) -> dict[str, Any]:
    # Build the metadata file that describes the dataset build.
    annotation_summary = build_annotation_summary(full_dataset)

    label_distribution = {
        label_name: int(count_value)
        for label_name, count_value in full_dataset["label_name"].value_counts().to_dict().items()
    }

    source_distribution = (
        full_dataset.groupby(["source_name", "data_origin_type", "is_synthetic"])
        .size()
        .reset_index(name="count")
    )

    split_distribution = {
        split_name: int(count_value)
        for split_name, count_value in full_dataset["split"].value_counts().to_dict().items()
    }

    sources_used = [
        source_summary["source_name"]
        for source_summary in source_summaries
        if source_summary["used_for_dataset"]
    ]

    sources_skipped = [
        {
            "source_name": source_summary["source_name"],
            "reason": source_summary["skip_reason"],
            "failure_categories": source_summary.get("failure_categories", []),
        }
        for source_summary in source_summaries
        if not source_summary["used_for_dataset"]
    ]

    warning_messages: list[str] = []

    if source_composition_summary["dataset_is_synthetic_only"]:
        warning_messages.append("The dataset depends only on synthetic rows because no real source contributed rows.")

    if source_composition_summary["label_coverage_depends_on_synthetic_rows"]:
        warning_messages.append("Final label coverage depends on synthetic rows.")

    return {
        "generated_at_universal_time": get_current_timestamp_in_universal_time(),
        "random_seed": random_seed,
        "train_split_ratio": train_split_ratio,
        "test_split_ratio": train_split_ratio if False else test_split_ratio,  # keeps naming explicit without changing behavior
        "total_raw_collected_samples": int(raw_record_count),
        "total_final_samples": int(len(full_dataset)),
        "real_sample_count": int((~full_dataset["is_synthetic"]).sum()),
        "synthetic_sample_count": int(full_dataset["is_synthetic"].sum()),
        "label_distribution": label_distribution,
        "split_distribution": split_distribution,
        "train_size": int(len(train_dataset)),
        "test_size": int(len(test_dataset)),
        "sources_used": sources_used,
        "sources_skipped": sources_skipped,
        "source_distribution": source_distribution.to_dict(orient="records"),
        "source_composition": source_composition_summary,
        "annotation_summary": annotation_summary,
        "required_columns": required_dataset_columns,
        "available_columns": list(full_dataset.columns),
        "recommended_vectorizer_configuration": build_recommended_vectorizer_configuration(),
        "dataset_validation": dataset_validation_summary,
        "warnings": warning_messages,
    }


def print_summary(
    raw_record_count: int,
    full_dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    source_summaries: list[dict[str, Any]],
    source_composition_summary: dict[str, Any],
) -> None:
   # Print a small readable summary at the end of the run.
    total_safe_samples = int((full_dataset["label"] == 0).sum())
    total_scam_samples = int((full_dataset["label"] == 1).sum())
    total_synthetic_samples = int(full_dataset["is_synthetic"].sum())

    sources_used = [
        source_summary["source_name"]
        for source_summary in source_summaries
        if source_summary["used_for_dataset"]
    ]

    sources_skipped = [
        source_summary["source_name"]
        for source_summary in source_summaries
        if not source_summary["used_for_dataset"]
    ]

    print(f"total collected samples: {raw_record_count}")
    print(f"total safe samples: {total_safe_samples}")
    print(f"total scam samples: {total_scam_samples}")
    print(f"real sample count: {source_composition_summary['real_sample_count']}")
    print(f"synthetic sample count: {total_synthetic_samples}")
    print(f"non-synthetic sources succeeded: {source_composition_summary['has_non_synthetic_source_success']}")
    print(f"train size: {len(train_dataset)}")
    print(f"test size: {len(test_dataset)}")
    print(f"sources used: {sources_used}")
    print(f"sources skipped: {sources_skipped}")

    if source_composition_summary["dataset_is_synthetic_only"]:
        print("warning: dataset depends only on synthetic rows.")
    elif source_composition_summary["label_coverage_depends_on_synthetic_rows"]:
        print("warning: final label coverage depends on synthetic rows.")



# main pipeline

def main() -> None:
    # Run the full dataset pipeline.
    ensure_output_directories_exist()

    request_session = create_default_request_session()
    validation_log: list[dict[str, Any]] = []
    skipped_rows_log: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    try:
        raw_records: list[dict[str, Any]] = []

        # Collect records from all configured real sources.
        for source_definition in source_manifest:
            source_records, source_skipped_rows, source_summary = collect_records_from_source(
                request_session=request_session,
                source_definition=source_definition,
                validation_log=validation_log,
            )
            raw_records.extend(source_records)
            skipped_rows_log.extend(source_skipped_rows)
            source_summaries.append(source_summary)

        # Add synthetic scam rows as supplemental data.
        synthetic_records = generate_synthetic_scam_messages(
            total_message_count=synthetic_scam_message_count,
            random_seed=random_seed,
        )
        raw_records.extend(synthetic_records)
        source_summaries.append(
            {
                "source_name": "synthetic_rule_based_scam_messages",
                "source_type": "synthetic_generation",
                "records_collected": len(synthetic_records),
                "used_for_dataset": len(synthetic_records) > 0,
                "is_synthetic_source": True,
                "failure_categories": [],
                "skip_reason": "" if synthetic_records else "No synthetic records were generated.",
            }
        )

        raw_dataset = pd.DataFrame(raw_records)

        if raw_dataset.empty:
            write_processing_logs(validation_log, skipped_rows_log)
            raise RuntimeError("No records were collected at all. Check source settings and internet access.")

        raw_dataset_output_path = raw_data_directory / raw_dataset_file_name
        save_dataframe_as_csv(raw_dataset, raw_dataset_output_path)

        prepared_dataset, preprocessing_skipped_rows = prepare_dataset_for_machine_learning(
            raw_dataset=raw_dataset,
            minimum_message_length=minimum_message_length,
            remove_stopwords=use_stopword_removal,
        )
        skipped_rows_log.extend(preprocessing_skipped_rows)

        if prepared_dataset.empty:
            write_processing_logs(validation_log, skipped_rows_log)
            raise RuntimeError("All rows were removed during preprocessing. Check source quality and rules.")

        train_dataset, test_dataset = split_dataset(prepared_dataset)

        full_dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)
        full_dataset = order_final_dataset_columns(full_dataset)
        train_dataset = order_final_dataset_columns(train_dataset)
        test_dataset = order_final_dataset_columns(test_dataset)

        source_composition_summary = build_source_composition_summary(full_dataset)
        validate_real_source_collection(source_summaries, source_composition_summary)

        dataset_validation_summary = validate_dataset_quality(
            full_dataset=full_dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            source_composition_summary=source_composition_summary,
        )

        full_dataset_output_path = processed_data_directory / full_dataset_file_name
        train_dataset_output_path = processed_data_directory / train_dataset_file_name
        test_dataset_output_path = processed_data_directory / test_dataset_file_name
        metadata_output_path = metadata_directory / metadata_file_name

        save_dataframe_as_csv(full_dataset, full_dataset_output_path)
        save_dataframe_as_csv(train_dataset, train_dataset_output_path)
        save_dataframe_as_csv(test_dataset, test_dataset_output_path)

        metadata_payload = build_metadata_payload(
            raw_record_count=len(raw_dataset),
            full_dataset=full_dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            source_summaries=source_summaries,
            source_composition_summary=source_composition_summary,
            dataset_validation_summary=dataset_validation_summary,
        )
        save_json_to_file(metadata_payload, metadata_output_path)

        write_processing_logs(validation_log, skipped_rows_log)

        validate_saved_output_files(
            [
                raw_dataset_output_path,
                full_dataset_output_path,
                train_dataset_output_path,
                test_dataset_output_path,
                metadata_output_path,
                logs_directory / source_validation_log_file_name,
                logs_directory / skipped_rows_log_file_name,
            ]
        )

        print_summary(
            raw_record_count=len(raw_dataset),
            full_dataset=full_dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            source_summaries=source_summaries,
            source_composition_summary=source_composition_summary,
        )

    finally:
        request_session.close()


if __name__ == "__main__":
    main()