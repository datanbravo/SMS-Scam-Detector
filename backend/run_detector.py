from __future__ import annotations

import json
import pickle
from typing import Any

from model_config import (
    get_best_model_output_path,
    get_best_vectorizer_output_path,
    training_text_column,
)
from preprocessing import (
    clean_message_text,
    convert_annotation_list_to_row_fields,
    create_unigram_bigram_ready_text,
    extract_suspicious_phrase_annotations,
)


# ------------------------------------------------------------
# loading helpers
# ------------------------------------------------------------

def load_pickle_object(file_path: Any) -> Any:
    # Load a saved pickle file.
    with open(file_path, "rb") as input_file:
        return pickle.load(input_file)


def load_trained_model() -> Any:
    # Load the best trained model.
    model_path = get_best_model_output_path()
    return load_pickle_object(model_path)


def load_trained_vectorizer() -> Any:
    # Load the saved text vectorizer.
    vectorizer_path = get_best_vectorizer_output_path()
    return load_pickle_object(vectorizer_path)


# ------------------------------------------------------------
# text preparation
# ------------------------------------------------------------

def build_prediction_text_fields(message_text: str) -> dict[str, str]:
    # Build the same text fields used during training.
    cleaned_message_text = clean_message_text(message_text)
    unigram_bigram_ready_text = create_unigram_bigram_ready_text(cleaned_message_text)

    return {
        "message_text": message_text,
        "cleaned_message_text": cleaned_message_text,
        "unigram_bigram_ready_text": unigram_bigram_ready_text,
    }


def get_training_text_value(text_field_dictionary: dict[str, str]) -> str:
    # Return the correct training text field based on model_config.
    if training_text_column not in text_field_dictionary:
        raise ValueError(
            f"Training text column '{training_text_column}' is not available in the prepared text fields."
        )

    return text_field_dictionary[training_text_column]


# ------------------------------------------------------------
# prediction helpers
# ------------------------------------------------------------

def predict_label(
    message_text: str,
    model: Any,
    vectorizer: Any,
) -> dict[str, Any]:
    # Predict the binary label for one message.
    text_field_dictionary = build_prediction_text_fields(message_text)
    training_text_value = get_training_text_value(text_field_dictionary)

    transformed_text = vectorizer.transform([training_text_value])
    predicted_label = int(model.predict(transformed_text)[0])
    predicted_label_name = "scam" if predicted_label == 1 else "safe"

    return {
        "label": predicted_label,
        "label_name": predicted_label_name,
        "text_fields": text_field_dictionary,
    }


def build_short_explanation(
    predicted_label_name: str,
    annotation_list: list[dict[str, Any]],
) -> str:
    # Build a short explanation for the final result.
    if predicted_label_name == "safe":
        if annotation_list:
            return (
                "The model predicted this message as safe, "
                "but some phrases may still look suspicious on their own."
            )

        return "The model predicted this message as safe."

    if annotation_list:
        return (
            "The model predicted this message as scam, "
            "and suspicious phrases were found that support the prediction."
        )

    return (
        "The model predicted this message as scam, "
        "but no suspicious phrases were matched by the current rule-based explanation system."
    )


def build_prediction_result(
    message_text: str,
    predicted_label_name: str,
    annotation_list: list[dict[str, Any]],
) -> dict[str, Any]:
    # Build the final user-facing result dictionary.
    annotation_row_fields = convert_annotation_list_to_row_fields(annotation_list)

    return {
        "message_text": message_text,
        "classification": predicted_label_name,
        "suspicious_phrases": annotation_list,
        "annotation_count": annotation_row_fields["annotation_count"],
        "risk_categories_present": annotation_row_fields["risk_categories_present"],
        "short_explanation": build_short_explanation(
            predicted_label_name=predicted_label_name,
            annotation_list=annotation_list,
        ),
    }


# ------------------------------------------------------------
# main prediction function
# ------------------------------------------------------------

def analyze_message(
    message_text: str,
    model: Any,
    vectorizer: Any,
) -> dict[str, Any]:
    # Run the full prediction flow for one message.
    message_text = str(message_text).strip()

    if not message_text:
        return {
            "message_text": "",
            "classification": "unknown",
            "suspicious_phrases": [],
            "annotation_count": 0,
            "risk_categories_present": "",
            "short_explanation": "No message text was provided.",
        }

    prediction_dictionary = predict_label(
        message_text=message_text,
        model=model,
        vectorizer=vectorizer,
    )

    annotation_list = extract_suspicious_phrase_annotations(message_text)

    return build_prediction_result(
        message_text=message_text,
        predicted_label_name=prediction_dictionary["label_name"],
        annotation_list=annotation_list,
    )


# ------------------------------------------------------------
# cli runner
# ------------------------------------------------------------

def main() -> None:
    # Run the detector in simple command-line mode.
    model = load_trained_model()
    vectorizer = load_trained_vectorizer()

    print("sms scam detector")
    print("type 'exit' to quit")
    print("-" * 50)

    while True:
        message_text = input("\nenter sms message: ").strip()

        if message_text.lower() == "exit":
            break

        result = analyze_message(
            message_text=message_text,
            model=model,
            vectorizer=vectorizer,
        )

        print("\nresult")
        print("-" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()