from __future__ import annotations

import json
import pickle
from typing import Any

from config import (
    high_risk_categories,
    scam_probability_cutoff,
    strong_rule_count_cutoff,
    suspicious_probability_cutoff,
)
from model_config import (
    get_best_embedder_output_path,
    get_best_model_output_path,
    training_text_column,
)
from preprocessing import (
    clean_message_text,
    convert_annotation_list_to_row_fields,
    create_unigram_bigram_ready_text,
    extract_suspicious_phrase_annotations,
)


#Loading helpers --------------------------------------


def load_pickle_object(file_path: Any) -> Any:
    #Load a saved pickle file.
    with open(file_path, "rb") as input_file:
        return pickle.load(input_file)


def load_trained_model() -> Any:
    #Load the best trained model.
    model_path = get_best_model_output_path()
    return load_pickle_object(model_path)


def load_trained_embedder() -> Any:
    #Load the saved sentence embedding model.
    embedder_path = get_best_embedder_output_path()
    return load_pickle_object(embedder_path)


#Text preparation --------------------------------------


def build_prediction_text_fields(message_text: str) -> dict[str, str]:
    #Build the same text fields the model saw during training.
    cleaned_message_text = clean_message_text(message_text)
    unigram_bigram_ready_text = create_unigram_bigram_ready_text(cleaned_message_text)

    return {
        "message_text": message_text,
        "cleaned_message_text": cleaned_message_text,
        "unigram_bigram_ready_text": unigram_bigram_ready_text,
    }


def get_training_text_value(text_fields: dict[str, str]) -> str:
    #Grab the exact text column the model was trained on.
    if training_text_column not in text_fields:
        raise ValueError(
            f"Training text column '{training_text_column}' is not available in the prepared text fields."
        )

    return text_fields[training_text_column]

#Model confidence --------------------------------------

def get_scam_probability(model: Any, embedded_text: Any) -> float:
    #Use real probabilities when the model has them.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(embedded_text)[0]
        return float(probabilities[1])

    #This backup is not as nice, but it stops the app from crashing.
    prediction = int(model.predict(embedded_text)[0])
    return 1.0 if prediction == 1 else 0.0


def predict_with_confidence(message_text: str, model: Any, embedder: Any) -> dict[str, Any]:
    #Predict one message and keep the scam probability too.
    text_fields = build_prediction_text_fields(message_text)
    training_text_value = get_training_text_value(text_fields)

    embedded_text = embedder.encode([training_text_value])
    scam_probability = get_scam_probability(model, embedded_text)

    return {
        "scam_probability": scam_probability,
        "text_fields": text_fields,
    }


#Rule scoring --------------------------------------

def count_high_risk_rules(annotations: list[dict[str, Any]]) -> int:
    #Count the stronger rule matches, not just every little thing.
    count = 0

    for annotation in annotations:
        risk_category = str(annotation.get("risk_category", ""))

        if risk_category in high_risk_categories:
            count += 1

    return count


def choose_final_classification(
    scam_probability: float,
    annotations: list[dict[str, Any]],
) -> str:
    #Mix the model and the rules without being way too dramatic.
    annotation_count = len(annotations)
    high_risk_rule_count = count_high_risk_rules(annotations)

    if scam_probability >= scam_probability_cutoff:
        return "scam"

    if scam_probability >= suspicious_probability_cutoff:
        return "suspicious"

    if high_risk_rule_count >= strong_rule_count_cutoff:
        return "suspicious"

    if annotation_count >= strong_rule_count_cutoff + 1:
        return "suspicious"

    return "safe"


#Explanation text --------------------------------------

def make_phrase_summary(annotations: list[dict[str, Any]]) -> str:
    #Make a tiny phrase list for the user explanation.
    if not annotations:
        return ""

    phrase_list = []

    for annotation in annotations[:3]:
        phrase_text = str(annotation.get("phrase_text", "")).strip()

        if phrase_text:
            phrase_list.append(f"'{phrase_text}'")

    return ", ".join(phrase_list)


def build_short_explanation(
    final_classification: str,
    scam_probability: float,
    annotations: list[dict[str, Any]],
) -> str:
    #Explain the result like the person using it needs to know.
    percent_text = f"{round(scam_probability * 100, 1)}%"
    phrase_summary = make_phrase_summary(annotations)

    if final_classification == "scam":
        if phrase_summary:
            return (
                f"This was marked as scam because the message has a high scam confidence of {percent_text} "
                f"and includes risky wording like {phrase_summary}."
            )

        return f"This was marked as scam because the scam confidence is high at {percent_text}."

    if final_classification == "suspicious":
        if phrase_summary:
            return (
                f"This was marked as suspicious because it has some warning signs, like {phrase_summary}, "
                f"but the scam confidence is only {percent_text}, so it is not called a confirmed scam."
            )

        return (
            f"This was marked as suspicious because the scam confidence is {percent_text}, "
            "which is high enough to be careful but not high enough to call it a scam."
        )

    if phrase_summary:
        return (
            f"This was marked as safe because the scam confidence is low at {percent_text}. "
            f"It did notice {phrase_summary}, but by itself that was not enough to flag the full message."
        )

    return f"This was marked as safe because the scam confidence is low at {percent_text} and no risky phrases stood out."


#Result builder --------------------------------------

def build_prediction_result(
    message_text: str,
    prediction: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    #Build the final user-facing dictionary.
    annotation_fields = convert_annotation_list_to_row_fields(annotations)

    scam_probability = float(prediction["scam_probability"])
    final_classification = choose_final_classification(
        scam_probability=scam_probability,
        annotations=annotations,
    )

    return {
        "message_text": message_text,
        "classification": final_classification,
        "scam_confidence": round(scam_probability, 4),
        "suspicious_phrases": annotations,
        "annotation_count": annotation_fields["annotation_count"],
        "risk_categories_present": annotation_fields["risk_categories_present"],
        "short_explanation": build_short_explanation(
            final_classification=final_classification,
            scam_probability=scam_probability,
            annotations=annotations,
        ),
    }


#Main prediction function --------------------------------------


def analyze_message(message_text: str, model: Any, embedder: Any) -> dict[str, Any]:
    #Run the whole detector for one message.
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

    prediction = predict_label(
        message_text=message_text,
        model=model,
        embedder=embedder,
    )

    annotations = extract_suspicious_phrase_annotations(message_text)

    return build_prediction_result(
        message_text=message_text,
        predicted=prediction["label_name"],
        annotations=annotations,
    )


#Cli runner --------------------------------------


def main() -> None:
    #Run the detector in simple command-line mode.
    model = load_trained_model()
    embedder = load_trained_embedder()

    print("SMS Scam Setector")
    print("Type 'exit' or 'q' to quit")
    print("-" * 50)

    while True:
        message_text = input("\nEnter sms message: ").strip()

        if message_text.lower() in {"exit", "q"}:
            break

        result = analyze_message(
            message_text=message_text,
            model=model,
            embedder=embedder,
        )

        print("\nresult")
        print("-" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
