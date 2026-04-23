from __future__ import annotations

import json
import pickle
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_config import (
    decision_tree_max_depth,
    decision_tree_min_samples_leaf,
    enabled_model_names,
    evaluation_metric_names,
    get_best_model_output_path,
    get_best_vectorizer_output_path,
    get_training_results_output_path,
    logistic_regression_max_iterations,
    logistic_regression_regularization_strength,
    primary_metric_name,
    svm_regularization_strength,
    svm_kernel,
    test_dataset_path,
    tfidf_lowercase,
    tfidf_max_features,
    tfidf_min_document_frequency,
    tfidf_ngram_range,
    tfidf_token_pattern,
    train_dataset_path,
    training_label_column,
    training_random_seed,
    training_text_column,
    ensure_model_output_directories_exist,
)


# ------------------------------------------------------------
# basic helpers
# ------------------------------------------------------------

def load_dataset(dataset_path: str | Any) -> pd.DataFrame:
    # Load one csv dataset.
    return pd.read_csv(dataset_path)


def validate_dataset_columns(dataset: pd.DataFrame, dataset_name: str) -> None:
    # Make sure the required training columns exist.
    required_columns = [training_text_column, training_label_column]

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )

    if dataset.empty:
        raise ValueError(f"{dataset_name} is empty.")


def validate_training_inputs(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame) -> None:
    # Validate both train and test datasets before training starts.
    validate_dataset_columns(train_dataset, "train_dataset")
    validate_dataset_columns(test_dataset, "test_dataset")

    if train_dataset[training_text_column].fillna("").str.strip().eq("").all():
        raise ValueError("train_dataset has no usable text in the training text column.")

    if test_dataset[training_text_column].fillna("").str.strip().eq("").all():
        raise ValueError("test_dataset has no usable text in the training text column.")


# ------------------------------------------------------------
# text vectorization
# ------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    # Create the TF-IDF vectorizer using the project settings.
    return TfidfVectorizer(
        max_features=tfidf_max_features,
        min_df=tfidf_min_document_frequency,
        ngram_range=tfidf_ngram_range,
        lowercase=tfidf_lowercase,
        token_pattern=tfidf_token_pattern,
    )


def prepare_feature_matrices(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
) -> tuple[TfidfVectorizer, Any, Any, pd.Series, pd.Series]:
    # Fit the vectorizer on training text, then transform train and test text.
    vectorizer = build_vectorizer()

    train_text_list = train_dataset[training_text_column].fillna("").astype(str).tolist()
    test_text_list = test_dataset[training_text_column].fillna("").astype(str).tolist()

    x_train = vectorizer.fit_transform(train_text_list)
    x_test = vectorizer.transform(test_text_list)

    y_train = train_dataset[training_label_column].astype(int)
    y_test = test_dataset[training_label_column].astype(int)

    return vectorizer, x_train, x_test, y_train, y_test


# ------------------------------------------------------------
# model builders
# ------------------------------------------------------------

def build_logistic_regression_model() -> LogisticRegression:
    # Logistic Regression is a strong baseline for text classification.
    return LogisticRegression(
        C=logistic_regression_regularization_strength,
        max_iter=logistic_regression_max_iterations,
        random_state=training_random_seed,
    )


def build_decision_tree_model() -> DecisionTreeClassifier:
    # Decision Tree is useful as a class-aligned comparison model.
    return DecisionTreeClassifier(
        max_depth=decision_tree_max_depth,
        min_samples_leaf=decision_tree_min_samples_leaf,
        random_state=training_random_seed,
    )


def build_svm_model() -> SVC:
    # Linear SVM is often strong for sparse text features like TF-IDF.
    return SVC(
        C=svm_regularization_strength,
        kernel=svm_kernel,
        random_state=training_random_seed,
    )


def build_model_dictionary() -> dict[str, Any]:
    # Build the set of enabled models we want to compare.
    available_model_builders = {
        "logistic_regression": build_logistic_regression_model,
        "decision_tree": build_decision_tree_model,
        "svm": build_svm_model,
    }

    model_dictionary: dict[str, Any] = {}

    for model_name in enabled_model_names:
        if model_name not in available_model_builders:
            raise ValueError(f"Unsupported model name in config: {model_name}")

        model_dictionary[model_name] = available_model_builders[model_name]()

    return model_dictionary


# ------------------------------------------------------------
# evaluation
# ------------------------------------------------------------

def calculate_classification_metrics(
    true_labels: pd.Series,
    predicted_labels: Any,
) -> dict[str, float]:
    # Compute the main classification metrics for one model.
    metric_dictionary = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, predicted_labels, zero_division=0)),
    }

    # Keep only the metrics listed in config, but still compute safely above.
    return {
        metric_name: metric_dictionary[metric_name]
        for metric_name in evaluation_metric_names
        if metric_name in metric_dictionary
    }


def train_and_evaluate_one_model(
    model_name: str,
    model: Any,
    x_train: Any,
    y_train: pd.Series,
    x_test: Any,
    y_test: pd.Series,
) -> dict[str, Any]:
    # Fit one model and evaluate it on the test set.
    model.fit(x_train, y_train)
    predicted_labels = model.predict(x_test)

    metric_dictionary = calculate_classification_metrics(
        true_labels=y_test,
        predicted_labels=predicted_labels,
    )

    return {
        "model_name": model_name,
        "metrics": metric_dictionary,
        "model_object": model,
    }


def choose_best_model_result(model_result_list: list[dict[str, Any]]) -> dict[str, Any]:
    # Pick the best model using the primary metric from config.
    if not model_result_list:
        raise ValueError("No model results were produced.")

    best_model_result = max(
        model_result_list,
        key=lambda model_result: (
            model_result["metrics"].get(primary_metric_name, float("-inf")),
            model_result["metrics"].get("accuracy", float("-inf")),
        ),
    )

    return best_model_result


# ------------------------------------------------------------
# saving outputs
# ------------------------------------------------------------

def save_pickle_object(object_value: Any, output_path: Any) -> None:
    # Save a Python object with pickle.
    with open(output_path, "wb") as output_file:
        pickle.dump(object_value, output_file)


def build_training_report(
    model_result_list: list[dict[str, Any]],
    best_model_result: dict[str, Any],
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> dict[str, Any]:
    # Build a small JSON report summarizing the training run.
    return {
        "training_text_column": training_text_column,
        "training_label_column": training_label_column,
        "train_row_count": int(len(train_dataset)),
        "test_row_count": int(len(test_dataset)),
        "enabled_models": enabled_model_names,
        "primary_metric_name": primary_metric_name,
        "vectorizer_settings": {
            "max_features": tfidf_max_features,
            "min_document_frequency": tfidf_min_document_frequency,
            "ngram_range": list(tfidf_ngram_range),
            "lowercase": tfidf_lowercase,
            "token_pattern": tfidf_token_pattern,
        },
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "model_results": [
            {
                "model_name": model_result["model_name"],
                "metrics": model_result["metrics"],
            }
            for model_result in model_result_list
        ],
        "best_model": {
            "model_name": best_model_result["model_name"],
            "metrics": best_model_result["metrics"],
        },
    }


def save_training_report(training_report: dict[str, Any]) -> None:
    # Save the training report as JSON.
    output_path = get_training_results_output_path()
    output_path.write_text(
        json.dumps(training_report, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------
# printing
# ------------------------------------------------------------

def print_model_results(model_result_list: list[dict[str, Any]], best_model_result: dict[str, Any]) -> None:
    # Print readable model metrics to the terminal.
    print("\nmodel evaluation results")
    print("-" * 50)

    for model_result in model_result_list:
        model_name = model_result["model_name"]
        metrics = model_result["metrics"]

        print(f"\n{model_name}")
        for metric_name in evaluation_metric_names:
            if metric_name in metrics:
                print(f"  {metric_name}: {metrics[metric_name]:.4f}")

    print("\nbest model")
    print("-" * 50)
    print(f"name: {best_model_result['model_name']}")
    for metric_name in evaluation_metric_names:
        if metric_name in best_model_result["metrics"]:
            print(f"{metric_name}: {best_model_result['metrics'][metric_name]:.4f}")


# ------------------------------------------------------------
# main training flow
# ------------------------------------------------------------

def main() -> None:
    # Run the full model training pipeline.
    ensure_model_output_directories_exist()

    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    validate_training_inputs(train_dataset, test_dataset)

    vectorizer, x_train, x_test, y_train, y_test = prepare_feature_matrices(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    model_dictionary = build_model_dictionary()
    model_result_list: list[dict[str, Any]] = []

    for model_name, model in model_dictionary.items():
        model_result = train_and_evaluate_one_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        model_result_list.append(model_result)

    best_model_result = choose_best_model_result(model_result_list)

    best_model_output_path = get_best_model_output_path()
    best_vectorizer_output_path = get_best_vectorizer_output_path()

    save_pickle_object(best_model_result["model_object"], best_model_output_path)
    save_pickle_object(vectorizer, best_vectorizer_output_path)

    training_report = build_training_report(
        model_result_list=model_result_list,
        best_model_result=best_model_result,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        vectorizer=vectorizer,
    )
    save_training_report(training_report)

    print_model_results(model_result_list, best_model_result)

    print("\nsaved files")
    print("-" * 50)
    print(f"best model: {best_model_output_path}")
    print(f"best vectorizer: {best_vectorizer_output_path}")
    print(f"training report: {get_training_results_output_path()}")


if __name__ == "__main__":
    main()