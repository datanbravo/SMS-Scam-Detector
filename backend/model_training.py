from __future__ import annotations

import json
import pickle
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer

from model_config import (
    decision_tree_max_depth,
    decision_tree_min_samples_leaf,
    enabled_model_names,
    evaluation_metric_names,
    get_best_model_output_path,
    get_best_embedder_output_path,
    embedding_model_name,
    get_training_results_output_path,
    logistic_regression_max_iterations,
    logistic_regression_regularization_strength,
    primary_metric_name,
    svm_regularization_strength,
    svm_kernel,
    svm_use_probability,
    test_dataset_path,
    train_dataset_path,
    training_label_column,
    training_random_seed,
    training_text_column,
    ensure_model_output_directories_exist,
)


#Basic helpers --------------------------------------


def load_dataset(dataset_path: str | Any) -> pd.DataFrame:
    #Load a CSV dataset from the provided path.
    # and return it as a pandas DataFrame.
    return pd.read_csv(dataset_path)


def validate_dataset_columns(dataset: pd.DataFrame, dataset_name: str) -> None:
    #Verify the dataset contains the required text
    #and label columns and its not empty
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


def validate_training_inputs(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    # Validate both train and test datasets before starting feature extraction and training.
    validate_dataset_columns(train_data, "train_data")
    validate_dataset_columns(test_data, "test_data")

    if train_dataset[training_text_column].fillna("").str.strip().eq("").all():
        raise ValueError("train_data has no usable text in the training text column.")

    if test_dataset[training_text_column].fillna("").str.strip().eq("").all():
        raise ValueError("test_data has no usable text in the training text column.")


#Text embedding --------------------------------------

def prepare_feature_matrices(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[SentenceTransformer, Any, Any, pd.Series, pd.Series]:
    #Fit the embedder on training text, then encode train and test text.
    embedder = SentenceTransformer(embedding_model_name)

    train_text_list = train_data[training_text_column].fillna("").astype(str).tolist()
    test_text_list = test_data[training_text_column].fillna("").astype(str).tolist()

    x_train = embedder.encode(train_text_list)
    x_test = embedder.encode(test_text_list)

    y_train = train_data[training_label_column].astype(int)
    y_test = test_data[training_label_column].astype(int)

    return embedder, x_train, x_test, y_train, y_test


#Model builders --------------------------------------


def build_logistic_regression_model() -> LogisticRegression:
    #Logistic Regression is a strong baseline for text classification.
    #Build and return a Logistic Regression model
    #configured for text classification tasks. 
    return LogisticRegression(
        C=logistic_regression_regularization_strength,
        max_iter=logistic_regression_max_iterations,
        random_state=training_random_seed,
    )


def build_decision_tree_model() -> DecisionTreeClassifier:
    # Decision Tree is useful as a class-aligned comparison model.
    #Build and return a Decision Tree classifier
    #for comparison against other models.
    return DecisionTreeClassifier(
        max_depth=decision_tree_max_depth,
        min_samples_leaf=decision_tree_min_samples_leaf,
        random_state=training_random_seed,
    )


def build_svm_model() -> SVC:
    # Linear SVM can work well on dense sentenc embeddings.
    return SVC(
        C=svm_regularization_strength,
        kernel=svm_kernel,
        probability=svm_use_probability,
        random_state=training_random_seed,
    )


def build_model_dictionary() -> dict[str, Any]:
    #Build the set of enabled models we want to compare.
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


#Evaluation --------------------------------------

def calculate_classification_metrics(
    true_labels: pd.Series,
    preds: Any,
) -> dict[str, float]:
    #Compute the main classification metrics for one model.
    scores = {
        "accuracy": float(accuracy_score(true_labels, preds)),
        "precision": float(precision_score(true_labels, preds, zero_division=0)),
        "recall": float(recall_score(true_labels, preds, zero_division=0)),
        "f1": float(f1_score(true_labels, preds, zero_division=0)),
    }

    #Keep only the metrics listed in config, but still compute safely above.
    return {
        metric_name: scores[metric_name]
        for metric_name in evaluation_metric_names
        if metric_name in scores
    }


def get_scam_probabilities(model: Any, x_values: Any) -> list[float]:
    #Get scam probabilities when the model supports it.
    if hasattr(model, "predict_proba"):
        probability_rows = model.predict_proba(x_values)
        return [float(row[1]) for row in probability_rows]

    #This is just a backup. The main models should have predict_proba.
    preds = model.predict(x_values)
    return [float(value) for value in preds]


def calculate_probability_report(true_labels: pd.Series, scam_probabilities: list[float]) -> dict[str, float]:
    #Check a few strict cutoffs so we know how careful the model is.
    cutoffs = [0.50, 0.60, 0.70, 0.80]
    report: dict[str, float] = {}

    for cutoff in cutoffs:
        cutoff_preds = [1 if value >= cutoff else 0 for value in scam_probabilities]
        key = str(cutoff).replace(".", "_")

        report[f"precision_at_{key}"] = float(precision_score(true_labels, cutoff_preds, zero_division=0))
        report[f"recall_at_{key}"] = float(recall_score(true_labels, cutoff_preds, zero_division=0))
        report[f"f1_at_{key}"] = float(f1_score(true_labels, cutoff_preds, zero_division=0))

    return report

def train_and_evaluate_one_model(
    model_name: str,
    model: Any,
    x_train: Any,
    y_train: pd.Series,
    x_test: Any,
    y_test: pd.Series,
) -> dict[str, Any]:
    #Train a single model and check how it did.
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    scam_probs = get_scam_probabilities(model, x_test)

    print(f"\n{model_name} classification report")
    print("-" * 50)

    report = classification_report(
        y_test,
        preds,
        target_names=["safe", "scam"]
    )   

    print(report)
    
    scores = calculate_classification_metrics(
        true_labels=y_test,
        preds=preds,
    )
    scores.update(calculate_probability_report(y_test, scam_probs))

    return {
        "model_name": model_name,
        "metrics": scores,
        "model_object": model,
    }


def choose_best_model_result(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    #Pick the best model using the primary metric from config.
    if not model_results:
        raise ValueError("No model results were produced.")

    best_model_result = max(
        model_results,
        key=lambda model_result: (
            model_result["metrics"].get("f1_at_0_70", float("-inf")),
            model_result["metrics"].get("precision_at_0_70", float("-inf")),
            model_result["metrics"].get(primary_metric_name, float("-inf")),
        ),
    )

    return best_model_result


#Saving outputs --------------------------------------


def save_pickle_object(object_value: Any, output_path: Any) -> None:
    #Save a Python object with pickle.
    with open(output_path, "wb") as output_file:
        pickle.dump(object_value, output_file)


def build_training_report(
    model_results: list[dict[str, Any]],
    best_model_result: dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    # Build a small JSON report summarizing the training run.
    # containing dataset, vectorizer, and model results.
    return {
        "training_text_column": training_text_column,
        "training_label_column": training_label_column,
        "train_row_count": int(len(train_data)),
        "test_row_count": int(len(test_data)),
        "enabled_models": enabled_model_names,
        "primary_metric_name": primary_metric_name,
        "embedding_model_name": embedding_model_name,
        "model_results": [
            {
                "model_name": model_result["model_name"],
                "metrics": model_result["metrics"],
            }
            for model_result in model_results
        ],
        "best_model": {
            "model_name": best_model_result["model_name"],
            "metrics": best_model_result["metrics"],
        },
    }


def save_training_report(training_report: dict[str, Any]) -> None:
    #Save the training report as JSON.
    output_path = get_training_results_output_path()
    output_path.write_text(
        json.dumps(training_report, indent=2),
        encoding="utf-8",
    )


#Printing --------------------------------------

def print_model_results(model_results: list[dict[str, Any]], best_model_result: dict[str, Any]) -> None:
    #Print model metrics to the terminal.
    print("\nmodel evaluation results")
    print("-" * 50)

    for model_result in model_results:
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


#Main training flow --------------------------------------

def main() -> None:
    #Run the full training flow, basically:.
    #dataset loading, validation, vectorization,.
    # model training, evaluation, and output saving.
    ensure_model_output_directories_exist()

    train_data = load_dataset(train_dataset_path)
    test_data = load_dataset(test_dataset_path)

    validate_training_inputs(train_data, test_data)

    embedder, x_train, x_test, y_train, y_test = prepare_feature_matrices(
        train_data=train_data,
        test_data=test_data,
    )

    models = build_model_dictionary()
    model_results: list[dict[str, Any]] = []

    for model_name, model in models.items():
        model_result = train_and_evaluate_one_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        model_results.append(model_result)

    best_model_result = choose_best_model_result(model_results)

    best_model_output_path = get_best_model_output_path()
    best_vectorizer_output_path = get_best_vectorizer_output_path()

    save_pickle_object(best_model_result["model_object"], best_model_output_path)
    save_pickle_object(embedder, best_vectorizer_output_path)

    training_report = build_training_report(
        model_results=model_results,
        best_model_result=best_model_result,
        train_data=train_data,
        test_data=test_data,
        embedder=embedder,
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
