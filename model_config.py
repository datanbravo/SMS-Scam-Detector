from __future__ import annotations

from pathlib import Path

from config import project_root_directory, random_seed


# ------------------------------------------------------------
# paths
# ------------------------------------------------------------
# These paths are only for the model-training part.

saved_models_directory = project_root_directory / "saved_models"
training_reports_directory = project_root_directory / "training_reports"

train_dataset_path = project_root_directory / "data" / "processed" / "train_dataset.csv"
test_dataset_path = project_root_directory / "data" / "processed" / "test_dataset.csv"

best_model_file_name = "best_sms_scam_model.pkl"
best_vectorizer_file_name = "best_sms_scam_vectorizer.pkl"
training_results_file_name = "model_results.json"


# ------------------------------------------------------------
# training data settings
# ------------------------------------------------------------
# This is the text column the model will learn from.
# For this project, unigram + bigram text is the best choice.

training_text_column = "unigram_bigram_ready_text"
training_label_column = "label"


# ------------------------------------------------------------
# tf-idf settings
# ------------------------------------------------------------
# These settings define how text becomes numeric features.

tfidf_max_features = 10000
tfidf_min_document_frequency = 1
tfidf_ngram_range = (1, 2)
tfidf_lowercase = False
tfidf_token_pattern = r"(?u)\b\w+\b"


# ------------------------------------------------------------
# model settings
# ------------------------------------------------------------
# These are the models we want to compare.
# Keep this list simple and aligned with class topics.

enabled_model_names = [
    "logistic_regression",
    "decision_tree",
    "svm",
]


# ------------------------------------------------------------
# logistic regression settings
# ------------------------------------------------------------

logistic_regression_max_iterations = 2000
logistic_regression_regularization_strength = 1.0


# ------------------------------------------------------------
# decision tree settings
# ------------------------------------------------------------

decision_tree_max_depth = None
decision_tree_min_samples_leaf = 1


# ------------------------------------------------------------
# svm settings
# ------------------------------------------------------------

svm_regularization_strength = 1.0
svm_kernel = "linear"


# ------------------------------------------------------------
# evaluation settings
# ------------------------------------------------------------
# F1-score is a strong main metric for scam detection because
# it balances precision and recall.

primary_metric_name = "f1"

evaluation_metric_names = [
    "accuracy",
    "precision",
    "recall",
    "f1",
]


# ------------------------------------------------------------
# reproducibility
# ------------------------------------------------------------

training_random_seed = random_seed


# ------------------------------------------------------------
# helper
# ------------------------------------------------------------

def ensure_model_output_directories_exist() -> None:
    # Create output folders for saved models and reports.
    saved_models_directory.mkdir(parents=True, exist_ok=True)
    training_reports_directory.mkdir(parents=True, exist_ok=True)


def get_best_model_output_path() -> Path:
    # Full path for the saved best model.
    return saved_models_directory / best_model_file_name


def get_best_vectorizer_output_path() -> Path:
    # Full path for the saved vectorizer.
    return saved_models_directory / best_vectorizer_file_name


def get_training_results_output_path() -> Path:
    # Full path for the saved training results report.
    return training_reports_directory / training_results_file_name