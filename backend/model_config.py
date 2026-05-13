from __future__ import annotations

from pathlib import Path

from config import project_root_directory, random_seed


#Paths --------------------------------------
#These paths are only for the model-training part.


saved_models_directory = project_root_directory / "saved_models"
training_reports_directory = project_root_directory / "training_reports"

train_dataset_path = project_root_directory / "data" / "processed" / "train_dataset.csv"
test_dataset_path = project_root_directory / "data" / "processed" / "test_dataset.csv"

best_model_file_name = "best_sms_scam_model.pkl"
best_embedder_file_name = "best_sms_scam_embedder.pkl"
training_results_file_name = "model_results.json"


#Training data settings --------------------------------------
#This is the text column the model will learn from.
#This is the text field that is encoded into sentence embeddings.


training_text_column = "unigram_bigram_ready_text"
training_label_column = "label"


#Embedding settings --------------------------------------
#This sentence-transformers model converts text into dense numeric embeddings.


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"


#Model settings --------------------------------------
#These are the models we want to compare.
#Keep this list simple and aligned with class topics.

enabled_model_names = [
    "logistic_regression",
    "svm",
    "decision_tree",
]


#Logistic regression settings --------------------------------------


logistic_regression_max_iterations = 2000
logistic_regression_regularization_strength = 1.0


#Decision tree settings --------------------------------------

decision_tree_max_depth = None
decision_tree_min_samples_leaf = 1

#SVM settings --------------------------------------

svm_regularization_strength = 1.0
svm_kernel = "linear"
svm_use_probability = True

#Evaluation settings --------------------------------------
#F1-score is a strong main metric for scam detection because.
#it balances precision and recall.


primary_metric_name = "f1"

evaluation_metric_names = [
    "accuracy",
    "precision",
    "recall",
    "f1",
]


#Reproducibility --------------------------------------


training_random_seed = random_seed


#Helper --------------------------------------


def ensure_model_output_directories_exist() -> None:
    # Create output folders for saved models and reports.
    saved_models_directory.mkdir(parents=True, exist_ok=True)
    training_reports_directory.mkdir(parents=True, exist_ok=True)


def get_best_model_output_path() -> Path:
    # Full path for the saved best model.
    return saved_models_directory / best_model_file_name


def get_best_embedder_output_path() -> Path:
    #Full path for the saved embedding model.
    return saved_models_directory / best_embedder_file_name


def get_training_results_output_path() -> Path:
    # Full path for the saved training results report.
    return training_reports_directory / training_results_file_name
