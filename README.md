# SMS Scam Detector

A machine learning project that detects whether an SMS message is **safe** or a **scam**, and explains why by highlighting suspicious phrases.

---

# Project Overview

This project builds a **complete end-to-end pipeline** for SMS scam detection.

It does not just train a model — it handles:

* data collection (real + synthetic)
* preprocessing
* feature extraction (TF-IDF with unigrams and bigrams)
* model training and evaluation
* prediction on new messages
* explanation using rule-based phrase detection

---

# How It Works

The system follows this pipeline:

```
data collection
    ↓
dataset_pipeline.py
    ↓
preprocessing.py
    ↓
train/test datasets
    ↓
model_training.py
    ↓
trained model + vectorizer
    ↓
run_detector.py
    ↓
prediction + explanation
```

It is a **hybrid system**:

* ML model → decides *safe vs scam*
* Rule-based system → explains *why*

---

# File Structure & What Each File Does

## config.py

This is the **main configuration file** used across the project.

It defines:

* project paths (data, logs, outputs)
* dataset settings (split ratio, minimum length)
* synthetic data settings
* dataset column structure
* TF-IDF default configuration
* regex rules for suspicious phrases
* explanation templates
* source definitions (UCI dataset + government pages)


---

## dataset_pipeline.py

This file builds the **entire dataset**.

It:

* creates required folders
* downloads and parses the UCI SMS dataset
* scrapes scam examples from public government pages
* generates synthetic scam messages
* preprocesses the dataset
* splits into train/test sets
* saves:

  * raw dataset
  * processed dataset
  * train/test datasets
  * metadata and logs


---

## preprocessing.py

This file prepares text for machine learning.

It handles:

### Text Cleaning

* lowercases text
* replaces URLs, emails, phone numbers with tokens
* removes noise

### Feature Creation

* builds:

  * unigrams (single words)
  * bigrams (word pairs like `pay_now`)

### Suspicious Phrase Detection

* uses regex patterns from `config.py`
* extracts:

  * phrase text
  * position in message
  * risk category
  * explanation

### Extra Features

* message length
* digit count
* exclamation count
* URL/phone presence

---

## source_validation.py

Handles validation of data sources.

It:

* checks if sources are allowed
* verifies URLs are reachable
* checks robots.txt for scraping permission
* classifies network errors

Prevents invalid or blocked data collection.

---

## synthetic_data.py

Generates **synthetic scam messages**.

It:

* uses templates (urgency, threat, payment, etc.)
* fills them with random values (names, amounts, links)
* avoids duplicates

---

## model_config.py

Stores all model training settings.

Includes:

* dataset paths
* TF-IDF parameters
* enabled models:

  * logistic regression
  * decision tree
  * svm
* hyperparameters
* evaluation metrics

---

## model_training.py

This file trains and evaluates models.

It:

1. loads train/test datasets
2. validates data
3. builds TF-IDF vectorizer
4. converts text → numeric features
5. trains models:

   * logistic regression
   * decision tree
   * svm
6. evaluates using:

   * accuracy
   * precision
   * recall
   * F1-score
7. selects best model (based on F1)
8. saves:

   * model
   * vectorizer
   * results report

---

## run_detector.py

This is the **final detector (CLI)**.

It:

* loads trained model + vectorizer
* takes user input
* preprocesses message
* predicts safe/scam
* extracts suspicious phrases
* prints structured result

Example output includes:

* classification
* suspicious phrases
* risk categories
* explanation

---

# Core Concepts Used

## TF-IDF + N-grams

Text is converted into numbers using:

* unigrams (single words)
* bigrams (word pairs)

Example:

```
"pay now"
→ pay, now, pay_now
```

---

## Models Used

* Logistic Regression
* Decision Tree
* Support Vector Machine (SVM)

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score (main metric)

---

## Rule-Based Explanation Layer

After prediction, the system uses regex rules to detect:

* urgency
* threats
* payment requests
* impersonation
* delivery scams
* etc.

This makes results **explainable**.

---

# How to Run the Project

## Step 1: Build Dataset

```
python dataset_pipeline.py
```

---

## Step 2: Train Models

```
python model_training.py
```

---

## Step 3: Run Detector

```
python run_detector.py
```

Then enter SMS messages to test.

---

# Example Output

```json
{
  "message_text": "Urgent: verify your account now",
  "classification": "scam",
  "suspicious_phrases": [...],
  "annotation_count": 1,
  "risk_categories_present": "urgency",
  "short_explanation": "The model predicted this message as scam, and suspicious phrases support it."
}
```

---

# Current Limitations

* Model may overpredict scam if safe data is limited
* Regex rules don’t cover all scam variations
* Synthetic data may bias results
* CLI only (no UI)

---

# Future Improvements

* add more real safe messages
* improve regex coverage
* balance dataset better
* add probability/confidence scores
* build UI or mobile integration?

---

# Summary

This project is a complete SMS scam detection system that:

* builds its own dataset
* preprocesses text
* extracts TF-IDF features
* trains and evaluates models
* predicts scam vs safe
* explains suspicious phrases
