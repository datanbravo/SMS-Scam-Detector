# SMS Scam Detector  
Machine Learning Project – CS 4361

## Overview
This project builds a system to:
- classify SMS messages as **safe** or **scam**
- highlight **suspicious phrases**
- explain **why those phrases are risky**

The goal is not just detection, but shows users *why* a message is considered a scam.

---

## Project Structure

### Dataset Pipeline (data preparation)
- `config.py` → project settings and rules
- `dataset_pipeline.py` → main dataset builder
- `preprocessing.py` → cleaning + feature + annotation logic
- `source_validation.py` → validates sources and access
- `synthetic_data.py` → generates synthetic scam messages

### LLM Pipeline (analysis + explanation)
- `prompt_builder.py` → builds prompts for the model
- `llm_pipeline.py` → runs predictions
- `llm_config.py` → LLM settings

---

## Dataset Features
The dataset includes:
- message text
- cleaned text
- label (`safe` or `scam`)
- scam subtype (optional)
- suspicious phrases
- risk categories (e.g. urgency, payment_request)
- explanations for each phrase

Example annotation:
```json
{
  "phrase_text": "urgent",
  "risk_category": "urgency",
  "risk_explanation": "This phrase creates pressure to act quickly."
}
