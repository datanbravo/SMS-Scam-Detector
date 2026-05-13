# SMS Scam Detector
Machine Learning Project - CS 4361

## What this does.

This project builds a small SMS scam detector. It uses:

- real SMS spam data.
- some synthetic scam examples.
- some synthetic normal examples so the model does not panic over every message.
- sentence embeddings.
- a few sklearn models.
- rule-based phrase checks for explanations.

The final output has three possible user-facing labels:

- `safe`.
- `suspicious`.
- `scam`.

I added `suspicious` because some messages are sketchy but not enough to fully call a scam. That makes the project way less dramatic lol.

---

## Files.

- `config.py` has general project settings.
- `dataset_pipeline.py` builds the CSV files.
- `preprocessing.py` cleans text and finds suspicious phrases.
- `synthetic_data.py` makes extra scam and normal examples.
- `model_training.py` trains the models.
- `model_config.py` has model settings.
- `run_detector.py` lets you test messages in the terminal.

---

## How to run it.

First install the packages:

```bash
pip install -r requirements.txt
```

Then run these in this order:

```bash
python dataset_pipeline.py
python model_training.py
python run_detector.py
```

The first file builds the dataset. The second trains the model. The third lets you test texts. In the detector, type `q`, `Q`, `exit`, or `EXIT` to quit.

---

## Example test texts.

Try these:

```text
Urgent: your account will be locked unless you verify now at https://fake.com
```

```text
Hey are we still meeting after class?
```

```text
Account will be locked.
```

The last one might be `safe` or `suspicious` depending on the model confidence. That is okay because it is super short and missing context.

---

## What changed in this version.

- Added scam confidence scores.
- Added a safer `suspicious` middle label.
- Made the model require stronger confidence before saying `scam`.
- Added more short scam examples.
- Added normal examples with words like account, payment, delivery, and verification so false positives are lower.
- Kept comments and code style more simple/student-like.
