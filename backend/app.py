from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

import os
import json
import re
import pickle
import unicodedata

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
conn.autocommit = True

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    name TEXT,
    avatar TEXT,
    message_text TEXT,
    classification TEXT,
    suspicious_phrases JSONB,
    annotation_count INTEGER,
    risk_categories_present TEXT,
    short_explanation TEXT,
    confidence FLOAT,
    created_at TEXT
)
""")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "saved_models" / "best_sms_scam_model.pkl"
EMBEDDER_PATH = PROJECT_DIR / "saved_models" / "best_sms_scam_embedder.pkl"

with open(MODEL_PATH, "rb") as model_file:
    scam_model = pickle.load(model_file)

with open(EMBEDDER_PATH, "rb") as embedder_file:
    embedder = pickle.load(embedder_file)


class MessageIn(BaseModel):
    name: str | None = None
    message: str

#Text processng
phone_number_pattern = re.compile(r"\b(?:\+?\d[\d\-\s()]{7,}\d)\b")
url_pattern = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
email_address_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)


def clean_message_text(message_text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", message_text)
    cleaned = cleaned.replace("\u00a0", " ")

    cleaned = url_pattern.sub(" url_token ", cleaned)
    cleaned = email_address_pattern.sub(" email_token ", cleaned)
    cleaned = phone_number_pattern.sub(" phone_token ", cleaned)

    cleaned = cleaned.lower()

    cleaned = re.sub(r"[^a-z0-9_'\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def create_unigram_bigram_ready_text(cleaned_message_text: str) -> str:
    tokens = cleaned_message_text.split()

    unigrams = list(tokens)
    bigrams = []

    for index in range(len(tokens) - 1):
        bigrams.append(f"{tokens[index]}_{tokens[index + 1]}")

    return " ".join(unigrams + bigrams)

#ML prediction
def get_scam_probability(message_text: str) -> float:
    cleaned_text = clean_message_text(message_text)
    model_text = create_unigram_bigram_ready_text(cleaned_text)

    embedded_text = embedder.encode([model_text])

    if hasattr(scam_model, "predict_proba"):
        probabilities = scam_model.predict_proba(embedded_text)[0]
        return float(probabilities[1])

    prediction = int(scam_model.predict(embedded_text)[0])
    return 1.0 if prediction == 1 else 0.0


trigger_words = [
    ("urgent", "urgency", "Urgent wording pressures people."),
    ("verify", "account_verification", "Requests to verify accounts are common in scams."),
    ("confirm", "account_verification", "Requests to confirm identity can be suspicious."),
    ("payment", "payment_request", "Money or payment language can be risky."),
    ("pay", "payment_request", "Payment requests are common in scam messages."),
    ("click", "link_request", "Links can lead to phishing sites."),
    ("link", "link_request", "Random links can lead to fake pages."),
    ("violation", "threat", "Threat-based language is suspicious."),
    ("suspended", "threat", "Fear tactics are common in scams."),
    ("locked", "threat", "Account lock threats are common in scams."),
    ("prize", "prize_scam", "Prize language is common in scam messages."),
    ("won", "prize_scam", "Unexpected prize claims are suspicious."),
    ("delivery", "delivery_scam", "Fake delivery notices are common scams."),
    ("package", "delivery_scam", "Package problems are often used in scams."),
]

def extract_suspicious_phrases(message_text: str) -> list[dict]:
    suspicious_phrases = []
    lowered = message_text.lower()

    for word, category, explanation in trigger_words:
        start = lowered.find(word)

        if start != -1:
            suspicious_phrases.append(
                {
                    "phrase_text": message_text[start:start + len(word)],
                    "start_index": start,
                    "end_index": start + len(word),
                    "risk_category": category,
                    "risk_explanation": explanation,
                }
            )

    return suspicious_phrases


def get_risk_categories_present(suspicious_phrases: list[dict]) -> str:
    categories = []

    for phrase in suspicious_phrases:
        category = phrase["risk_category"]

        if category not in categories:
            categories.append(category)

    return ", ".join(categories)


SCAM_CUTOFF = 0.72
SUSPICIOUS_CUTOFF = 0.45
STRONG_RULE_COUNT_CUTOFF = 2


def choose_final_classification(
    scam_probability: float,
    suspicious_phrases: list[dict],
) -> str:
    if scam_probability >= SCAM_CUTOFF:
        return "scam"

    if scam_probability >= SUSPICIOUS_CUTOFF:
        return "suspicious"

    if len(suspicious_phrases) >= STRONG_RULE_COUNT_CUTOFF:
        return "suspicious"

    return "safe"


def build_short_explanation(
    classification: str,
    scam_probability: float,
    suspicious_phrases: list[dict],
) -> str:
    percent = round(scam_probability * 100)

    if suspicious_phrases:
        phrase_text = ", ".join(
            f"'{phrase['phrase_text']}'"
            for phrase in suspicious_phrases[:3]
        )
    else:
        phrase_text = ""

    if classification == "scam":
        if phrase_text:
            return f"Scam confidence: {percent}%. Risky wording found: {phrase_text}."
        return f"Scam confidence: {percent}%."

    if classification == "suspicious":
        if phrase_text:
            return f"This looks suspicious. Scam confidence: {percent}%. Warning signs: {phrase_text}."
        return f"This looks suspicious. Scam confidence: {percent}%."

    if phrase_text:
        return f"Low scam confidence: {percent}%. Some words were noticed, but not enough to call it a scam."

    return f"Low scam confidence: {percent}%. No major scam patterns detected."


#Routes... Sorry, forgot to update this file. 
@app.get("/")
def home():
    return {"status": "SMS Scam Backend is running"}


@app.get("/api/messages")
def get_messages():
    cur.execute("""
        SELECT * FROM messages
        ORDER BY created_at DESC
    """)

    rows = cur.fetchall()
    return rows


@app.post("/api/messages")
def create_message(payload: MessageIn):
    text = payload.message.strip()

    scam_probability = get_scam_probability(text)
    suspicious_phrases = extract_suspicious_phrases(text)

    classification = choose_final_classification(
        scam_probability=scam_probability,
        suspicious_phrases=suspicious_phrases,
    )

    confidence = round(scam_probability, 4)

    new_message = {
        "id": str(uuid4()),
        "name": payload.name or "Anonymous diver",
        "avatar": "🦈",
        "message_text": text,
        "classification": classification,
        "suspicious_phrases": suspicious_phrases,
        "annotation_count": len(suspicious_phrases),
        "risk_categories_present": get_risk_categories_present(suspicious_phrases),
        "short_explanation": build_short_explanation(
            classification=classification,
            scam_probability=scam_probability,
            suspicious_phrases=suspicious_phrases,
        ),
        "confidence": confidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    cur.execute("""
    INSERT INTO messages (
        id,
        name,
        avatar,
        message_text,
        classification,
        suspicious_phrases,
        annotation_count,
        risk_categories_present,
        short_explanation,
        confidence,
        created_at
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        new_message["id"],
        new_message["name"],
        new_message["avatar"],
        new_message["message_text"],
        new_message["classification"],
        json.dumps(new_message["suspicious_phrases"]),
        new_message["annotation_count"],
        new_message["risk_categories_present"],
        new_message["short_explanation"],
        new_message["confidence"],
        new_message["created_at"],
    ))

    return new_message


@app.delete("/api/messages/{message_id}")
def delete_message(message_id: str):
    cur.execute("DELETE FROM messages WHERE id = %s", (message_id,))
    return {"deleted": True, "id": message_id}
