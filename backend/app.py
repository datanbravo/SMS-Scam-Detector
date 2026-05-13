from datetime import datetime, timezone
from uuid import uuid4

import os
import json
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

class MessageIn(BaseModel):
    name: str | None = None
    message: str

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

    trigger_words = [
        ("urgent", "urgency", "Urgent wording pressures people."),
        ("verify", "verification", "Requests to verify accounts are common scams."),
        ("payment", "money", "Money/payment language is suspicious."),
        ("click", "link", "Links can lead to phishing sites."),
        ("violation", "threat", "Threat-based language is suspicious."),
        ("suspended", "fear", "Fear tactics are common in scams."),
    ]

    suspicious_phrases = []

    lowered = text.lower()

    for word, category, explanation in trigger_words:
        start = lowered.find(word)

        if start != -1:
            suspicious_phrases.append({
                "phrase_text": text[start:start+len(word)],
                "start_index": start,
                "end_index": start + len(word),
                "risk_category": category,
                "risk_explanation": explanation,
            })

    looks_scam = len(suspicious_phrases) > 0

    confidence = round(min(0.55 + (0.08 * len(suspicious_phrases)), 0.99), 2)

    new_message = {
        "id": str(uuid4()),
        "name": payload.name or "Anonymous diver",
        "avatar": "🦈",
        "message_text": text,
        "classification": "scam" if looks_scam else "safe",
        "suspicious_phrases": suspicious_phrases,
        "annotation_count": len(suspicious_phrases),
        "risk_categories_present": ", ".join(
            list(set(p["risk_category"] for p in suspicious_phrases))
        ),
        "short_explanation": (
            f"Scam confidence: {confidence * 100:.0f}%"
            if looks_scam
            else "No major scam patterns detected."
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
