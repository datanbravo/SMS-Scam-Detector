from datetime import datetime, timezone
from uuid import uuid4

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

messages = []

class MessageIn(BaseModel):
    name: str | None = None
    message: str

@app.get("/")
def home():
    return {"status": "SMS Scam Backend is running"}

@app.get("/api/messages")
def get_messages():
    return messages

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

    messages.insert(0, new_message)

    return new_message
