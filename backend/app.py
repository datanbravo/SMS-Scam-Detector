from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4

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

    looks_scam = any(
        word in text.lower()
        for word in ["urgent", "verify", "payment", "suspended", "click", "notice", "violation"]
    )

    new_message = {
        "id": str(uuid4()),
        "name": payload.name or "Anonymous diver",
        "avatar": "🦈",
        "message_text": text,
        "classification": "scam" if looks_scam else "safe",
        "suspicious_phrases": [],
        "annotation_count": 0,
        "risk_categories_present": "possible scam language" if looks_scam else "",
        "short_explanation": "This message has suspicious scam-like wording." if looks_scam else "No major scam patterns detected.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    messages.insert(0, new_message)
    return new_message
