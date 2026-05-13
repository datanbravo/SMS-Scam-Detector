import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "SMS Scam Backend is running"}

@app.get("/api/messages")
def get_messages():
    return []

@app.post("/api/messages")
def create_message(message: dict):
    return {
        "id": "demo-1",
        "name": message.get("name", "Anonymous"),
        "message_text": message.get("message", ""),
        "classification": "safe",
        "suspicious_phrases": [],
        "annotation_count": 0,
        "risk_categories_present": "",
        "short_explanation": "Backend received the message.",
    }
