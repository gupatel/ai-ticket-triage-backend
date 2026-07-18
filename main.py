from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import triage_ticket
import json
from datetime import datetime

app = FastAPI(title="AI Ticket Triage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ai-ticket-triage-frontend.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    issue_text: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/triage")
def triage(request: TicketRequest):
    if not request.issue_text.strip():
        raise HTTPException(status_code=400, detail="issue_text cannot be empty")

    try:
        result = triage_ticket(request.issue_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class FeedbackRequest(BaseModel):
    issue_text: str
    ai_category: str
    ai_priority: str
    was_correct: bool
    corrected_category: str | None = None

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    entry = feedback.dict()
    entry["timestamp"] = datetime.utcnow().isoformat()

    with open("data/feedback_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"status": "logged"}