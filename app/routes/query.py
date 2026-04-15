from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from app.services.rag import ask_question

router = APIRouter()

INJECTION_PHRASES = [
    "ignore", "disregard", "forget", "override", "you are now",
    "act as", "pretend", "new instruction", "system prompt",
]

class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if len(v) > 500:
            raise ValueError("Query too long (max 500 characters)")
        lower = v.lower()
        for phrase in INJECTION_PHRASES:
            if phrase in lower:
                raise ValueError(f"Query contains disallowed phrase: '{phrase}'")
        return v

@router.post("/query")
def query(req: QueryRequest):
    return ask_question(req.query)