import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from tech_support_ai.assistant import TechSupportAssistantModel

app = FastAPI(
    title="Tech.AI Laptop Support API",
    description="API for the enterprise laptop support AI assistant.",
    version="0.1.0",
)

# Enable CORS so the TypeScript frontend can call the API regardless of origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development. Secure this in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model upon app startup
# This loads the fallback seed data provided in datasets.py
model = TechSupportAssistantModel.build_default()

# -----------------
# API Data Models
# -----------------
class SupportQueryRequest(BaseModel):
    query: str

class SupportQueryResponse(BaseModel):
    issue_category: str
    confidence: float
    answer: str
    steps: List[str]
    evidence_doc_ids: List[str]

# -----------------
# API Endpoints
# -----------------
@app.post("/api/support", response_model=SupportQueryResponse)
async def ask_support_bot(request: SupportQueryRequest) -> SupportQueryResponse:
    """
    Receives an IT support query from the user, processes it via the Tech.AI model,
    and returns categorized steps + confidence and evidence.
    """
    # 1. Ask the model
    result = model.answer(request.query)
    
    # 2. Map the dataclass to Pydantic for validation and serialization
    return SupportQueryResponse(
        issue_category=result.issue_category,
        confidence=result.confidence,
        answer=result.answer,
        steps=result.steps,
        evidence_doc_ids=result.evidence_doc_ids
    )

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    # If file is executed via `python scripts/api.py`
    uvicorn.run("scripts.api:app", host="0.0.0.0", port=8000, reload=True)
