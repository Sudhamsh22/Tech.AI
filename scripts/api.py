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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load trained model instead of retraining
MODEL_PATH = "artifacts/classifier.pt"
model = TechSupportAssistantModel.load_from_checkpoint(MODEL_PATH)


class SupportQueryRequest(BaseModel):
    query: str


class SupportQueryResponse(BaseModel):
    issue_category: str
    confidence: float
    answer: str
    steps: List[str]
    evidence_doc_ids: List[str]


@app.post("/api/support", response_model=SupportQueryResponse)
async def ask_support_bot(request: SupportQueryRequest) -> SupportQueryResponse:
    result = model.answer(request.query)
    return SupportQueryResponse(
        issue_category=result.issue_category,
        confidence=result.confidence,
        answer=result.answer,
        steps=result.steps,
        evidence_doc_ids=result.evidence_doc_ids,
    )


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)