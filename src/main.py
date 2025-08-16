import sys

from pathlib import Path

import uvicorn


sys.path.append(str(Path(__file__).parent.parent))


from fastapi import FastAPI
from pydantic import BaseModel
from src.llm import get_answer_with_sources

app = FastAPI(title="EORA QA Bot", description="Ответы на основе кейсов EORA", version="1.0")

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest):
    answer, sources = get_answer_with_sources(payload.question)
    return QAResponse(answer=answer.content, sources=sources)

@app.get("/")
def root():
    return {"message": "Добро пожаловать в EORA QA Bot API"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)