# main.py

from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
from loader import load_document_from_url
from embed import create_faiss_index
from retandans import answer_all_queries

app = FastAPI()

# Replace this with your actual test token (or load from env)
EXPECTED_API_KEY = "bb5c875952970375580fee401d1c04f00ef766644baa1e41b1a0e50ff517d6dc"

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# ✅ Health check for Render
@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/hackrx/run")
async def hackrx_run(request: Request, body: QueryRequest, authorization: Optional[str] = Header(None)):
    # ✅ Step 1: Check for Bearer token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    try:
        # ✅ Step 2: Load document from URL
        print("📥 Loading document...")
        text = load_document_from_url(body.documents)

        # ✅ Step 3: Create FAISS index
        print("📚 Creating vector index...")
        create_faiss_index(text, source_url=body.documents)

        # ✅ Step 4: Answer all questions
        print("💬 Answering questions...")
        answers = answer_all_queries(body.questions)
        print(answers)

        return {"answers": answers}

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Local test only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
