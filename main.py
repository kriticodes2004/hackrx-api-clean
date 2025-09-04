
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
from loader import load_document_from_url
from embed import create_faiss_index
from retandans import answer_all_queries

app = FastAPI()

EXPECTED_API_KEY = "bb5c875952970375580fee401d1c04f00ef766644baa1e41b1a0e50ff517d6dc"

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/api/v1/hackrx/run")
async def hackrx_run(body: QueryRequest, authorization: Optional[str] = Header(None)):
    # 1️⃣ Auth check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    try:
        
        print("Loading document...")
        text = load_document_from_url(body.documents)

        
        print(" Creating vector index...")
        create_faiss_index(text, source_url=body.documents)

        
        print(" Answering questions...")
        answers = answer_all_queries(body.questions)
        print(" Answers ready.")

        return {"answers": answers}

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
