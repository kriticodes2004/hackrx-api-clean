import os
import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #previously chromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def create_faiss_index(text: str, source_url: str, index_path: str = "faiss_index") -> FAISS:
   
    docs = [Document(page_content=text, metadata={"source": source_url})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f" Chunked text into {len(chunks)} segments.")

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(index_path)
    print(f" FAISS index created and saved at '{index_path}' with metadata.")

    return vectordb
device = "cuda" if torch.cuda.is_available() else "cpu"
def load_faiss_index(index_path: str = "faiss_index") -> FAISS:
    
    embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
    vectordb = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    print(f" Loaded FAISS index from '{index_path}'")
    return vectordb

if __name__ == "__main__":
    from loader import load_document_from_url
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    text = load_document_from_url(test_url)
    db = create_faiss_index(text, source_url=test_url)

    retriever = db.as_retriever(search_kwargs={"k": 2})
    query = "What is this document about?"
    results = retriever.get_relevant_documents(query)
    print("\n Top matches:")
    for r in results:
        print("-", r.metadata.get("source", "unknown"), "|", r.page_content[:200])
