import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from query_parser import parse_queries

def safe_json_loads(data):
    """Safely parse JSON output from LLM, even if extra text is present."""
    try:
        start = data.find('[')
        end = data.rfind(']') + 1
        if start != -1 and end != -1:
            data = data[start:end]
        return json.loads(data)
    except Exception as e:
        print("❌ JSON parsing failed:", e)
        print("🔹 Raw LLM Output:\n", data)
        return []

def init_retriever(index_path="faiss_index"):
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})  
def multi_retriever(parsed_query_json, retriever):
    """
    Retrieves relevant clauses from FAISS based on parsed queries.
    Uses query + key_clauses + context for better retrieval.
    """
    parsed_queries = safe_json_loads(parsed_query_json)
    all_docs = []

    for pq in parsed_queries:
        query = pq.get("query", "")
        key_clauses = " ".join(pq.get("key_clauses", []))
        context = " ".join(pq.get("additional_context", []))

        # Build combined semantic query
        combined_query = f"{query} {key_clauses} {context}".strip()

        results = retriever.get_relevant_documents(combined_query)
        all_docs.extend([r.page_content for r in results])

    # Deduplicate docs while preserving order
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)

    print("\n🔎 Preview of top retrieved docs:")
    for i, doc in enumerate(unique_docs[:2]):
        print(f"\n📄 [Doc {i+1}]")
        print(doc[:200] + "...\n")

    return unique_docs[:3]  # ✅ limit to top 2 results


if __name__ == "__main__":
    queries = ["Does my policy cover artificial pregnancy?", "What is the grace period?"]
    parsed = parse_queries(queries)

    print("\n🧠 Parsed Query Output:\n", parsed)

    retriever = init_retriever("faiss_index")
    results = multi_retriever(parsed, retriever)

    print("\n✅ Final Ranked Clauses:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res[:200]}")
