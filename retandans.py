import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from query_parser import parse_queries  # This must exist

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(" GROQ_API_KEY not found in .env")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-70b-8192",
    temperature=0.1
)

def safe_json_loads(data):
    try:
        start = data.find('[')
        end = data.rfind(']') + 1
        if start != -1 and end != -1:
            data = data[start:end]
        return json.loads(data)
    except Exception as e:
        print(" JSON parsing failed:", e)
        print(" Raw LLM Output:\n", data)
        return []

def init_retriever(index_path="faiss_index"):
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

def retrieve_docs(parsed_queries, retriever):
    all_results = []
    for pq in parsed_queries:
        query = pq.get("query", "")
        key_clauses = " ".join(pq.get("key_clauses", []))
        context = " ".join(pq.get("additional_context", []))
        combined = f"{query} {key_clauses} {context}".strip()
        results = retriever.get_relevant_documents(combined)
        all_results.append({
            "query": query,
            "parsed": pq,
            "docs": list({r.page_content for r in results})  # deduplicated
        })
    return all_results


def generate_answer(query, retrieved_docs, parsed_info=None):
    context = "\n\n".join(retrieved_docs[:5]) if retrieved_docs else "No relevant information found."

    prompt_template = PromptTemplate.from_template("""
    You are an expert insurance policy assistant.
    Based strictly on the provided policy context, answer the question in ONE clear and complete sentence.

    If the context does not include the answer, say:
    "Information not found in the provided policy document. or no this is not covered under the policy."

    -----
    Question: {query}

    Policy Context:
    {context}
    -----
    Additional Info:
    {parsed_info}
    """)

    response = llm.invoke(prompt_template.format(
        query=query,
        context=context,
        parsed_info=parsed_info or "N/A"
    ))

    return response.content.strip()


def answer_all_queries(queries: list, index_path="faiss_index") -> list[str]:
    """
    Given a list of queries, returns clean, one-sentence answers for each.
    """
    parsed_raw = parse_queries(queries)
    parsed_queries = safe_json_loads(parsed_raw)

    if not parsed_queries:
        return ["Could not parse query." for _ in queries]

    retriever = init_retriever(index_path)
    retrievals = retrieve_docs(parsed_queries, retriever)

    answers = []
    for item in retrievals:
        answer = generate_answer(item["query"], item["docs"], parsed_info=item["parsed"])
        answers.append(answer)

    return answers


if __name__ == "__main__":
    queries = [
        "Does my policy cover artificial pregnancy?",
        "What is the grace period?",
        "Is accidental death covered under this policy?",
        "Are pre-existing diseases excluded?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "What is the extent of coverage for AYUSH treatments?",
    ]

    answers = answer_all_queries(queries)

    print("\n🧠 Final Answers:")
    for q, a in zip(queries, answers):
        print(f"Q: {q}\nA: {a}\n")
