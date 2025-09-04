import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(" GROQ_API_KEY not found in environment variables!")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-70b-8192",
    temperature=0.1
)

def generate_answer(query, retrieved_docs, parsed_info=None):
    """
    Generates concise, structured answers from retrieved policy text.
    Falls back to 'Information not found' if no relevant context is available.
    """
    context = "\n\n".join(retrieved_docs[:5]) if retrieved_docs else "No relevant information found."

    prompt_template = PromptTemplate.from_template("""
    You are an expert insurance policy assistant.
    Answer the question STRICTLY based on the given context.
    If the context does not mention the answer, say:
    "Information not found in the provided policy document."

    Output JSON only in this format:
    {{
      "query": "{query}",
      "answer": "<one-line answer or 'Information not found in the provided policy document.'>",
      "supporting_clauses": ["list of relevant clauses from context"],
      "explanation": "short reasoning (1-2 sentences)"
    }}

    -----
    Policy Context:
    {context}
    -----
    Additional Parsed Info:
    {parsed_info}
    """)

    response = llm.invoke(prompt_template.format(
        query=query,
        context=context,
        parsed_info=parsed_info or "N/A"
    ))

    
    try:
        start = response.content.find("{")
        end = response.content.rfind("}") + 1
        return json.loads(response.content[start:end])
    except Exception:
        return {
            "query": query,
            "answer": "Information not found in the provided policy document.",
            "supporting_clauses": [],
            "explanation": "No relevant clauses were retrieved for this query."
        }
    

if __name__ == "__main__":
    sample_query = "Does my policy cover artificial pregnancy?"
    sample_docs = [
        "Clause 5.2: This policy covers maternity expenses including artificial pregnancy procedures under medical advice."
    ]
    parsed_info = {"domain": "insurance", "sub_domain": "health", "key_clauses": ["artificial pregnancy"]}

    print("🧠 Generated Answer:")
    print(generate_answer(sample_query, sample_docs, parsed_info))
