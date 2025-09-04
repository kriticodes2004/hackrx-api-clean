
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

print(" Current working directory:", os.getcwd())

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
print(" Loaded GROQ_API_KEY:", groq_api_key)   # Debugging

if not groq_api_key:
    raise ValueError(" GROQ_API_KEY not found! Please check your .env file.")

from langchain_groq import ChatGroq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-70b-8192"
)
def parse_queries(queries):
    
    parse_prompt = PromptTemplate.from_template("""
    You are an expert in understanding insurance/legal queries.
    Parse the following questions and extract for EACH:
    - domain (e.g., insurance, legal, healthcare)
    - sub_domain (e.g., health, car, life, property, etc.)
    - key_clauses (main clause name or type the user is asking about)
    - additional_context (any other important details like duration, time period, beneficiary)
   
    Example 1:
    Query: "Does my policy cover artificial pregnancy?"
    Output:
    {{
      "query": "Does my policy cover artificial pregnancy?",
      "domain": "insurance",
      "sub_domain": "health",
      "key_clauses": ["artificial pregnancy coverage"],
      "additional_context": ["fertility treatments"],
      
    }}

    Example 2:
    Query: "What is the grace period for premium payment?"
    Output:
    {{
      "query": "What is the grace period for premium payment?",
      "domain": "insurance",
      "sub_domain": "life",
      "key_clauses": ["grace period clause"],
      "additional_context": ["premium due date", "policy lapse rules"],
       }}

    Example 3:
    Query: "What is the waiting period before surgery coverage starts?"
    Output:
    {{
      "query": "What is the waiting period before surgery coverage starts?",
      "domain": "insurance",
      "sub_domain": "health",
      "key_clauses": ["waiting period clause"],
      "additional_context": ["surgery eligibility", "policy start date"],
     }}

    Example 4:
    Query: "Is accidental death covered under this policy?"
    Output:
    {{
      "query": "Is accidental death covered under this policy?",
      "domain": "insurance",
      "sub_domain": "life",
      "key_clauses": ["accidental death benefit"],
      "additional_context": ["death benefit eligibility", "coverage conditions"],
     }}

    Example 5:
    Query: "Does my car insurance include natural disaster damage?"
    Output:
    {{
      "query": "Does my car insurance include natural disaster damage?",
      "domain": "insurance",
      "sub_domain": "vehicle",
      "key_clauses": ["natural disaster coverage"],
      "additional_context": ["comprehensive plan", "flood, earthquake"],
     }}

    Example 6:
    Query: "Are pre-existing conditions excluded from this health plan?"
    Output:
    {{
      "query": "Are pre-existing conditions excluded from this health plan?",
      "domain": "insurance",
      "sub_domain": "health",
      "key_clauses": ["pre-existing condition exclusion"],
      "additional_context": ["chronic illness", "policy limitations"],
      }}

    Now parse the following questions and return a JSON array following this pattern:

    Questions: {queries}
    """)

    structured_query = llm.invoke(parse_prompt.format(queries=queries))
    return structured_query.content

if __name__ == "__main__":
    test_queries = ["What is the grace period?", "Does my policy cover artificial pregnancy?"]
    print("\n Parsed Queries:")
    print(parse_queries(test_queries))
