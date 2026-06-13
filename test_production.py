import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tn_law_agent.workflows.legal_graph import TNLawGraph

load_dotenv()
os.environ["OPENAI_API_KEY"] = "REDACTED_OPENAI_API_KEY"

# Mock Vector Store for testing
class MockVectorStore:
    def search(self, query):
        # Simple keyword search in our synthetic files
        results = []
        for filename in os.listdir('/root/tn-lawmaster-ai/data/tca_raw/'):
            with open(f'/root/tn-lawmaster-ai/data/tca_raw/{filename}', 'r') as f:
                content = f.read()
                if any(word in content.lower() for word in query.lower().split()):
                    results.append({"text": content, "source": filename})
        return results

llm = ChatOpenAI(model="gpt-4o")
vector_store = MockVectorStore()
agent = TNLawGraph(llm, vector_store)

query = "What is the penalty for theft in Tennessee?"
result = agent.invoke(query)

print(f"QUERY: {query}")
print("-" * 30)
print(f"ANALYSIS:\n{result['analysis']}")
print("-" * 30)
print(f"CITATIONS: {result['citations']}")
