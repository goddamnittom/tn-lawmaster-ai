from model_config import get_llm
from tn_law_agent.core import TNLawAgent

llm = get_llm()
agent = TNLawAgent(llm=llm)

print(f"✅ TN-LawMaster running with model: {__import__('model_config').ACTIVE_MODEL}\n")

while True:
    q = input("Enter legal question (or 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break
    result = agent.analyze(q)
    print("\n" + result.get("analysis", "No analysis") + "\n")