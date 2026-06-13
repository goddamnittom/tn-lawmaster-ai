from .workflows.legal_graph import TNLawGraph

class TNLawAgent:
    def __init__(self, llm):
        self.graph = TNLawGraph(llm)

    def analyze(self, query: str):
        return self.graph.invoke(query)