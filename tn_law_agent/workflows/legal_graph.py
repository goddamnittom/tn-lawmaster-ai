from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
import operator

class LegalAgentState(TypedDict):
    query: str
    documents: Annotated[List[Dict], operator.add]
    reranked_docs: List[Dict]
    analysis: str
    citations: List[str]
    status: str

class TNLawGraph:
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store
        self.workflow = StateGraph(LegalAgentState)
        
        # Define the Production Pipeline
        self.workflow.add_node("retrieve_law", self.retrieve_law)
        self.workflow.add_node("rerank_context", self.rerank_context)
        self.workflow.add_node("legal_analyze", self.legal_analyze)
        self.workflow.add_node("generate_citations", self.generate_citations)
        
        # Define Edges
        self.workflow.add_edge(START, "retrieve_law")
        self.workflow.add_edge("retrieve_law", "rerank_context")
        self.workflow.add_edge("rerank_context", "legal_analyze")
        self.workflow.add_edge("legal_analyze", "generate_citations")
        self.workflow.add_edge("generate_citations", END)
        
        self.graph = self.workflow.compile()

    def retrieve_law(self, state):
        print("Step 1: Retrieving raw law from TCA Vector Store...")
        # In a real prod env, this calls the vector db. 
        # For the PR, we implement the architectural hook.
        docs = self.vector_store.search(state['query']) if self.vector_store else [{"text": "Sample TCA Law", "source": "TCA 39-1-1"}]
        return {"documents": docs, "status": "retrieved"}

    def rerank_context(self, state):
        print("Step 2: Reranking context for legal precision...")
        # Implement a simple relevance score or a cross-encoder call
        docs = state.get('documents', [])
        # Heuristic: prioritize docs that contain the core keywords of the query
        keywords = state['query'].lower().split()
        scored = []
        for doc in docs:
            score = sum(1 for k in keywords if k in doc['text'].lower())
            scored.append((score, doc))
        
        reranked = [doc for score, doc in sorted(scored, key=lambda x: x[0], reverse=True)]
        return {"reranked_docs": reranked[:5], "status": "reranked"}

    def legal_analyze(self, state):
        print("Step 3: Performing grounded legal analysis...")
        context = "\n".join([f"[{d['source']}] {d['text']}" for d in state['reranked_docs']])
        prompt = f"You are an expert TN Law AI. Base your answer ONLY on the provided context. If the answer isn't there, state that you cannot find a specific TCA reference.\n\nContext:\n{context}\n\nQuery: {state['query']}"
        
        response = self.llm.invoke([SystemMessage(content="You are a precise legal assistant."), HumanMessage(content=prompt)])
        return {"analysis": response.content, "status": "analyzed"}

    def generate_citations(self, state):
        print("Step 4: Mapping citations for auditability...")
        analysis = state['analysis']
        docs = state['reranked_docs']
        citations = []
        for doc in docs:
            if doc['source'] in analysis:
                citations.append(doc['source'])
        
        return {"citations": citations, "status": "cited"}

    def invoke(self, query):
        return self.graph.invoke({"query": query, "documents": [], "reranked_docs": [], "analysis": "", "citations": [], "status": "init"})
