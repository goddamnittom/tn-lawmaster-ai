import streamlit as st
from tn_law_agent.core import TNLawAgent
from langchain_groq import ChatGroq

st.set_page_config(page_title="TN-LawMaster", layout="wide")
st.title("TN-LawMaster - Tennessee Legal Expert")

with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password")
    domain = st.selectbox("Domain", ["criminal", "family", "property", "business", "torts", "estates", "traffic", "tipa", "general"])

if st.button("Initialize") and api_key:
    llm = ChatGroq(model="grok-beta", temperature=0.1, api_key=api_key)
    st.session_state.agent = TNLawAgent(llm)
    st.success("Agent ready")

query = st.text_area("Legal Query")
if st.button("Analyze") and "agent" in st.session_state:
    result = st.session_state.agent.analyze(query)
    st.markdown(result.get("analysis", ""))