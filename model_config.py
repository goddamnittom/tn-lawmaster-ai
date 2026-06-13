"""
TN-LawMaster Model Switcher
Change ACTIVE_MODEL in one place only.
"""

from langchain_ollama import ChatOllama

# ============================================
# CHANGE MODEL HERE
# ============================================
ACTIVE_MODEL = "gemma3-1070"          # Options: gemma3-1070, gemma2:2b, llama3.2:3b

MODEL_CONFIGS = {
    "gemma3-1070": {
        "model": "gemma3-1070",
        "num_ctx": 4096,
        "temperature": 0.1,
        "num_thread": 6,
    },
    "gemma2:2b": {
        "model": "gemma2:2b",
        "num_ctx": 4096,
        "temperature": 0.1,
        "num_thread": 4,
    },
    "llama3.2:3b": {
        "model": "llama3.2:3b",
        "num_ctx": 4096,
        "temperature": 0.1,
        "num_thread": 6,
    },
}

def get_llm():
    config = MODEL_CONFIGS.get(ACTIVE_MODEL, MODEL_CONFIGS["gemma3-1070"])
    return ChatOllama(
        model=config["model"],
        temperature=config["temperature"],
        num_ctx=config["num_ctx"],
        num_thread=config["num_thread"],
    )