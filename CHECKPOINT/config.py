# config.py
import os


# Using llama3 for extraction/logic and tinyllama for fast summarization
EMBEDDER_MODEL = "nomic-embed-text"
SUMMARY_MODEL = "qwen2.5:0.5b"
DB_PATH = "./local_rag_db"
CACHE_PATH = "./doc_cache"


MAIN_MODEL = "gpt-oss:20b"
TEMPERATURE = 0

os.makedirs(CACHE_PATH, exist_ok=True)



