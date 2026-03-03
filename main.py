import os
import json
import uuid
import hashlib
import requests
from typing import List, Dict, Any
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434"

SUMMARY_MODEL = "qwen2.5:0.5b"
AGENT_MODEL = "mistral:7b-instruct"
EMBED_MODEL = "nomic-embed-text"

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "agentic_doc_store"
SUMMARY_CACHE_FILE = "summary_cache.json"


# =========================
# UTILS
# =========================

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_summary_cache() -> Dict:
    if os.path.exists(SUMMARY_CACHE_FILE):
        with open(SUMMARY_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_summary_cache(cache: Dict):
    with open(SUMMARY_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# =========================
# FILE LOADING
# =========================

def load_document(path: str) -> str:
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join([p.extract_text() for p in reader.pages])
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")


# =========================
# CHUNKING
# =========================

def split_into_chunks(text: str, size=800, overlap=150):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks


# =========================
# OLLAMA CALLS
# =========================

def ollama_generate(model: str, prompt: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def ollama_embed(text: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )
    return response.json()["embedding"]


# =========================
# SUMMARY WITH CACHING
# =========================

def summarize_chunk(file_hash: str, chunk_id: int, text: str, cache: Dict):

    cache_key = f"{file_hash}_{chunk_id}"

    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""
Summarize in 1 sentence focusing on major theme:

{text}
"""
    summary = ollama_generate(SUMMARY_MODEL, prompt).strip()
    cache[cache_key] = summary
    save_summary_cache(cache)
    return summary


# =========================
# CHROMA
# =========================

def get_collection():
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_DIR,
        is_persistent=True
    ))
    return client.get_or_create_collection(COLLECTION_NAME)


# =========================
# INGEST
# =========================

def ingest_document(path: str):

    text = load_document(path)
    file_hash = hash_text(text)

    collection = get_collection()
    cache = load_summary_cache()

    chunks = split_into_chunks(text)
    as_rag_chunks = []

    for i, chunk in enumerate(chunks):

        summary = summarize_chunk(file_hash, i, chunk, cache)

        as_chunk = {
            "header": "Document Context",
            "content": chunk,
            "footer": f"Chunk {i}",
            "metadata": {
                "id": i,
                "file_hash": file_hash,
                "prev_id": i-1 if i > 0 else None,
                "next_id": i+1 if i < len(chunks)-1 else None,
                "summary": summary
            }
        }

        embedding = ollama_embed(chunk)

        clean_metadata = {}

        for k, v in as_chunk["metadata"].items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)

        collection.add(
            ids=[f"{file_hash}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[clean_metadata]
        )

        as_rag_chunks.append(as_chunk)

    print("Ingestion complete.")
    return file_hash


# =========================
# TOOLS
# =========================

def vector_search(query: str, file_hash: str, k=3):

    collection = get_collection()
    embedding = ollama_embed(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results


def get_chunk_by_id(file_hash: str, chunk_id: int):

    collection = get_collection()
    doc_id = f"{file_hash}_{chunk_id}"

    result = collection.get(ids=[doc_id])

    if not result["documents"]:
        return None

    metadata = result["metadatas"][0]
    content = result["documents"][0]

    return {
        "header": "Document Context",
        "content": content,
        "footer": f"Chunk {chunk_id}",
        "metadata": metadata
    }


# =========================
# AGENT LOOP
# =========================

def agent_chat(query: str, file_hash: str):
    tools_description = f"""
    You are a document reasoning agent.

    You have access to external tools that retrieve information from a vector database of a single ingested document.

    The document has been split into numbered chunks.
    Each chunk has:
    - id (integer)
    - summary (short theme sentence)
    - prev_id (previous chunk id or -1)
    - next_id (next chunk id or -1)

    Your job:
    - Use tools to retrieve relevant chunks
    - Think step by step
    - Expand context if necessary using prev_id / next_id
    - Only give final answer once confident

    ------------------------------------------------
    TOOLS AVAILABLE
    ------------------------------------------------

    1) vector_search

    Purpose:
    Use this when you do NOT know which chunk is relevant.
    It performs semantic search over chunk embeddings.

    Input format:
    {{
      "query": "<user question rewritten if needed>",
      "file_hash": "{file_hash}"
    }}

    Returns:
    Top K matching chunks with:
    - ids
    - documents
    - metadata (including summary, prev_id, next_id)

    Use this first if unsure which chunk contains the answer.

    ------------------------------------------------

    2) get_chunk_by_id

    Purpose:
    Fetch a specific chunk by id.
    Use this after vector_search if you want:
    - Full context of a chunk
    - To expand using prev_id / next_id
    - To verify exact wording

    Input format:
    {{
      "file_hash": "{file_hash}",
      "chunk_id": <integer>
    }}

    Returns:
    {{
      "header": "...",
      "content": "...",
      "footer": "...",
      "metadata": {{
         "id": ...,
         "summary": "...",
         "prev_id": ...,
         "next_id": ...
      }}
    }}

    ------------------------------------------------

    STRATEGY

    Step 1:
    If unsure → call vector_search.

    Step 2:
    Inspect returned metadata summaries.
    Pick best chunk id.

    Step 3:
    Call get_chunk_by_id for that id.

    Step 4:
    If context seems incomplete:
    Use prev_id or next_id to expand.

    Step 5:
    Once confident → return final answer.

    ------------------------------------------------

    IMPORTANT

    You MUST respond in STRICT JSON format:

    {{
      "thought": "your reasoning",
      "action": "vector_search OR get_chunk_by_id OR final",
      "action_input": {{ ... }}
    }}

    Do not output anything outside JSON.
    """

    conversation = f"{tools_description}\nUser: {query}\n"

    while True:

        response = ollama_generate(AGENT_MODEL, conversation)

        try:
            parsed = json.loads(response)
        except:
            print("MODEL RAW OUTPUT:\n", response)
            break

        print("\n🧠 Thought:", parsed["thought"])

        if parsed["action"] == "final":
            print("\n✅ Final Answer:\n", parsed["action_input"])
            break

        elif parsed["action"] == "vector_search":
            print("🔎 Calling vector_search...")
            result = vector_search(
                parsed["action_input"]["query"],
                file_hash
            )

        elif parsed["action"] == "get_chunk_by_id":
            print("📦 Fetching chunk...")
            result = get_chunk_by_id(
                file_hash,
                parsed["action_input"]["chunk_id"]
            )

        else:
            print("Unknown action.")
            break

        observation = json.dumps(result)
        print("📥 Observation:", observation[:500], "...")

        conversation += f"\nModel: {response}\nObservation: {observation}\n"


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    file_path = "/Users/vks/Downloads/Major Project/01 Major project guidelines for students.pdf"

    file_hash = ingest_document(file_path)

    while True:
        user_query = input("\nAsk something (or 'exit'): ")
        if user_query.lower() == "exit":
            break

        agent_chat(user_query, file_hash)