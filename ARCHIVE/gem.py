import os
import json
import hashlib
import requests
from typing import List, Dict, Any, Optional
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
# UTILS & FORMATTING
# =========================
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f: return json.load(f)
    return {}


def save_json(path: str, data: Dict):
    with open(path, "w") as f: json.dump(data, f, indent=2)


def format_id(idx: int) -> str:
    """Standardizes IDs to 4-digit strings (0001, 0002) for efficiency."""
    return f"{idx:04d}"


# =========================
# OLLAMA WRAPPERS
# =========================
def ollama_generate(model: str, prompt: str):
    response = requests.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": model, "prompt": prompt, "stream": False})
    return response.json().get("response", "")


def ollama_embed(text: str):
    response = requests.post(f"{OLLAMA_URL}/api/embeddings",
                             json={"model": EMBED_MODEL, "prompt": text})
    return response.json()["embedding"]


# =========================
# TOC GENERATOR
# =========================
def generate_toc(chunks: List[Dict]) -> List[Dict]:
    """Creates a high-level map of the document by looking at the start of chunks."""
    toc = []
    current_chapter = "Introduction / Start"
    start_id = "0000"

    for i, c in enumerate(chunks):
        content_sample = c['content'][:150].lower()
        # Heuristic for Chapter/Section detection
        if any(x in content_sample for x in ["chapter", "section", "table of contents"]):
            if i > 0:
                toc.append({"title": current_chapter, "range": f"{start_id}-{format_id(i - 1)}"})
            current_chapter = c['content'].split('\n')[0][:60].strip()
            start_id = format_id(i)

    toc.append({"title": current_chapter, "range": f"{start_id}-{format_id(len(chunks) - 1)}"})
    return toc


# =========================
# CORE TOOLS
# =========================
def vector_search(query: str, file_hash: str, k=3) -> List[Dict]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    query_embedding = ollama_embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"file_hash": file_hash}
    )

    formatted = []
    for i in range(len(results["ids"])):
        formatted.append({
            "id": results["metadatas"][i][0]["id"],
            "summary": results["metadatas"][i][0]["summary"]
        })
    return formatted


def get_chunks_by_ids(file_hash: str, chunk_ids: List[str]) -> List[Dict]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # We retrieve the target chunks plus their immediate neighbors' metadata
    output = []
    for cid in chunk_ids:
        # Get Current Chunk
        res = collection.get(ids=[f"{file_hash}_{cid}"])
        if not res["documents"]: continue

        meta = res["metadatas"][0]

        # Get neighbor summaries (minimal data transfer)
        prev_id = meta.get("prev_id")
        next_id = meta.get("next_id")

        prev_sum = "START OF DOC"
        if prev_id:
            p_res = collection.get(ids=[f"{file_hash}_{prev_id}"])
            if p_res["metadatas"]: prev_sum = p_res["metadatas"][0]["summary"]

        next_sum = "END OF DOC"
        if next_id:
            n_res = collection.get(ids=[f"{file_hash}_{next_id}"])
            if n_res["metadatas"]: next_sum = n_res["metadatas"][0]["summary"]

        output.append({
            "chunk_id": cid,
            "prev_chunk_summary": prev_sum,
            "content": res["documents"][0],
            "next_chunk_summary": next_sum
        })
    return output


# =========================
# INGESTION
# =========================
def ingest_document(path: str):
    print(f"📄 Loading: {path}")
    reader = PdfReader(path)
    text = "\n".join([p.extract_text() for p in reader.pages])
    file_hash = hash_text(text)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    cache = load_json(SUMMARY_CACHE_FILE)

    # Chunking: 800 chars with 150 overlap
    raw_chunks = [text[i:i + 800] for i in range(0, len(text), 650)]
    processed_chunks = []

    print(f"📦 Processing {len(raw_chunks)} chunks...")

    for i, content in enumerate(raw_chunks):
        cid = format_id(i)
        cache_key = f"{file_hash}_{cid}"

        summary = cache.get(cache_key)
        if not summary:
            # Short theme-based summary
            summary = ollama_generate(SUMMARY_MODEL, f"Summarize major theme in 1 sentence: {content}").strip()
            cache[cache_key] = summary

        metadata = {
            "id": cid,
            "file_hash": file_hash,
            "summary": summary,
            "next_id": format_id(i + 1) if i < len(raw_chunks) - 1 else None,
            "prev_id": format_id(i - 1) if i > 0 else None
        }

        collection.upsert(
            ids=[f"{file_hash}_{cid}"],
            embeddings=[ollama_embed(content)],
            documents=[content],
            metadatas=[{k: v for k, v in metadata.items() if v is not None}]
        )
        processed_chunks.append({"content": content, "metadata": metadata})

    save_json(SUMMARY_CACHE_FILE, cache)
    toc = generate_toc(processed_chunks)

    manifest = {
        "file_hash": file_hash,
        "toc": toc,
        "doc_summary": f"Document: {os.path.basename(path)} (Hash: {file_hash[:8]})"
    }
    return manifest


# =========================
# AGENT LOOP
# =========================
def agent_chat(query: str, manifest: Dict):
    file_hash = manifest['file_hash']

    system_prompt = f"""
    You are a professional Document Reasoning Agent.

    DOC SUMMARY: {manifest['doc_summary']}
    TABLE OF CONTENTS (Chunk Ranges):
    {json.dumps(manifest['toc'], indent=2)}

    TOOLS:
    1. vector_search: {{"query": "text", "file_hash": "{file_hash}"}} -> Returns ID and Summary.
    2. get_chunks_by_ids: {{"ids": ["0001", "0002"], "file_hash": "{file_hash}"}} -> Returns full content + neighbor summaries.

    STRATEGY:
    - Use vector_search if you don't know where to look.
    - Use get_chunks_by_ids to read specific chunks or expand context.
    - Use TOC to identify logical sections.

    STRICT JSON RESPONSE:
    {{
      "thought": "Your step-by-step reasoning",
      "action": "vector_search" | "get_chunks_by_ids" | "final",
      "action_input": {{ ... }}
    }}
    """

    conversation = f"System: {system_prompt}\nUser: {query}\n"

    for _ in range(10):  # Max 10 turns to avoid infinite loops
        response = ollama_generate(AGENT_MODEL, conversation)

        try:
            # Attempt to extract JSON if model adds fluff
            clean_json = response[response.find("{"):response.rfind("}") + 1]
            parsed = json.loads(clean_json)
        except Exception:
            print(f"⚠️ Model Error. Raw Output: {response}")
            break

        print(f"\n🧠 {parsed.get('thought')}")

        if parsed["action"] == "final":
            print(f"\n✅ FINAL ANSWER:\n{parsed['action_input']}")
            break

        # Tool Execution
        obs = "No observation."
        action = parsed["action"]
        input_data = parsed["action_input"]

        if action == "vector_search":
            print(f"🔎 Searching for: {input_data.get('query')}")
            obs = vector_search(input_data["query"], file_hash)

        elif action == "get_chunks_by_ids":
            print(f"📦 Fetching Chunks: {input_data.get('ids')}")
            obs = get_chunks_by_ids(file_hash, input_data["ids"])

        conversation += f"\nAgent: {json.dumps(parsed)}\nObservation: {json.dumps(obs)}\n"


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Update this path to your file
    target_file = "/Users/vks/Downloads/Major Project/01 Major project guidelines for students.pdf"

    if os.path.exists(target_file):
        doc_manifest = ingest_document(target_file)

        while True:
            q = input("\nQuestion (or 'exit'): ")
            if q.lower() in ['exit', 'quit']: break
            agent_chat(q, doc_manifest)
    else:
        print(f"File not found: {target_file}")