import os
import json
import hashlib
import requests
import re
from typing import List, Dict, Any
from pypdf import PdfReader
import chromadb

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434"
SUMMARY_MODEL = "qwen2.5:0.5b"
AGENT_MODEL = "gpt-oss:20b"
EMBED_MODEL = "nomic-embed-text"

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "agentic_doc_store"


# =========================
# ROBUST UTILS
# =========================
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def format_id(idx: int) -> str:
    return f"{idx:03d}"


def clean_json_response(raw_str: str) -> Dict:
    """Extracts and repairs JSON from LLM output."""
    # 1. Remove control characters that break JSON parsing
    clean_str = re.sub(r'[\x00-\x1F\x7F]', '', raw_str)

    # 2. Extract JSON block using regex
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in model response.")

    json_block = match.group()

    # 3. Handle common LLM JSON errors (like unescaped quotes in the middle of a string)
    try:
        return json.loads(json_block)
    except json.JSONDecodeError:
        # Emergency repair for newlines inside JSON values
        fixed = json_block.replace('\n', '\\n')
        return json.loads(fixed)


# =========================
# CORE TOOLS
# =========================
def vector_search(query: str, file_hash: str, k=4):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION_NAME)

    emb = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": query}).json()[
        "embedding"]
    res = col.query(query_embeddings=[emb], n_results=k, where={"file_hash": file_hash})

    return [{"id": m["id"], "summary": m["summary"]} for m in res["metadatas"][0]]


def get_chunks_grouped(file_hash: str, chunk_ids: List[str]):
    """Groups continuous chunks into a single clean text block."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION_NAME)

    ids = sorted(list(set(chunk_ids)))
    output_blocks = []

    for cid in ids:
        res = col.get(ids=[f"{file_hash}_{cid}"])
        if not res["documents"]: continue

        meta = res["metadatas"][0]
        # Get neighbor summaries for context
        p_res = col.get(ids=[f"{file_hash}_{format_id(int(cid) - 1)}"])
        n_res = col.get(ids=[f"{file_hash}_{format_id(int(cid) + 1)}"])

        output_blocks.append({
            "chunk_id": cid,
            "prev_sum": p_res["metadatas"][0]["summary"] if p_res["metadatas"] else "START",
            "content": res["documents"][0],
            "next_sum": n_res["metadatas"][0]["summary"] if n_res["metadatas"] else "END"
        })
    return output_blocks


# =========================
# AGENT LOOP
# =========================
def agent_chat(query: str, manifest: Dict):
    file_hash = manifest['file_hash']

    # Updated Prompt for 20B Models (uses clearer delimiters)
    prompt = f"""### INSTRUCTION
You are a Document Retrieval Agent. Use the following document context to answer the user query.
DOC: {manifest['doc_summary']}
TOC: {json.dumps(manifest['toc'])}

AVAILABLE TOOLS:
1. vector_search: {{"query": "search_term"}}
2. get_chunks_grouped: {{"ids": ["001", "002"]}}

STRICT OUTPUT FORMAT:
{{
  "thought": "your reasoning",
  "action": "vector_search" | "get_chunks_grouped" | "final",
  "action_input": "..." or {{"ids": ["..."]}}
}}

### USER QUERY
{query}
"""

    history = prompt
    for turn in range(5):
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate",
                                 json={"model": AGENT_MODEL, "prompt": history, "stream": False}).json()["response"]

            parsed = clean_json_response(resp)
            print(f"\n🧠 {parsed['thought']}")

            if parsed["action"] == "final":
                print(f"\n✅ FINAL ANSWER:\n{parsed['action_input']}")
                break

            # Tool logic
            if parsed["action"] == "vector_search":
                obs = vector_search(parsed["action_input"]["query"], file_hash)
            elif parsed["action"] == "get_chunks_grouped":
                obs = get_chunks_grouped(file_hash, parsed["action_input"]["ids"])
            else:
                obs = "Invalid action."

            history += f"\n### RESPONSE\n{json.dumps(parsed)}\n### OBSERVATION\n{json.dumps(obs)}"

        except Exception as e:
            print(f"⚠️ Error: {e}")
            break


# =========================
# INGESTION
# =========================
def ingest_document(path: str):
    print(f"📦 Loading {path}...")
    reader = PdfReader(path)
    text = "\n".join([p.extract_text() for p in reader.pages])
    file_hash = hash_text(text)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION_NAME)

    # 800 chars / 150 overlap
    chunks = [text[i:i + 800] for i in range(0, len(text), 650)]
    toc = []

    for i, content in enumerate(chunks):
        cid = format_id(i)
        # Fast summary
        sum_prompt = f"Summarize theme in 8 words: {content}"
        summary = requests.post(f"{OLLAMA_URL}/api/generate",
                                json={"model": SUMMARY_MODEL, "prompt": sum_prompt, "stream": False}).json()["response"]

        emb = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": content}).json()[
            "embedding"]

        col.upsert(
            ids=[f"{file_hash}_{cid}"],
            embeddings=[emb],
            documents=[content],
            metadatas=[{"id": cid, "file_hash": file_hash, "summary": summary.strip()}]
        )
        if i % 5 == 0:
            toc.append({"section": f"Segment {i}", "range": f"{cid}-{format_id(min(i + 4, len(chunks) - 1))}"})

    return {"file_hash": file_hash, "toc": toc, "doc_summary": f"Guidelines Document ({len(chunks)} chunks)"}


if __name__ == "__main__":
    file_path = "/Users/vks/Downloads/Major Project/01 Major project guidelines for students.pdf"
    manifest = ingest_document(file_path)

    while True:
        user_in = input("\nAsk: ")
        if user_in.lower() in ['exit', 'quit']: break
        agent_chat(user_in, manifest)