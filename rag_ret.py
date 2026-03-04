import hashlib
import json
import os
from typing import List, Optional
from pypdf import PdfReader

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

from config import MAIN_MODEL, SUMMARY_MODEL, DB_PATH, CACHE_PATH

# 1. Initialize Tools
embeddings = OllamaEmbeddings(model=MAIN_MODEL)
summarizer_llm = OllamaLLM(model=SUMMARY_MODEL)
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Paths for our persistent caches
SUMMARY_CACHE_PATH = os.path.join(CACHE_PATH, "chunk_summaries.json")
CATALOG_PATH = os.path.join(CACHE_PATH, "catalog.json")


def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_json_cache(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_summary_on_demand(chunk_content: str, chunk_id: str) -> str:
    """Checks cache for a summary, generates and saves if missing."""
    cache = load_json_cache(SUMMARY_CACHE_PATH)

    if chunk_id in cache:
        return cache[chunk_id]

    # Generate if not found
    prompt = f"Summarize this text in one sentence for context: {chunk_content[:600]}"
    summary = summarizer_llm.invoke(prompt).strip()

    # Update cache
    cache[chunk_id] = summary
    save_json_cache(SUMMARY_CACHE_PATH, cache)
    return summary


# --- AGENT TOOLS ---

@tool
def list_available_documents():
    """Returns all indexed document names and their IDs."""
    catalog = load_json_cache(CATALOG_PATH)
    if not catalog: return "No documents found."
    return "\n".join([f"- {v['name']} (ID: {k})" for k, v in catalog.items()])


@tool
def index_new_document(file_path: str):
    """Indexes a file. Does NOT generate chunk summaries (Lazy Loading)."""
    file_path = file_path.strip("'").strip('"')
    if not os.path.exists(file_path): return "File not found."

    file_hash = get_file_hash(file_path)
    catalog = load_json_cache(CATALOG_PATH)
    if file_hash in catalog: return "Already indexed."

    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        for page in PdfReader(file_path).pages:
            text += (page.extract_text() or "") + "\n"
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # Split into chunks
    raw_chunks = [text[i:i + 1000] for i in range(0, len(text), 800)]

    # Store in Vector DB with minimal metadata
    docs_to_add = []
    for i, content in enumerate(raw_chunks):
        c_id = f"{file_hash}_{i}"
        docs_to_add.append(Document(
            page_content=content,
            metadata={"doc_id": file_hash, "doc_name": os.path.basename(file_path), "id": c_id}
        ))

    vector_store.add_documents(docs_to_add, ids=[d.metadata["id"] for d in docs_to_add])

    # Update catalog
    catalog[file_hash] = {"name": os.path.basename(file_path), "chunk_count": len(raw_chunks)}
    save_json_cache(CATALOG_PATH, catalog)
    return f"Indexed {os.path.basename(file_path)}. Summaries will be generated on-demand."


@tool
def rag_search(query: str, doc_id: Optional[str] = None):
    """Search with context. Generates neighbor summaries only if they are missing from cache."""
    search_kwargs = {"k": 3}
    if doc_id: search_kwargs["filter"] = {"doc_id": doc_id}

    results = vector_store.similarity_search(query, **search_kwargs)

    final_results = []
    for r in results:
        curr_id = r.metadata["id"]
        # Fetch or generate summary for this specific chunk
        summary = get_summary_on_demand(r.page_content, curr_id)

        final_results.append({
            "content": r.page_content,
            "context_summary": summary,
            "metadata": r.metadata
        })
    return final_results


@tool
def fetch_chunks_by_id(chunk_ids: List[str]):
    """Fetch raw chunks by ID."""
    res = vector_store.get(ids=chunk_ids)
    return [{"id": res['ids'][i], "content": res['documents'][i]} for i in range(len(res['ids']))]


# --- PRE-HEAT FUNCTION (Importable, not a tool) ---

def pre_heat_summaries(doc_id: Optional[str] = None):
    """
    Run this manually to generate all missing summaries.
    If doc_id is None, it processes all documents.
    """
    print("🔥 Starting Pre-heat: Generating missing summaries...")
    where_filter = {"doc_id": doc_id} if doc_id else None
    all_chunks = vector_store.get(where=where_filter)

    total = len(all_chunks['ids'])
    for i in range(total):
        c_id = all_chunks['ids'][i]
        content = all_chunks['documents'][i]
        # This will trigger generation only if not in cache
        get_summary_on_demand(content, c_id)
        if i % 5 == 0:
            print(f"Progress: {i}/{total} chunks checked.")

    print("✅ Pre-heat complete. All summaries are cached.")