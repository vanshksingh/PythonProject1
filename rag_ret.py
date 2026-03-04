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

# Initialize
embeddings = OllamaEmbeddings(model=MAIN_MODEL)
summarizer_llm = OllamaLLM(model=SUMMARY_MODEL)
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

SUMMARY_CACHE_PATH = os.path.join(CACHE_PATH, "chunk_summaries.json")
CATALOG_PATH = os.path.join(CACHE_PATH, "catalog.json")


def load_json_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_json_cache(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_summary_on_demand(chunk_content: str, chunk_id: str) -> str:
    cache = load_json_cache(SUMMARY_CACHE_PATH)
    if chunk_id in cache:
        return cache[chunk_id]

    prompt = f"Summarize this text in one sentence for context: {chunk_content[:600]}"
    summary = summarizer_llm.invoke(prompt).strip()

    cache[chunk_id] = summary
    save_json_cache(SUMMARY_CACHE_PATH, cache)
    return summary


# --- TOOLS ---

@tool
def list_available_documents():
    """
    Returns names, IDs, and summaries of all indexed documents.
    Use this first to identify which DOC_ID to query.
    """
    catalog = load_json_cache(CATALOG_PATH)
    if not catalog: return "No documents found."
    return "\n".join([f"- {v['name']} (Use ID: {k}) | Total Chunks: {v['chunk_count']}" for k, v in catalog.items()])


@tool
def index_new_document(file_path: str):
    """Indexes a file. Assigns IDs like DOC1_001, DOC1_002 sequentially."""
    file_path = file_path.strip("'").strip('"')
    if not os.path.exists(file_path): return "File not found."

    catalog = load_json_cache(CATALOG_PATH)
    doc_index = len(catalog) + 1
    doc_prefix = f"DOC{doc_index}"

    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        for page in PdfReader(file_path).pages:
            text += (page.extract_text() or "") + "\n"
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    raw_chunks = [text[i:i + 1000] for i in range(0, len(text), 800)]

    docs_to_add = []
    for i, content in enumerate(raw_chunks):
        # Sequential ID formatting: DOC1_000, DOC1_001...
        c_id = f"{doc_prefix}_{i:03d}"
        docs_to_add.append(Document(
            page_content=content,
            metadata={"doc_id": doc_prefix, "doc_name": os.path.basename(file_path), "id": c_id}
        ))

    vector_store.add_documents(docs_to_add, ids=[d.metadata["id"] for d in docs_to_add])

    catalog[doc_prefix] = {"name": os.path.basename(file_path), "chunk_count": len(raw_chunks)}
    save_json_cache(CATALOG_PATH, catalog)
    return f"Indexed as {doc_prefix}. Sequential IDs created for look-ahead retrieval."


@tool
def rag_search(query: str, doc_ids: Optional[List[str]] = None):
    """
    Semantic search across documents.
    Provide 'doc_ids' (e.g. ['DOC1', 'DOC2']) to narrow search, otherwise searches all.
    Use this to find starting points or specific facts.
    """
    search_kwargs = {"k": 3}
    if doc_ids:
        search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}

    results = vector_store.similarity_search(query, **search_kwargs)

    final_results = []
    for r in results:
        curr_id = r.metadata["id"]
        summary = get_summary_on_demand(r.page_content, curr_id)
        final_results.append({
            "chunk_id": curr_id,
            "content": r.page_content,
            "neighbor_context": summary,
            "doc_name": r.metadata["doc_name"]
        })
    return final_results


@tool
def fetch_chunks_by_id(chunk_ids: List[str]):
    """
    Directly retrieves specific chunks by their IDs (e.g., ['DOC1_005', 'DOC1_006']).
    Use this for:
    1. Look-ahead: If you have DOC1_004, fetch DOC1_005 to see what's next.
    2. Stability: Reading a continuous section of a document.
    """
    res = vector_store.get(ids=chunk_ids)
    return [{"id": res['ids'][i], "content": res['documents'][i]} for i in range(len(res['ids']))]


def pre_heat_summaries(doc_id: Optional[str] = None):
    where_filter = {"doc_id": doc_id} if doc_id else None
    all_chunks = vector_store.get(where=where_filter)
    total = len(all_chunks['ids'])
    for i in range(total):
        get_summary_on_demand(all_chunks['documents'][i], all_chunks['ids'][i])
        if i % 10 == 0: print(f"Pre-heating {i}/{total}...")