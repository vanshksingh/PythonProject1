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

# Initialize directories
os.makedirs(CACHE_PATH, exist_ok=True)

# 1. Initialize Tools
embeddings = OllamaEmbeddings(model=MAIN_MODEL)
summarizer_llm = OllamaLLM(model=SUMMARY_MODEL)
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

SUMMARY_CACHE_PATH = os.path.join(CACHE_PATH, "chunk_summaries.json")
CATALOG_PATH = os.path.join(CACHE_PATH, "catalog.json")


def load_json_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


def save_json_cache(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_summary_on_demand(chunk_id: str, content: Optional[str] = None) -> str:
    """Retrieves from cache or generates a summary for a specific ID."""
    cache = load_json_cache(SUMMARY_CACHE_PATH)
    if chunk_id in cache:
        return cache[chunk_id]

    if content is None:
        # If content isn't provided, fetch it from DB to summarize
        res = vector_store.get(ids=[chunk_id])
        if not res['documents']: return "[No Context]"
        content = res['documents'][0]

    prompt = f"Summarize this text in one sentence for context: {content[:600]}"
    summary = summarizer_llm.invoke(prompt).strip()

    cache[chunk_id] = summary
    save_json_cache(SUMMARY_CACHE_PATH, cache)
    return summary


def assemble_chunk_with_context(chunk_id: str, content: str) -> str:
    """Constructs the final string with Prev Summary + Content + Next Summary."""
    # Split ID like DOC1_005 into ('DOC1', 5)
    parts = chunk_id.split('_')
    prefix = parts[0]
    idx = int(parts[1])

    prev_id = f"{prefix}_{str(idx - 1).zfill(3)}"
    next_id = f"{prefix}_{str(idx + 1).zfill(3)}"

    prev_sum = get_summary_on_demand(prev_id) if idx > 0 else "START OF DOCUMENT"
    next_sum = get_summary_on_demand(next_id)

    return (
        f"--- CONTEXT WINDOW: {chunk_id} ---\n"
        f"PREVIOUS SECTION SUMMARY: {prev_sum}\n\n"
        f"ACTIVE CONTENT:\n{content}\n\n"
        f"FOLLOWING SECTION SUMMARY: {next_sum}\n"
        f"--- END OF WINDOW ---"
    )


# --- AGENT TOOLS ---

@tool
def list_available_documents():
    """Returns indexed documents with Serial IDs (DOC1, DOC2) and names."""
    catalog = load_json_cache(CATALOG_PATH)
    if not catalog: return "No documents found."
    return "\n".join([f"- {v['serial_id']}: {v['name']} ({v['chunk_count']} chunks)" for k, v in catalog.items()])


@tool
def index_new_document(file_path: str):
    """Indexes a file and assigns a Serial ID (DOC1, DOC2...)."""
    file_path = file_path.strip("'").strip('"')
    if not os.path.exists(file_path): return "File not found."

    file_hash = get_file_hash(file_path)
    catalog = load_json_cache(CATALOG_PATH)
    if file_hash in catalog: return f"Already indexed as {catalog[file_hash]['serial_id']}."

    serial_id = f"DOC{len(catalog) + 1}"
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
        c_id = f"{serial_id}_{str(i).zfill(3)}"
        docs_to_add.append(Document(
            page_content=content,
            metadata={"doc_id": serial_id, "id": c_id, "doc_name": os.path.basename(file_path)}
        ))

    vector_store.add_documents(docs_to_add, ids=[d.metadata["id"] for d in docs_to_add])
    catalog[file_hash] = {"name": os.path.basename(file_path), "serial_id": serial_id, "chunk_count": len(raw_chunks)}
    save_json_cache(CATALOG_PATH, catalog)
    return f"Indexed as {serial_id}. {len(raw_chunks)} chunks added."


@tool
def rag_search(query: str, doc_id: Optional[str] = None):
    """
    Search across docs. Returns chunks with their immediate neighbors' summaries
    automatically appended for context stability.
    """
    search_kwargs = {"k": 3}
    if doc_id: search_kwargs["filter"] = {"doc_id": doc_id}

    results = vector_store.similarity_search(query, **search_kwargs)
    final_results = []
    for r in results:
        c_id = r.metadata["id"]
        # Wrap content with neighbor summaries
        full_context = assemble_chunk_with_context(c_id, r.page_content)
        final_results.append({"chunk_id": c_id, "text": full_context})
    return final_results


@tool
def fetch_chunks_by_id(chunk_ids: List[str]):
    """
    Fetches specific chunks by ID (e.g. ['DOC1_005']).
    Includes previous and next summaries automatically to maintain process stability.
    """
    res = vector_store.get(ids=chunk_ids)
    final_output = []
    for i in range(len(res['ids'])):
        c_id = res['ids'][i]
        content = res['documents'][i]
        full_context = assemble_chunk_with_context(c_id, content)
        final_output.append({"id": c_id, "text": full_context})
    return final_output


# --- PRE-HEAT ---

def pre_heat_summaries(serial_id: Optional[str] = None):
    """Generates all missing summaries for a doc (or all docs) and caches them."""
    print(f"🔥 Pre-heating {serial_id if serial_id else 'All'}...")
    where_filter = {"doc_id": serial_id} if serial_id else None
    all_chunks = vector_store.get(where=where_filter)
    total = len(all_chunks['ids'])
    for i in range(total):
        get_summary_on_demand(all_chunks['ids'][i], all_chunks['documents'][i])
        if i % 10 == 0: print(f"Progress: {i}/{total}")
    print("✅ Complete.")