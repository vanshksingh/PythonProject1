import os
import hashlib
from typing import List, Annotated
from pypdf import PdfReader

# LangChain & Ollama Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain1.tools import tool
# New / Correct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from deepagents import create_deep_agent
from langchain_chroma import Chroma


# =========================
# 1. DOCUMENT STORE UTILITY
# =========================

class DocStore:
    def __init__(self):
        # Using Nomic for high-quality local embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.persist_directory = "./chroma_store"
        self.collection_name = "agentic_doc_store"
        self.vectorstore = None

    def get_vs(self):
        if not self.vectorstore:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self.vectorstore

    def ingest(self, path: str):
        print(f"📄 Loading: {path}")
        reader = PdfReader(path)
        text = "\n".join([p.extract_text() for p in reader.pages])
        file_hash = hashlib.sha256(text.encode()).hexdigest()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(text)

        metadatas = [
            {"file_hash": file_hash, "chunk_id": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        vs = self.get_vs()
        vs.add_texts(texts=chunks, metadatas=metadatas)
        print(f"✅ Ingested {len(chunks)} chunks. Hash: {file_hash[:8]}")
        return file_hash


# Global instance for tool access
doc_store = DocStore()


# =========================
# 2. AGENT TOOLS
# =========================

@tool
def vector_search(query: str):
    """
    Search the document for semantically relevant chunks.
    Use this first if you don't know where the answer is.
    """
    vs = doc_store.get_vs()
    docs = vs.similarity_search(query, k=4)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


@tool
def get_neighboring_chunk(chunk_id: int, file_hash: str, direction: str):
    """
    Retrieve the chunk immediately before or after a known chunk.
    'direction' must be 'prev' or 'next'.
    """
    target_id = chunk_id - 1 if direction == "prev" else chunk_id + 1
    vs = doc_store.get_vs()

    # Simple metadata filtering via Chroma
    results = vs.get(where={
        "$and": [
            {"chunk_id": {"$eq": target_id}},
            {"file_hash": {"$eq": file_hash}}
        ]
    })

    if results["documents"]:
        return {
            "content": results["documents"][0],
            "metadata": results["metadatas"][0]
        }
    return f"No {direction} chunk found for ID {chunk_id}."


# =========================
# 3. AGENT CONFIGURATION
# =========================

# Initialize the Mistral model via Ollama
# Mistral 7B v0.3+ is excellent at following tool-calling instructions
llm = ChatOllama(
    model="mistral:7b-instruct",
    temperature=0,
)

system_prompt = """
You are a 'Deep Document Agent' powered by LangChain. 
Your goal is to answer questions using the provided tools to browse a local document database.

GUIDELINES:
1. Always start with 'vector_search' to find relevant keywords/concepts.
2. If the context in a chunk feels cut off, use 'get_neighboring_chunk' to read the 'next' or 'prev' sections.
3. If multiple chunks are returned, synthesize them into a coherent answer.
4. Be precise. If the document doesn't mention a topic, say you don't know.
"""

# Create the Deep Agent
# This replaces your manual conversation loop
agent = create_deep_agent(
    name="DocInspector",
    model=llm,
    tools=[vector_search, get_neighboring_chunk],
    system_prompt=system_prompt,
)


# =========================
# 4. EXECUTION LOOP
# =========================

def run_chat():
    # Update this path to your local project file
    file_path = "/Users/vks/Downloads/Major Project/01 Major project guidelines for students.pdf"

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    # Ingest the file
    file_hash = doc_store.ingest(file_path)

    print("\n🚀 DeepAgent is online. Ask about your Project Guidelines (type 'exit' to quit).")

    # Initialize an empty list for message history
    messages = []

    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        # Add user query to history
        messages.append({"role": "user", "content": query})

        # The agent.invoke handles the internal reasoning steps automatically
        # It will call tools as many times as needed before responding
        try:
            result = agent.invoke({"messages": messages})

            # The last message in the state is the agent's final response
            response_msg = result["messages"][-1]
            print(f"\n🤖 Agent: {response_msg.content}")

            # Update history for multi-turn conversation
            messages = result["messages"]

        except Exception as e:
            print(f"⚠️ An error occurred: {e}")


if __name__ == "__main__":
    run_chat()