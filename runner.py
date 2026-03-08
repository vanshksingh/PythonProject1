import uuid
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
# External
from rag_ret import list_available_documents, rag_search, fetch_chunks_by_id, index_new_document
from config  import MAIN_MODEL, TEMPERATURE



# 1. Define Context Schema
@dataclass
class Context:
    user_id: str

# 2. Define Tools with Runtime Context
#defined in rag_ret.py

# 3. Configure Model (gpt-oss via Ollama)
#Chane in Config.py

model = ChatOllama(
    model=MAIN_MODEL,
    temperature=TEMPERATURE,
)

# 4. System Prompt
SYSTEM_PROMPT = (
    "You are a local RAG assistant. "
    "CRITICAL: When you need to use a tool, output ONLY the tool call. "
    "Do not explain your reasoning or 'think out loud' before the JSON. "
    "Use 'list_available_documents' to see what you have access to."
    "You are a researcher with a sequential RAG system.\n"
    "1. Use 'list_available_documents' to find DOC_IDs.\n"
    "2. Use 'rag_search' to find where a topic starts.\n"
    "3. Chunks are sequential (e.g., DOC1_001 is followed by DOC1_002).\n"
    "If you find a relevant chunk, use 'fetch_chunks_by_id' to read the next 2-3 chunks "
    "to ensure you haven't missed the full context of a paragraph or section."
)

# 5. Set up Memory and Agent
memory = InMemorySaver()
tools = [ list_available_documents, rag_search, fetch_chunks_by_id, index_new_document ]



# create_agent is the standard LangGraph way to create a tool-calling loop
agent_executor = create_agent(
    model,
    tools,
    checkpointer=memory
)

def start_chat():
    # Generate a unique thread ID for this specific session
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": "1" # This mimics your context logic
        }
    }

    print("--- 🌦️ Weather Pun-dit 3000 Online ---")
    print("(Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Pun-dit: See ya on the 'sunny side'!")
            break

        # Invoke the agent
        # The agent maintains state via the thread_id in config
        result = agent_executor.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        # The last message in the list is the AI's final response
        final_response = result["messages"][-1].content
        print(f"\nPun-dit: {final_response}\n")

if __name__ == "__main__":
    start_chat()