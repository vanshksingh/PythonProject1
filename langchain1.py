from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver


# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = """
You are a simple math assistant.

You have access to:
- multiply_numbers: multiply two numbers

If the user asks for multiplication, use the tool.
"""


# =========================
# Response Schema
# =========================
@dataclass
class ResponseFormat:
    answer: str
    tool_used: Optional[str] = None


# =========================
# Tools
# =========================
@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# =========================
# Model Factory
# =========================
def create_model():
    """
    Requires Ollama running locally:
    ollama pull llama3
    ollama serve
    """
    return init_chat_model(
        "ollama:mistral:7b-instruct",  # change to any tool-calling capable model
        temperature=0
    )


# =========================
# Agent Factory
# =========================
def create_math_agent():
    model = create_model()
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[multiply_numbers],
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer
    )

    return agent


# =========================
# Runner
# =========================
def run_example():
    agent = create_math_agent()

    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What is 6 times 7?"}
            ]
        },
        config=config
    )

    print(response["structured_response"])


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    run_example()