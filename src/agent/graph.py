"""LangGraph single-node graph template — customized to call an OpenAI model."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

# ---- Context (optional configuration) ----
class Context(TypedDict):
    """Optional context parameters for the agent."""
    my_configurable_param: str


# ---- State definition ----
@dataclass
class State:
    """The state that moves through the graph."""
    user_input: str = ""
    text: str = ""      
    response: str = ""


# ---- Model setup ----
model = ChatOpenAI(model="gpt-4o-mini")  # You can change this to "gpt-4o" or "gpt-3.5-turbo"


# ---- Node logic ----
async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    user_message = getattr(state, "user_input", "") or getattr(state, "text", "") or "Hello!"
    system_prompt = "Always respond in 3 lines or less. Be concise and direct."
    final_prompt = f"{system_prompt}\n\nUser: {user_message}"
    response = await model.ainvoke(final_prompt)
    return {"response": response.content}


# ---- Define the graph ----
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .add_edge("call_model", "__end__")
    .compile(name="AI Agent Graph")
)