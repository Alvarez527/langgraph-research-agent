
from dotenv import load_dotenv
import os
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()


SYSTEM_MESSAGE = "You are a helpful assistant that can perform various tasks using tools. Always use the tools when necessary to provide accurate and efficient responses to the user's queries."

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning state
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)


