from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from react import llm, structured_llm, tools
from state import AgentState

load_dotenv()


SYSTEM_MESSAGE = (
    "You are a research assistant. Use the available tools (web search, "
    "calculator) to gather facts before answering. When you have enough "
    "information, produce a concise, factual answer and cite the sources you used."
)

MAX_RETRIES = 2


def run_agent_reasoning(state: AgentState) -> dict:
    """Reasoning step: ask the tool-bound LLM what to do next."""
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)


def validate_output(state: AgentState) -> dict:
    """Convert the conversation into a `ResearchResult`.

    On Pydantic validation failure, increment the retry counter and feed a
    correction prompt back into the conversation so `agent_reasoning` can try again.
    """
    user_query = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "",
    )
    last_answer = state["messages"][-1].content

    formatting_prompt = (
        "Format the agent's final answer as a ResearchResult.\n\n"
        f"User question: {user_query}\n\n"
        f"Agent answer: {last_answer}"
    )

    try:
        result = structured_llm.invoke(formatting_prompt)
        return {"result": result}
    except ValidationError:
        retries = state.get("retries", 0) + 1
        return {
            "retries": retries,
            "messages": [
                HumanMessage(
                    content=(
                        "Your previous answer could not be parsed into the required "
                        "ResearchResult schema (summary, sources, confidence). "
                        "Please refine the answer with explicit sources and a "
                        "confidence score between 0 and 1."
                    )
                )
            ],
        }
