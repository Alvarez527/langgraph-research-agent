from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes import MAX_RETRIES, run_agent_reasoning, tool_node, validate_output
from state import AgentState

load_dotenv()


AGENT_REASON = "agent_reasoning"
ACT = "act"
VALIDATE = "validate"
LAST = -1


def should_continue(state: AgentState) -> str:
    """After reasoning, route to tool execution or to validation."""
    if state["messages"][LAST].tool_calls:
        return ACT
    return VALIDATE


def after_validate(state: AgentState) -> str:
    """After validation, finish if we have a valid result, retry otherwise."""
    if state.get("result") is not None:
        return END
    if state.get("retries", 0) >= MAX_RETRIES:
        return END
    return AGENT_REASON


flow = StateGraph(AgentState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)
flow.add_node(VALIDATE, validate_output)

flow.set_entry_point(AGENT_REASON)
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {ACT: ACT, VALIDATE: VALIDATE},
)
flow.add_edge(ACT, AGENT_REASON)
flow.add_conditional_edges(
    VALIDATE,
    after_validate,
    {AGENT_REASON: AGENT_REASON, END: END},
)

memory = MemorySaver()
app = flow.compile(checkpointer=memory)
app.get_graph().draw_mermaid_png(output_file_path="agent_flow.png")


def _print_result(state: dict) -> None:
    result = state.get("result")
    if result is not None:
        print(result.model_dump_json(indent=2))
    else:
        print(state["messages"][LAST].content)


if __name__ == "__main__":
    print("LangGraph Research Agent — OpenAI + Structured Output + Checkpointing\n")

    config = {"configurable": {"thread_id": "demo-1"}}

    print("=== First turn ===")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the weather in Tokyo right now? Then triple the temperature."
                )
            ],
            "retries": 0,
        },
        config=config,
    )
    _print_result(res)

    print("\n=== Resuming the same thread (checkpointed) ===")
    res2 = app.invoke(
        {
            "messages": [
                HumanMessage(content="And what was the weather I asked about? Restate it briefly.")
            ],
        },
        config=config,
    )
    _print_result(res2)
