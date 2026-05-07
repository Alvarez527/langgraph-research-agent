from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from schemas import ResearchResult


class AgentState(TypedDict):
    """State carried through the research agent graph."""

    messages: Annotated[list, add_messages]
    result: Optional[ResearchResult]
    retries: int
