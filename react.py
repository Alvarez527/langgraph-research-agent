from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import ResearchResult

load_dotenv()


@tool
def triple(num: float) -> float:
    """Returns the triple of a number."""
    return num * 3


tools = [TavilySearch(max_results=3), triple]

MODEL = "gpt-4o-mini"

# Reasoning LLM with tools bound (drives the ReAct loop)
llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(tools)

# Separate LLM with structured output for the final answer
structured_llm = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(ResearchResult)
