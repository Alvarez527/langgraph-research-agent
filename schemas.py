from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A source cited in the research result."""

    url: str = Field(description="URL of the source")
    title: str = Field(description="Title or short description of the source")


class ResearchResult(BaseModel):
    """Structured final output produced by the research agent."""

    summary: str = Field(description="Concise answer to the user's question")
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources cited in the summary",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence in the answer (0.0 - 1.0)",
    )
