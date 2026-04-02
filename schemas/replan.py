from pydantic import BaseModel, Field
from typing import List, Literal


class ReplanSchema(BaseModel):
    reasoning: str = Field(
        ...,
        description="Short reasoning about why more tools are needed or why we are done."
    )

    agent_calls: List[Literal["rag", "web", "memory"]] = Field(
        default_factory=list,
        description="Tools to call next. Empty list means no more tools needed."
    )

    done: bool = Field(
        ...,
        description="True if the system has enough information to generate final answer."
    )