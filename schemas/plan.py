from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AgentCall(BaseModel):
    tool: str = Field(description="The tool to call: 'rag', 'web', or 'memory'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0, le=1)
    reasoning: Optional[str] = Field(description="Brief reason for this tool's score")


class PlanSchema(BaseModel):
    thought: str
    agent_calls: List[AgentCall] = Field(default_factory=list)