from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class PlanSchema(BaseModel):
    thought: str
    agent_calls: List[Dict[str, Any]] = []