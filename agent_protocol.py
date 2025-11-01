from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json
import time
import uuid


@dataclass
class Capability:
    name: str
    rationale: Optional[str] = None
    # optional: fine-grained scopes or policies
    scopes: Optional[List[str]] = None


@dataclass
class AgentOutput:
    type: str  # e.g., "capability_request", "message"
    id: str
    timestamp: float
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def capability_request(capabilities: List[Capability], request_id: Optional[str] = None) -> AgentOutput:
    rid = request_id or str(uuid.uuid4())
    return AgentOutput(
        type="capability_request",
        id=rid,
        timestamp=time.time(),
        payload={
            "capabilities": [asdict(c) for c in capabilities],
        },
    )


def wants_access_to_screen_and_input(rationale: Optional[str] = None) -> AgentOutput:
    caps = [
        Capability(name="screen.read", rationale=rationale or "KroxAI wants to analyze on-screen content to ground responses."),
        Capability(name="input.mouse", rationale=rationale or "KroxAI needs to interact with UI elements for tasks."),
        Capability(name="input.keyboard", rationale=rationale or "KroxAI needs to type text or shortcuts to complete actions."),
    ]
    return capability_request(caps)
