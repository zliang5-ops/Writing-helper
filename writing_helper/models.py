from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InterruptionContext:
    termination_point: str = ""
    last_sentence: str = ""
    current_sentence: str = ""
    replacement_start: int = 0


@dataclass
class InterpreterReasonCandidate:
    id: str
    reason: str


@dataclass
class ReplacementGuidance:
    goal: str
    desired_properties: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)


@dataclass
class ProfileUpdateSuggestion:
    preference_summary: str
    confidence: float


@dataclass
class InterpreterResult:
    stop_point: InterruptionContext
    likely_user_intent: str
    reason_candidates: List[InterpreterReasonCandidate] = field(default_factory=list)
    replacement_guidance: ReplacementGuidance = field(default_factory=lambda: ReplacementGuidance(goal=""))
    profile_update: ProfileUpdateSuggestion = field(
        default_factory=lambda: ProfileUpdateSuggestion(preference_summary="", confidence=0.0)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_point": {
                "termination_point": self.stop_point.termination_point,
                "last_sentence": self.stop_point.last_sentence,
                "current_sentence": self.stop_point.current_sentence,
            },
            "likely_user_intent": self.likely_user_intent,
            "reason_candidates": [asdict(item) for item in self.reason_candidates],
            "replacement_guidance": asdict(self.replacement_guidance),
            "profile_update": asdict(self.profile_update),
        }


@dataclass
class ReplacementOption:
    option_id: str
    reason_id: str
    reason: str
    explanation: str
    replacement_text: str
    option_kind: str = "reason"


@dataclass
class RevisionEvent:
    event_id: str
    timestamp: float
    username: str
    task: str
    stop_point: InterruptionContext
    interpreter_result: Dict[str, Any]
    selected_reason_id: str
    selected_reason: str
    selected_revision: str
    selection_kind: str
    custom_input: str
    updated_preference_profile: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    username: str
    preference_profile: List[str] = field(default_factory=list)
    revision_log: List[RevisionEvent] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class SessionState:
    username: str = ""
    task: str = ""
    live_text: str = ""
    accepted_text: str = ""
    preference_profile: List[str] = field(default_factory=list)
    replacement_options: List[ReplacementOption] = field(default_factory=list)
    interruption_context: InterruptionContext = field(default_factory=InterruptionContext)
    active_interpreter_result: Optional[InterpreterResult] = None
    revision_log: List[RevisionEvent] = field(default_factory=list)
    user_guidance_shown: bool = False

    def context_snapshot(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "task": self.task,
            "accepted_text": self.accepted_text,
            "live_text": self.live_text,
            "preference_profile": list(self.preference_profile),
            "stop_point": {
                "termination_point": self.interruption_context.termination_point,
                "last_sentence": self.interruption_context.last_sentence,
                "current_sentence": self.interruption_context.current_sentence,
            },
            "revision_count": len(self.revision_log),
        }

    def format_revision_history(self, limit: int = 6) -> str:
        if not self.revision_log:
            return "- None yet."

        lines: List[str] = []
        for index, event in enumerate(self.revision_log[-limit:], start=1):
            reasons = event.interpreter_result.get("reason_candidates", [])
            reason_text = ", ".join(f"{item.get('id')}: {item.get('reason')}" for item in reasons) or "None recorded"
            lines.append(
                f"{index}. Stop sentence: {event.stop_point.current_sentence or '[None]'}\n"
                f"   Last sentence: {event.stop_point.last_sentence or '[None]'}\n"
                f"   Potential reasons: {reason_text}\n"
                f"   Selected reason: {event.selected_reason or '[Pending]'}\n"
                f"   Selected revision: {event.selected_revision or '[Pending]'}\n"
                f"   Selection kind: {event.selection_kind}\n"
                f"   Custom input: {event.custom_input or '[None]'}"
            )
        return "\n".join(lines)
