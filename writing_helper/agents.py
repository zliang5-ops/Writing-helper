import json
import uuid
from typing import Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .constants import MAX_REASON_OPTIONS, TARGET_REASON_OPTIONS
from .models import (
    InterpreterReasonCandidate,
    InterpreterResult,
    ProfileUpdateSuggestion,
    ReplacementGuidance,
    ReplacementOption,
    SessionState,
)
from .text_utils import extract_json_object


class BaseLocalAgent:
    def __init__(self, name: str):
        self.name = name


class StatelessLLMAgent(BaseLocalAgent):
    def __init__(self, name: str, model: str, system_message: str):
        super().__init__(name)
        self.model_client = OpenAIChatCompletionClient(model=model)
        self.agent = AssistantAgent(
            name=name,
            model_client=self.model_client,
            model_client_stream=True,
            system_message=system_message,
        )

    async def complete(self, task: str) -> str:
        parts: List[str] = []
        async for item in self.agent.run_stream(task=task):
            text = getattr(item, "content", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts).strip()

    async def close(self) -> None:
        await self.model_client.close()


class PreferenceMemoryAgent(BaseLocalAgent):
    def __init__(self, name: str = "preference_memory_agent"):
        super().__init__(name)

    def update_profile(self, existing_profile: List[str], summary: str) -> List[str]:
        cleaned = summary.strip()
        if not cleaned:
            return list(existing_profile)
        return list(dict.fromkeys(existing_profile + [cleaned]))


class InterruptionInterpreterAgent(StatelessLLMAgent):
    def __init__(self, model: str = "gpt-4o-mini", name: str = "interruption_interpreter_agent"):
        super().__init__(
            name=name,
            model=model,
            system_message=(
                "You interpret why a user interrupted generated writing. "
                "Return only valid JSON in the requested structure."
            ),
        )

    async def interpret(self, state: SessionState) -> InterpreterResult:
        context = state.interruption_context
        preferences = "\n".join(f"- {item}" for item in state.preference_profile) or "- None yet."
        prompt = f"""
Interpret this writing interruption and return exactly the requested JSON structure with no extra keys.

User task description:
{state.task}

Existing user profile:
{preferences}

Stop point information:
{{
  "termination_point": "{context.termination_point}",
  "last_sentence": "{context.last_sentence}",
  "current_sentence": "{context.current_sentence}"
}}

Return exactly this structure:
{{
  "stop_point": {{
    "termination_point": "<exact point where user interrupted>",
    "last_sentence": "<last completed sentence>",
    "current_sentence": "<sentence active at interruption>"
  }},
  "likely_user_intent": "<general description of what the user probably wants>",
  "reason_candidates": [
    {{
      "id": "R1",
      "reason": "<detailed possible reason>"
    }},
    {{
      "id": "R2",
      "reason": "<detailed possible reason>"
    }}
  ],
  "replacement_guidance": {{
    "goal": "<what the replacement-generation agent should generate>",
    "desired_properties": ["<p1>", "<p2>"],
    "avoid": ["<a1>", "<a2>"]
  }},
  "profile_update": {{
    "preference_summary": "<possible user preference inferred from this interruption>",
    "confidence": 0.0
  }}
}}

Constraints:
- Use at most {MAX_REASON_OPTIONS} reason candidates.
- Prefer around {TARGET_REASON_OPTIONS} reason candidates.
- Keep reason candidates detailed enough to guide rewriting.
- Do not include any keys beyond the required structure.
- Return JSON only.
"""
        try:
            return self._parse_result(extract_json_object(await self.complete(prompt)))
        except Exception:
            return self._fallback_interpretation(state)

    def _parse_result(self, payload: dict) -> InterpreterResult:
        stop_point = payload.get("stop_point", {})
        reasons = payload.get("reason_candidates", [])[:MAX_REASON_OPTIONS]
        parsed_reasons = [
            InterpreterReasonCandidate(
                id=str(item.get("id", f"R{index + 1}")).strip() or f"R{index + 1}",
                reason=str(item.get("reason", "")).strip(),
            )
            for index, item in enumerate(reasons)
            if str(item.get("reason", "")).strip()
        ]
        guidance = payload.get("replacement_guidance", {})
        profile_update = payload.get("profile_update", {})
        return InterpreterResult(
            stop_point=stateful_stop_point(
                termination_point=str(stop_point.get("termination_point", "")).strip(),
                last_sentence=str(stop_point.get("last_sentence", "")).strip(),
                current_sentence=str(stop_point.get("current_sentence", "")).strip(),
            ),
            likely_user_intent=str(payload.get("likely_user_intent", "")).strip(),
            reason_candidates=parsed_reasons,
            replacement_guidance=ReplacementGuidance(
                goal=str(guidance.get("goal", "")).strip(),
                desired_properties=[str(item).strip() for item in guidance.get("desired_properties", []) if str(item).strip()],
                avoid=[str(item).strip() for item in guidance.get("avoid", []) if str(item).strip()],
            ),
            profile_update=ProfileUpdateSuggestion(
                preference_summary=str(profile_update.get("preference_summary", "")).strip(),
                confidence=float(profile_update.get("confidence", 0.0) or 0.0),
            ),
        )

    def _fallback_interpretation(self, state: SessionState) -> InterpreterResult:
        context = state.interruption_context
        reasons: List[InterpreterReasonCandidate] = []

        def add(reason: str) -> None:
            if len(reasons) >= TARGET_REASON_OPTIONS:
                return
            reason_id = f"R{len(reasons) + 1}"
            reasons.append(InterpreterReasonCandidate(id=reason_id, reason=reason))

        current_lower = context.current_sentence.lower()
        task_lower = state.task.lower()
        if any(word in current_lower for word in ["moreover", "furthermore", "therefore", "indeed"]):
            add("The sentence may sound too formal or too stiff for the user's intended voice.")
        if len(context.current_sentence.split()) > 20:
            add("The sentence may be too long or too dense for the pacing the user wants.")
        if "academic" in task_lower:
            add("The sentence may not sound rigorous or academic enough for the task.")
        add("The sentence may be too generic and may need more concrete or targeted wording.")
        add("The sentence may not align tightly enough with the user's likely intent.")

        return InterpreterResult(
            stop_point=state.interruption_context,
            likely_user_intent=state.task.strip() or "The user wants writing that better matches the stated task.",
            reason_candidates=reasons,
            replacement_guidance=ReplacementGuidance(
                goal="Rewrite the interrupted sentence so it better matches the user's likely intent.",
                desired_properties=[
                    "sentence-level replacement",
                    "closer alignment with the task",
                    "more natural fit for the user profile",
                ],
                avoid=[
                    "generic filler",
                    "drifting off task",
                ],
            ),
            profile_update=ProfileUpdateSuggestion(
                preference_summary="Prefer sentence rewrites that stay tightly aligned with the task and feel more intentional.",
                confidence=0.45,
            ),
        )


def stateful_stop_point(termination_point: str, last_sentence: str, current_sentence: str):
    from .models import InterruptionContext

    return InterruptionContext(
        termination_point=termination_point,
        last_sentence=last_sentence,
        current_sentence=current_sentence,
        replacement_start=0,
    )


class BehaviorInterpreterAgent(InterruptionInterpreterAgent):
    async def interpret_behavior(
        self,
        state: SessionState,
        behavior_text: str,
        behavior_mode: str,
    ) -> InterpreterResult:
        context = state.interruption_context
        preferences = "\n".join(f"- {item}" for item in state.preference_profile) or "- None yet."
        prompt = f"""
Interpret this user behavior and return exactly the requested JSON structure with no extra keys.

Behavior mode:
{behavior_mode}

User task description:
{state.task}

Existing user profile:
{preferences}

Stop point information:
{{
  "termination_point": "{context.termination_point}",
  "last_sentence": "{context.last_sentence}",
  "current_sentence": "{context.current_sentence}"
}}

User behavior text:
{behavior_text}

Return exactly this structure:
{{
  "stop_point": {{
    "termination_point": "<exact point where user interrupted>",
    "last_sentence": "<last completed sentence>",
    "current_sentence": "<sentence active at interruption>"
  }},
  "likely_user_intent": "<general description of what the user probably wants>",
  "reason_candidates": [
    {{
      "id": "R1",
      "reason": "<detailed possible reason>"
    }},
    {{
      "id": "R2",
      "reason": "<detailed possible reason>"
    }}
  ],
  "replacement_guidance": {{
    "goal": "<what the replacement-generation agent should generate>",
    "desired_properties": ["<p1>", "<p2>"],
    "avoid": ["<a1>", "<a2>"]
  }},
  "profile_update": {{
    "preference_summary": "<possible user preference inferred from this interruption>",
    "confidence": 0.0
  }}
}}

Return JSON only.
"""
        try:
            return self._parse_result(extract_json_object(await self.complete(prompt)))
        except Exception:
            return InterpreterResult(
                stop_point=state.interruption_context,
                likely_user_intent=state.task,
                reason_candidates=[
                    InterpreterReasonCandidate(id="R1", reason="The user wants a more specific local revision than the offered options."),
                    InterpreterReasonCandidate(id="R2", reason="The user is showing a direct writing preference through manual feedback."),
                ],
                replacement_guidance=ReplacementGuidance(
                    goal="Produce a sentence-level revision that follows the user's direct behavior.",
                    desired_properties=["follow the user's explicit revision preference", "stay aligned with the task"],
                    avoid=["ignoring the user's explicit intent"],
                ),
                profile_update=ProfileUpdateSuggestion(
                    preference_summary=f"User often prefers: {behavior_text.strip()}",
                    confidence=0.6,
                ),
            )


class ReplacementAgent(StatelessLLMAgent):
    def __init__(self, model: str = "gpt-4o-mini", name: str = "replacement_agent"):
        super().__init__(
            name=name,
            model=model,
            system_message=(
                "You generate replacement options for an interrupted sentence. "
                "Return only valid JSON."
            ),
        )

    async def build_replacements(
        self,
        state: SessionState,
        interpreter_result: InterpreterResult,
    ) -> List[ReplacementOption]:
        prompt = f"""
Generate one replacement option for each reason candidate.

User task:
{state.task}

Sentence to revise:
{state.interruption_context.current_sentence}

Interpreter result:
{json.dumps(interpreter_result.to_dict(), ensure_ascii=False, indent=2)}

Return JSON with this structure:
{{
  "options": [
    {{
      "reason_id": "R1",
      "reason": "<copy of the reason>",
      "explanation": "<why this option fits>",
      "replacement_text": "<replacement sentence>"
    }}
  ]
}}

Return JSON only.
"""
        try:
            payload = extract_json_object(await self.complete(prompt))
            options: List[ReplacementOption] = []
            reason_map = {item.id: item.reason for item in interpreter_result.reason_candidates}
            for item in payload.get("options", []):
                reason_id = str(item.get("reason_id", "")).strip()
                replacement_text = str(item.get("replacement_text", "")).strip()
                if reason_id and replacement_text and reason_id in reason_map:
                    options.append(
                        ReplacementOption(
                            option_id=str(uuid.uuid4()),
                            reason_id=reason_id,
                            reason=reason_map[reason_id],
                            explanation=str(item.get("explanation", "")).strip() or reason_map[reason_id],
                            replacement_text=replacement_text,
                        )
                    )
            if options:
                return options
        except Exception:
            pass
        return self._fallback_replacements(state, interpreter_result)

    async def build_custom_revision(
        self,
        task: str,
        passage: str,
        custom_instruction: str,
    ) -> str:
        prompt = f"""
Rewrite the interrupted sentence based on the user's custom revision request.

Task:
{task}

Current passage:
{passage}

User revision request:
{custom_instruction}

Return only the replacement sentence or short local revision.
"""
        try:
            text = await self.complete(prompt)
            if text:
                return text.strip()
        except Exception:
            pass
        return custom_instruction.strip()

    def _fallback_replacements(self, state: SessionState, interpreter_result: InterpreterResult) -> List[ReplacementOption]:
        sentence = state.interruption_context.current_sentence.strip() or "Continue with a clearer sentence."
        options: List[ReplacementOption] = []
        for reason in interpreter_result.reason_candidates:
            options.append(
                ReplacementOption(
                    option_id=str(uuid.uuid4()),
                    reason_id=reason.id,
                    reason=reason.reason,
                    explanation=reason.reason,
                    replacement_text=sentence,
                )
            )
        return options


class StreamingWriterAgent(StatelessLLMAgent):
    def __init__(self, model: str = "gpt-4o-mini", name: str = "streaming_writer_agent"):
        super().__init__(
            name=name,
            model=model,
            system_message=(
                "You are the main writing generator in an interruption-aware writing system. "
                "Generate streaming prose that follows the user's task and saved profile. "
                "Do not explain your process."
            ),
        )

    async def stream_generate(
        self,
        state: SessionState,
        on_token: Callable[[str], None],
        should_stop: Callable[[], bool],
    ) -> str:
        prompt = self._build_prompt(state)
        accumulated = ""
        async for item in self.agent.run_stream(task=prompt):
            if should_stop():
                break
            text = getattr(item, "content", None)
            if isinstance(text, str) and text:
                accumulated += text
                on_token(text)
        return accumulated

    def _build_prompt(self, state: SessionState) -> str:
        preferences = "\n".join(f"- {item}" for item in state.preference_profile) or "- None yet."
        revision_history = state.format_revision_history()
        return f"""
Username:
{state.username}

User task description:
{state.task}

Saved user profile:
{preferences}

Interruption history:
{revision_history}

Current accepted text:
{state.accepted_text}

Current live text:
{state.live_text}

Instruction:
Continue the writing as streaming prose based on the user's description of task and saved profile.
"""
