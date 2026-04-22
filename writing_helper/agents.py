import asyncio
import json
import uuid
from typing import Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .constants import (
    MAX_REPLACEMENT_WORDS,
    MAX_REASON_OPTIONS,
    PROFILE_MEMORY_TEMPERATURE,
    REPLACEMENT_TEMPERATURE,
    STREAM_TOKEN_DELAY_SECONDS,
    STREAMING_TEMPERATURE,
    TARGET_REASON_OPTIONS,
)
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
    def __init__(self, name: str, model: str, system_message: str, temperature: float | None = None):
        super().__init__(name)
        client_kwargs = {"model": model}
        if temperature is not None:
            client_kwargs["temperature"] = temperature
        self.model_client = OpenAIChatCompletionClient(**client_kwargs)
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
    def __init__(self, model: str = "gpt-4o-mini", name: str = "preference_memory_agent"):
        super().__init__(name)
        self.model_client = OpenAIChatCompletionClient(model=model, temperature=PROFILE_MEMORY_TEMPERATURE)
        self.agent = AssistantAgent(
            name=name,
            model_client=self.model_client,
            model_client_stream=True,
            system_message=(
                "You write short reusable user writing preferences from revision choices. "
                "Return only one concise preference sentence."
            ),
        )

    def update_profile(self, existing_profile: List[str], summary: str) -> List[str]:
        cleaned = summary.strip()
        if not cleaned:
            return list(existing_profile)
        return list(dict.fromkeys(existing_profile + [cleaned]))

    async def summarize_choice(
        self,
        task: str,
        current_sentence: str,
        selected_reason: str,
        selected_revision: str,
        existing_profile: List[str],
    ) -> str:
        preferences = "\n".join(f"- {item}" for item in existing_profile) or "- None yet."
        prompt = f"""
Write one concise reusable user preference based on the chosen revision.

Task:
{task}

Current interrupted sentence:
{current_sentence}

Selected reason:
{selected_reason}

Selected revision:
{selected_revision}

Existing user profile:
{preferences}

Constraints:
- Return exactly one sentence.
- Describe a durable writing preference, not a one-off edit.
- Keep it under 18 words.
- Do not mention JSON, tasks, or specific quoted text unless necessary.
- Return plain text only.
"""
        try:
            parts: List[str] = []
            async for item in self.agent.run_stream(task=prompt):
                text = getattr(item, "content", None)
                if isinstance(text, str) and text:
                    parts.append(text)
            summary = " ".join("".join(parts).split()).strip()
            return summary.strip("\"' ")
        except Exception:
            return self._fallback_summary(selected_reason)

    async def close(self) -> None:
        await self.model_client.close()

    def _fallback_summary(self, selected_reason: str) -> str:
        lowered = selected_reason.lower()
        if any(word in lowered for word in ["generic", "concrete", "detail"]):
            return "Prefers concrete writing with sharper detail."
        if any(word in lowered for word in ["specific", "narrow", "overcommit"]):
            return "Prefers ideas that stay flexible before narrowing."
        if any(word in lowered for word in ["example", "evidence", "support"]):
            return "Prefers claims supported by examples or evidence."
        if any(word in lowered for word in ["thoughtful", "insight", "developed"]):
            return "Prefers more thoughtful and developed points."
        if any(word in lowered for word in ["repeat", "redund", "duplicate"]):
            return "Dislikes repetition and redundant phrasing."
        if any(word in lowered for word in ["tone", "voice", "formal", "stiff"]):
            return "Prefers a tone that matches the intended voice."
        if any(word in lowered for word in ["long", "dense", "unclear"]):
            return "Prefers clear sentences with lighter density."
        if any(word in lowered for word in ["transition", "align", "task"]):
            return "Prefers smoother transitions and tighter task alignment."
        return selected_reason.strip()


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
- Use the interrupted sentence, the previous sentence, the task, and the saved user profile as evidence.
- Return exactly {TARGET_REASON_OPTIONS} reason candidates.
- Make the five reasons meaningfully different from one another.
- Prefer content-related critiques when they are supported by the stop point.
- Keep style and voice critiques available, but do not let them crowd out content critiques.
- The reasons may include issues such as:
- too generic
- too specific or too narrow
- weak or missing example
- not thoughtful enough
- repetitive or redundant
- weak contribution claim
- unclear mechanism or intuition
- unsupported or under-qualified claim
- tone or voice mismatch
- too long, dense, or unclear
- weak transition or weak task alignment
- Use at most {MAX_REASON_OPTIONS} reason candidates.
- Keep reason candidates detailed enough to guide rewriting.
- Make each reason specific to this stop point instead of generic writing advice.
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
        parsed_reasons = self._ensure_target_reason_candidates(parsed_reasons, state)
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

    def _ensure_target_reason_candidates(
        self,
        parsed_reasons: List[InterpreterReasonCandidate],
        state: SessionState,
    ) -> List[InterpreterReasonCandidate]:
        templates = self._reason_templates(state)
        reasons: List[InterpreterReasonCandidate] = []
        used_reason_texts: set[str] = set()

        for index, template in enumerate(templates, start=1):
            matched_reason = next(
                (
                    candidate.reason.strip()
                    for candidate in parsed_reasons
                    if candidate.reason.strip() and template["match"](candidate.reason.strip().lower())
                ),
                "",
            )
            chosen_reason = matched_reason or template["text"]
            normalized = chosen_reason.lower()
            if normalized in used_reason_texts:
                continue
            reasons.append(InterpreterReasonCandidate(id=f"R{len(reasons) + 1}", reason=chosen_reason))
            used_reason_texts.add(normalized)
            if len(reasons) >= TARGET_REASON_OPTIONS:
                break

        return reasons[:TARGET_REASON_OPTIONS]

    def _fallback_interpretation(self, state: SessionState) -> InterpreterResult:
        reasons = [
            InterpreterReasonCandidate(id=f"R{index}", reason=template["text"])
            for index, template in enumerate(self._reason_templates(state), start=1)
        ]

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

    def _reason_templates(self, state: SessionState) -> List[dict]:
        current_sentence = state.interruption_context.current_sentence.strip() or "the interrupted sentence"
        last_sentence = state.interruption_context.last_sentence.strip() or "the previous sentence"
        task = state.task.strip() or "the user's task"
        profile = ", ".join(state.preference_profile[-3:]) if state.preference_profile else "the saved user profile"

        return [
            {
                "text": (
                    f"The sentence may be too generic for {task}; based on '{current_sentence}', it may need more "
                    f"concrete detail, examples, or sharper wording."
                ),
                "match": lambda reason: "generic" in reason or "concrete" in reason or "detail" in reason,
            },
            {
                "text": (
                    f"The sentence may be too specific or too narrow too early, which could overcommit the draft "
                    f"in a way that limits how the writing can develop for {task}."
                ),
                "match": lambda reason: "specific" in reason or "narrow" in reason or "overcommit" in reason,
            },
            {
                "text": (
                    f"The sentence may need a stronger example or supporting detail; at this stop point, the writing "
                    f"may gesture at an idea without grounding it enough for {task}."
                ),
                "match": lambda reason: "example" in reason or "support" in reason or "evidence" in reason,
            },
            {
                "text": (
                    f"The point in '{current_sentence}' may not feel thoughtful or developed enough yet, so the user "
                    f"may want a more substantial claim or insight."
                ),
                "match": lambda reason: "thoughtful" in reason or "developed" in reason or "insight" in reason,
            },
            {
                "text": (
                    f"The sentence may repeat information that is already implied by '{last_sentence}', so it may need "
                    f"less redundancy and a fresher next move."
                ),
                "match": lambda reason: "repeat" in reason or "redund" in reason or "duplicate" in reason,
            },
            {
                "text": (
                    f"The contribution claim in '{current_sentence}' may not yet feel strong or distinct enough, so "
                    f"the user may want a clearer statement of what is new or important."
                ),
                "match": lambda reason: "contribution" in reason or "novel" in reason or "new" in reason or "important" in reason,
            },
            {
                "text": (
                    f"The sentence may point toward a result without making the mechanism or intuition clear enough, "
                    f"so the reader may not yet see why the claim should hold."
                ),
                "match": lambda reason: "mechanism" in reason or "intuition" in reason or "why" in reason or "explain" in reason,
            },
            {
                "text": (
                    f"The claim in '{current_sentence}' may feel under-supported or insufficiently qualified, so the "
                    f"user may want stronger evidence, framing, or caveats."
                ),
                "match": lambda reason: "under-supported" in reason or "unsupported" in reason or "qualified" in reason or "caveat" in reason,
            },
            {
                "text": (
                    f"The sentence may not match the user's preferred tone or voice suggested by {profile}; "
                    f"the wording at '{current_sentence}' may sound off-style."
                ),
                "match": lambda reason: "tone" in reason or "voice" in reason or "formal" in reason or "stiff" in reason,
            },
            {
                "text": (
                    f"The sentence may be too long, dense, or unclear at the stop point, making it harder to process "
                    f"quickly during streaming."
                ),
                "match": lambda reason: "long" in reason or "dense" in reason or "clear" in reason or "unclear" in reason,
            },
            {
                "text": (
                    f"The sentence may transition weakly from '{last_sentence}' or may not align tightly enough "
                    f"with the task '{task}'."
                ),
                "match": lambda reason: "transition" in reason or "align" in reason or "task" in reason or "intent" in reason,
            },
        ]


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
            temperature=REPLACEMENT_TEMPERATURE,
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

Constraints:
- Return exactly one option for each reason candidate provided in the interpreter result.
- Make the options noticeably different from each other.
- Use the reason diagnoses as distinct revision directions instead of repeating the same diagnosis.
- Rewrite only the interrupted sentence or immediate local span.
- Do not copy the full task description, bullet lists, or prompt text into the replacement.
- Each replacement_text must be short: at most two sentences and preferably under {MAX_REPLACEMENT_WORDS} words.

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
                    replacement_text = self._sanitize_replacement_text(
                        replacement_text=replacement_text,
                        state=state,
                        reason_id=reason_id,
                    )
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
                return self._ensure_target_replacements(options, state, interpreter_result)
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
        options: List[ReplacementOption] = []
        for reason in interpreter_result.reason_candidates:
            options.append(
                ReplacementOption(
                    option_id=str(uuid.uuid4()),
                    reason_id=reason.id,
                    reason=reason.reason,
                    explanation=reason.reason,
                    replacement_text=self._fallback_rewrite_for_reason(state, reason.id),
                )
            )
        return self._ensure_target_replacements(options, state, interpreter_result)

    def _ensure_target_replacements(
        self,
        options: List[ReplacementOption],
        state: SessionState,
        interpreter_result: InterpreterResult,
    ) -> List[ReplacementOption]:
        completed = list(options)
        covered_reason_ids = {item.reason_id for item in completed}

        for reason in interpreter_result.reason_candidates:
            if reason.id in covered_reason_ids:
                continue
            completed.append(
                ReplacementOption(
                    option_id=str(uuid.uuid4()),
                    reason_id=reason.id,
                    reason=reason.reason,
                    explanation=reason.reason,
                    replacement_text=self._fallback_rewrite_for_reason(state, reason.id),
                )
            )
            covered_reason_ids.add(reason.id)
            if len(completed) >= TARGET_REASON_OPTIONS:
                break

        return completed[:MAX_REASON_OPTIONS]

    def _fallback_rewrite_for_reason(self, state: SessionState, reason_id: str) -> str:
        sentence = (state.interruption_context.current_sentence or "").strip()
        task = state.task.strip() or "the task"
        previous_sentence = state.interruption_context.last_sentence.strip()
        compact = " ".join(sentence.split()).rstrip(".!?")

        if not compact:
            return "Continue with a clearer sentence that fits the task."

        if reason_id == "R1":
            return f"{compact}, with a more concrete detail tied directly to {task}."
        if reason_id == "R2":
            return f"{compact}, but framed a bit more broadly so it does not lock the draft in too early."
        if reason_id == "R3":
            return f"{compact}, with a stronger supporting example or clearer evidence."
        if reason_id == "R4":
            return f"{compact}, with a more thoughtful and developed point."
        if reason_id == "R5":
            return f"{compact}, without repeating what has already been said."
        if reason_id == "R6":
            return f"{compact}, expressed in a more natural and user-aligned voice."
        if reason_id == "R7":
            first_clause = compact.split(",")[0].strip()
            return f"{first_clause}."
        if reason_id == "R8":
            prefix = "Building on that point, " if previous_sentence else "From there, "
            return f"{prefix}{compact[:1].lower() + compact[1:] if len(compact) > 1 else compact.lower()}."
        return f"{compact}, revised to better fit the user's intent."

    def _sanitize_replacement_text(self, replacement_text: str, state: SessionState, reason_id: str) -> str:
        normalized = " ".join(replacement_text.split()).strip()
        if not normalized:
            return self._fallback_rewrite_for_reason(state, reason_id)

        lowered = normalized.lower()
        forbidden_markers = [
            "please produce:",
            "instruction:",
            "saved user profile:",
            "current live text:",
            "current accepted text:",
            "user task:",
            "interpreter result:",
        ]
        if any(marker in lowered for marker in forbidden_markers):
            return self._fallback_rewrite_for_reason(state, reason_id)

        if len(normalized.split()) > MAX_REPLACEMENT_WORDS:
            return self._fallback_rewrite_for_reason(state, reason_id)

        return normalized


class StreamingWriterAgent(StatelessLLMAgent):
    def __init__(self, model: str = "gpt-4o-mini", name: str = "streaming_writer_agent"):
        super().__init__(
            name=name,
            model=model,
            temperature=STREAMING_TEMPERATURE,
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
                if STREAM_TOKEN_DELAY_SECONDS > 0:
                    await asyncio.sleep(STREAM_TOKEN_DELAY_SECONDS)
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
