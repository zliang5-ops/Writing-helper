import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Optional

from .agents import (
    BehaviorInterpreterAgent,
    InterruptionInterpreterAgent,
    PreferenceMemoryAgent,
    ReplacementAgent,
    StreamingWriterAgent,
)
from .constants import MAX_REASON_OPTIONS, TARGET_REASON_OPTIONS
from .models import InterpreterResult, ReplacementOption, RevisionEvent, SessionState, UserProfile
from .storage import load_or_create_user_profile, save_user_profile
from .text_utils import extract_interruption_context


class WritingOrchestrator:
    def __init__(self, ui_event_queue: "queue.Queue[tuple[str, Any]]", model: str = "gpt-4o-mini"):
        self.ui_event_queue = ui_event_queue
        self.state = SessionState()
        self.user_profile: Optional[UserProfile] = None

        self.memory_agent = PreferenceMemoryAgent()
        self.interpreter_agent = InterruptionInterpreterAgent(model=model)
        self.behavior_interpreter_agent = BehaviorInterpreterAgent(model=model)
        self.replacement_agent = ReplacementAgent(model=model)
        self.writer_agent = StreamingWriterAgent(model=model)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._busy = threading.Event()

    def start_background_loop(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        def runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._worker_thread = threading.Thread(target=runner, daemon=True)
        self._worker_thread.start()

        while self._loop is None:
            time.sleep(0.02)

    def shutdown(self) -> None:
        async def _close() -> None:
            await self.writer_agent.close()
            await self.interpreter_agent.close()
            await self.behavior_interpreter_agent.close()
            await self.replacement_agent.close()

        try:
            if self._loop and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_close(), self._loop)
                future.result(timeout=10)
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    def _emit(self, event_type: str, payload: Any) -> None:
        self.ui_event_queue.put((event_type, payload))

    def _set_busy(self, value: bool) -> None:
        if value:
            self._busy.set()
        else:
            self._busy.clear()
        self._emit("busy", value)

    def can_start(self) -> bool:
        return not self._busy.is_set()

    def stop_streaming(self) -> None:
        self._stop_flag.set()

    def start_new_task(self, username: str, task: str) -> None:
        if not self.can_start():
            return
        clean_username = username.strip()
        clean_task = task.strip()
        if not clean_username:
            self._emit("error", "Please enter a user name.")
            return
        if not clean_task:
            self._emit("error", "Please enter a writing task or purpose.")
            return

        profile, created_new = load_or_create_user_profile(clean_username)
        self.user_profile = profile
        self.state.username = profile.username
        self.state.task = clean_task
        self.state.live_text = ""
        self.state.accepted_text = ""
        self.state.preference_profile = list(profile.preference_profile)
        self.state.revision_log = list(profile.revision_log)
        self.state.replacement_options = []
        self.state.active_interpreter_result = None
        self.state.interruption_context = extract_interruption_context("")
        self._stop_flag.clear()

        status = (
            f"Loaded existing profile for user '{profile.username}'."
            if not created_new
            else f"User '{profile.username}' was not found. Created a new local profile."
        )
        self._emit("credential_status", status)
        self._emit(
            "guidance",
            (
                "Guidance before use:\n"
                "If the current content becomes unsatisfactory, click 'Stop Streaming'.\n"
                "The interpreter will analyze the stop point and generate replacement options."
            ),
        )
        self._emit("profile_update", self.state.preference_profile)
        self._emit("status", "Starting streaming generation...")
        self._submit_coroutine(self._run_main_stream())

    def continue_generation(self) -> None:
        if not self.can_start():
            return
        if not self.state.username or not self.state.task:
            self._emit("error", "Enter a user name and task before continuing.")
            return
        self._stop_flag.clear()
        self._submit_coroutine(self._run_main_stream())

    def accept_current_text(self) -> None:
        self.state.accepted_text = self.state.live_text
        self._emit("accepted_text", self.state.accepted_text)

    def apply_selected_option(self, option_id: str, other_mode: str, other_text: str) -> None:
        if not self.can_start():
            return
        selected = next((item for item in self.state.replacement_options if item.option_id == option_id), None)
        if selected is None:
            self._emit("error", "Please select a replacement option.")
            return

        if selected.option_kind == "other":
            if other_mode not in {"describe_revision", "write_own_text"}:
                self._emit("error", "Please choose how you want to use Other.")
                return
            if not other_text.strip():
                self._emit("error", "Please enter text for the selected Other action.")
                return
            self._submit_coroutine(self._handle_other_flow(other_mode, other_text.strip()))
            return

        self._apply_revision_selection(
            selected_reason_id=selected.reason_id,
            selected_reason=selected.reason,
            selected_revision=selected.replacement_text,
            selection_kind="replacement_option",
            custom_input="",
            profile_summary=self._preferred_profile_summary(self.state.active_interpreter_result, selected.reason),
        )

    def export_session_json(self) -> str:
        payload = {
            "state": self.state.context_snapshot(),
            "active_interpreter_result": self.state.active_interpreter_result.to_dict() if self.state.active_interpreter_result else None,
            "replacement_options": [asdict(option) for option in self.state.replacement_options],
            "revision_log": [asdict(event) for event in self.state.revision_log],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _submit_coroutine(self, coro: Any) -> None:
        self.start_background_loop()
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _run_main_stream(self) -> None:
        self._set_busy(True)
        try:
            self._emit("stream_mode", "main")
            self._emit("status", "Streaming main writer...")

            def on_token(token: str) -> None:
                self.state.live_text += token
                self._emit("append_text", token)

            await self.writer_agent.stream_generate(
                state=self.state,
                on_token=on_token,
                should_stop=self._stop_flag.is_set,
            )

            if self._stop_flag.is_set():
                await self._prepare_replacement_options()
            else:
                self._emit("status", "Streaming finished.")
        except Exception as e:
            self._emit("error", f"{type(e).__name__}: {e}")
        finally:
            self._set_busy(False)

    async def _prepare_replacement_options(self) -> None:
        self.state.interruption_context = extract_interruption_context(self.state.live_text)
        self._emit("interruption_context", self._stop_point_payload())
        self._emit("status", "Analyzing stop point and building replacement options...")

        interpreter_result = await self.interpreter_agent.interpret(self.state)
        interpreter_result.stop_point.replacement_start = self.state.interruption_context.replacement_start
        self.state.active_interpreter_result = interpreter_result
        options = await self.replacement_agent.build_replacements(self.state, interpreter_result)
        options = options[:MAX_REASON_OPTIONS]
        options.append(
            ReplacementOption(
                option_id=str(uuid.uuid4()),
                reason_id="OTHER",
                reason="Other",
                explanation="Choose this if none of the generated options fit.",
                replacement_text="",
                option_kind="other",
            )
        )
        self.state.replacement_options = options

        self._emit("interpreter_result", interpreter_result.to_dict())
        self._emit("replacement_options", [asdict(option) for option in options])
        self._emit(
            "status",
            f"Replacement options are ready. Select one and click Apply to replace the interrupted sentence. "
            f"Target options: {TARGET_REASON_OPTIONS}; current generated options: {max(0, len(options) - 1)}.",
        )

    async def _handle_other_flow(self, other_mode: str, other_text: str) -> None:
        self._set_busy(True)
        try:
            behavior_result = await self.behavior_interpreter_agent.interpret_behavior(
                self.state,
                behavior_text=other_text,
                behavior_mode=other_mode,
            )
            self.state.active_interpreter_result = behavior_result
            if other_mode == "describe_revision":
                generated_revision = await self.replacement_agent.build_custom_revision(
                    task=self.state.task,
                    passage=self.state.live_text,
                    custom_instruction=other_text,
                )
                self._apply_revision_selection(
                    selected_reason_id="OTHER",
                    selected_reason="Other",
                    selected_revision=generated_revision,
                    selection_kind="other_describe_revision",
                    custom_input=other_text,
                    profile_summary=self._preferred_profile_summary(behavior_result, other_text),
                )
            else:
                self._apply_revision_selection(
                    selected_reason_id="OTHER",
                    selected_reason="Other",
                    selected_revision=other_text,
                    selection_kind="other_write_own_text",
                    custom_input=other_text,
                    profile_summary=self._preferred_profile_summary(behavior_result, other_text),
                )
            self._emit("interpreter_result", behavior_result.to_dict())
        except Exception as e:
            self._emit("error", f"{type(e).__name__}: {e}")
        finally:
            self._set_busy(False)

    def _apply_revision_selection(
        self,
        selected_reason_id: str,
        selected_reason: str,
        selected_revision: str,
        selection_kind: str,
        custom_input: str,
        profile_summary: str,
    ) -> None:
        if profile_summary:
            self.state.preference_profile = self.memory_agent.update_profile(self.state.preference_profile, profile_summary)

        start = self.state.interruption_context.replacement_start
        prefix = self.state.live_text[:start].rstrip()
        revision = selected_revision.strip()
        self.state.live_text = f"{prefix} {revision}".strip() if prefix else revision
        self.state.accepted_text = self.state.live_text

        interpreter_result = self.state.active_interpreter_result.to_dict() if self.state.active_interpreter_result else {}
        event = RevisionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            username=self.state.username,
            task=self.state.task,
            stop_point=self.state.interruption_context,
            interpreter_result=interpreter_result,
            selected_reason_id=selected_reason_id,
            selected_reason=selected_reason,
            selected_revision=selected_revision,
            selection_kind=selection_kind,
            custom_input=custom_input,
            updated_preference_profile=list(self.state.preference_profile),
        )
        self.state.revision_log.append(event)
        self._save_profile()

        self.state.replacement_options = []
        self.state.active_interpreter_result = None
        self._emit("set_text", self.state.live_text)
        self._emit("accepted_text", self.state.accepted_text)
        self._emit("profile_update", self.state.preference_profile)
        self._emit("replacement_options", [])
        self._emit(
            "revision_applied",
            {
                "selected_reason_id": selected_reason_id,
                "selected_reason": selected_reason,
                "selected_revision": selected_revision,
                "selection_kind": selection_kind,
                "profile_summary_added": profile_summary,
            },
        )
        self._emit("status", "Revision applied, saved to the user profile, and accepted as the new baseline. Continuing generation...")
        self._submit_coroutine(self._resume_after_revision())

    async def _resume_after_revision(self) -> None:
        while self._busy.is_set():
            await asyncio.sleep(0.05)
        self._stop_flag.clear()
        await self._run_main_stream()

    def _preferred_profile_summary(self, interpreter_result: Optional[InterpreterResult], fallback: str) -> str:
        if interpreter_result is None:
            return fallback.strip()
        summary = interpreter_result.profile_update.preference_summary.strip()
        return summary or fallback.strip()

    def _stop_point_payload(self) -> dict:
        return {
            "termination_point": self.state.interruption_context.termination_point,
            "last_sentence": self.state.interruption_context.last_sentence,
            "current_sentence": self.state.interruption_context.current_sentence,
        }

    def _save_profile(self) -> None:
        if self.user_profile is None:
            return
        self.user_profile.preference_profile = list(self.state.preference_profile)
        self.user_profile.revision_log = list(self.state.revision_log)
        save_user_profile(self.user_profile)
