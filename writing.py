
    """
    interruption_writing_helper.py

    Single-file prototype for an interruption-based multi-agent writing helper.

    What this file gives you:
    1. A popup desktop frontend using tkinter.
    2. A streaming main writer agent using Microsoft AutoGen.
    3. Explicit agent structure:
       - StreamingWriterAgent
       - MonitorAgent
       - MemoryAgent
       - WritingStyleAgent
       - RevisionAgent
       - Orchestrator
    4. Clear wiring for who sends information to whom.
    5. Placeholders where you can later decide:
       - exact dissatisfaction labels
       - exact restriction format
       - exact prompt protocol between agents
       - whether agents share one document or separate state
       - whether monitor/style/memory should later become LLM agents too

    Install:
        pip install -U "autogen-agentchat" "autogen-ext[openai]" openai

    Environment:
        set OPENAI_API_KEY=...
        # Windows PowerShell:
        # $env:OPENAI_API_KEY="..."

    Run:
        python interruption_writing_helper.py

    Notes:
    - The main writer uses model_client_stream=True.
    - The "Stop Streaming" button interrupts the current stream.
    - After interruption, the monitor agent proposes dissatisfaction options.
    - The user can choose one or type a custom reason.
    - Memory/style/revision agents update constraints and prepare the next generation.
    """

    import asyncio
    import json
    import os
    import queue
    import threading
    import time
    import uuid
    from dataclasses import dataclass, field
    from typing import Dict, List, Optional, Any, Callable

    import tkinter as tk
    from tkinter import ttk, messagebox

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient


    # =========================
    # Data structures
    # =========================

    @dataclass
    class RevisionEvent:
        event_id: str
        timestamp: float
        task: str
        style_goal: str
        partial_text: str
        interruption_tail: str
        selected_reason: str
        custom_reason: str
        inferred_style_notes: List[str] = field(default_factory=list)
        added_restrictions: List[str] = field(default_factory=list)


    @dataclass
    class SessionState:
        task: str = ""
        style_goal: str = ""
        live_text: str = ""
        accepted_text: str = ""
        last_interruption_tail: str = ""
        dissatisfaction_options: List[str] = field(default_factory=list)
        restrictions: List[str] = field(default_factory=list)
        style_notes: List[str] = field(default_factory=list)
        revision_log: List[RevisionEvent] = field(default_factory=list)
        user_guidance_shown: bool = False

        def context_snapshot(self) -> Dict[str, Any]:
            return {
                "task": self.task,
                "style_goal": self.style_goal,
                "accepted_text": self.accepted_text,
                "live_text": self.live_text,
                "restrictions": list(self.restrictions),
                "style_notes": list(self.style_notes),
                "revision_count": len(self.revision_log),
            }


    # =========================
    # Agent base classes
    # =========================

    class BaseLocalAgent:
        def __init__(self, name: str):
            self.name = name


    class MemoryAgent(BaseLocalAgent):
        """
        Records dissatisfaction reasons and converts them into reusable restrictions.
        You said you want to decide the exact protocol later, so the mapping here is simple and editable.
        """
        def __init__(self, name: str = "memory_agent"):
            super().__init__(name)

        def build_restrictions(
            self,
            selected_reason: str,
            custom_reason: str,
            state: SessionState,
        ) -> List[str]:
            reason = (custom_reason or selected_reason or "").strip().lower()
            new_rules: List[str] = []

            # You can later replace this with your own protocol or a separate LLM agent.
            if "generic" in reason:
                new_rules.append("Avoid generic filler and broad vague claims.")
            if "formal" in reason:
                new_rules.append("Reduce stiffness and avoid overly polished corporate phrasing.")
            if "repetitive" in reason:
                new_rules.append("Vary phrasing and avoid repeating the same structure.")
            if "long" in reason:
                new_rules.append("Prefer shorter sentences and faster point delivery.")
            if "natural" in reason or "human" in reason:
                new_rules.append("Sound more natural and less formulaic.")
            if "off-topic" in reason:
                new_rules.append("Stay tightly aligned with the user's original purpose.")
            if "academic" in reason:
                new_rules.append("Use a more academic tone with tighter logic.")

            if custom_reason and not new_rules:
                new_rules.append(f"User-specific revision restriction: {custom_reason.strip()}")

            # Deduplicate while preserving order.
            merged = list(dict.fromkeys(state.restrictions + new_rules))
            return merged


    class WritingStyleAgent(BaseLocalAgent):
        """
        Infers style hints from the style goal and revision history.
        This is heuristic for now and can later become another AutoGen/LLM agent.
        """
        def __init__(self, name: str = "style_agent"):
            super().__init__(name)

        def infer_style_notes(self, state: SessionState, selected_reason: str, custom_reason: str) -> List[str]:
            text = f"{state.style_goal} {selected_reason} {custom_reason}".lower()
            notes = list(state.style_notes)

            def add(note: str) -> None:
                if note not in notes:
                    notes.append(note)

            if "concise" in text or "too long" in text or "long" in text:
                add("Prefer concise sentences.")
            if "natural" in text or "human" in text:
                add("Avoid formulaic AI-sounding phrasing.")
            if "academic" in text:
                add("Maintain academic clarity and logical precision.")
            if "formal" in text:
                add("Keep professional tone but avoid stiffness.")
            if "casual" in text:
                add("Use lighter, more conversational wording.")
            if "direct" in text:
                add("Get to the point quickly.")
            if "persuasive" in text:
                add("Use purpose-driven argument structure.")

            return notes


    class MonitorAgent(BaseLocalAgent):
        """
        Reviews the interruption point and proposes likely dissatisfaction categories.
        You asked for all interaction options to be given based on input.
        This agent returns those candidate options.
        """
        DEFAULT_OPTIONS = [
            "Too generic",
            "Too formal",
            "Too repetitive",
            "Too long",
            "Not natural enough",
            "Off-topic",
            "Need more academic tone",
            "Need more concise wording",
            "Other",
        ]

        def __init__(self, name: str = "monitor_agent"):
            super().__init__(name)

        def analyze_interruption(self, state: SessionState) -> List[str]:
            tail = state.last_interruption_tail.lower()
            style = state.style_goal.lower()
            options = []

            # Lightweight heuristics. Replace freely later.
            if len(state.live_text) > 400:
                options.append("Too long")
            if any(word in tail for word in ["moreover", "furthermore", "therefore", "indeed"]):
                options.append("Too formal")
            if any(word in style for word in ["natural", "human", "less formulaic"]):
                options.append("Not natural enough")
            if any(word in style for word in ["academic"]):
                options.append("Need more academic tone")
            if any(word in style for word in ["concise", "direct"]):
                options.append("Need more concise wording")

            for item in self.DEFAULT_OPTIONS:
                if item not in options:
                    options.append(item)

            return options


    class StreamingWriterAgent(BaseLocalAgent):
        """
        Main generation agent using AutoGen with model_client_stream=True.
        This is the streaming agent required by your spec.
        """
        def __init__(self, model: str = "gpt-4o-mini", name: str = "streaming_writer_agent"):
            super().__init__(name)
            self.model = model
            self.model_client = OpenAIChatCompletionClient(model=model)
            self.agent = AssistantAgent(
                name="writer",
                model_client=self.model_client,
                model_client_stream=True,
                system_message=(
                    "You are the main writing generator in an interruption-aware writing system. "
                    "Follow the user's task, style instructions, and current restrictions carefully. "
                    "Write usable prose, not bullet points unless explicitly asked. "
                    "Do not explain the system. Only generate the requested writing."
                ),
            )

        async def stream_generate(
            self,
            state: SessionState,
            on_token: Callable[[str], None],
            should_stop: Callable[[], bool],
            mode: str = "continue",
        ) -> str:
            prompt = self._build_prompt(state, mode=mode)

            accumulated = ""
            async for item in self.agent.run_stream(task=prompt):
                if should_stop():
                    break

                # AutoGen stream items vary by type. We only append text-bearing chunks.
                text = getattr(item, "content", None)
                if isinstance(text, str) and text:
                    accumulated += text
                    on_token(text)

            return accumulated

        def _build_prompt(self, state: SessionState, mode: str = "continue") -> str:
            restrictions = "\n".join(f"- {r}" for r in state.restrictions) or "- None yet."
            style_notes = "\n".join(f"- {s}" for s in state.style_notes) or "- None yet."

            if mode == "continue":
                return f"""
Task:
{state.task}

Broad style goal:
{state.style_goal}

Current style notes:
{style_notes}

Current restrictions:
{restrictions}

Current accepted text:
{state.accepted_text}

Current live text before the next continuation:
{state.live_text}

Instruction:
Continue the writing from where it currently stands.
Do not restart from scratch.
Do not explain your choices.
"""
            elif mode == "revise_from_interruption":
                return f"""
Task:
{state.task}

Broad style goal:
{state.style_goal}

Current style notes:
{style_notes}

Current restrictions:
{restrictions}

Existing text before the friction point:
{state.accepted_text}

Interrupted partial text:
{state.live_text}

Local interruption tail:
{state.last_interruption_tail}

Instruction:
Write a revised continuation or local replacement starting from the interruption region.
Address the restrictions directly.
Do not explain the revisions.
"""
            else:
                return f"""
Task:
{state.task}

Broad style goal:
{state.style_goal}

Current style notes:
{style_notes}

Current restrictions:
{restrictions}

Instruction:
Write the requested content from scratch.
"""


        async def close(self) -> None:
            await self.model_client.close()


    class RevisionAgent(BaseLocalAgent):
        """
        Produces revised continuation by calling the same streaming writer in a revision mode.
        If you later want a separate model or separate prompt protocol, split it here.
        """
        def __init__(self, writer_agent: StreamingWriterAgent, name: str = "revision_agent"):
            super().__init__(name)
            self.writer_agent = writer_agent

        async def stream_revision(
            self,
            state: SessionState,
            on_token: Callable[[str], None],
            should_stop: Callable[[], bool],
        ) -> str:
            return await self.writer_agent.stream_generate(
                state=state,
                on_token=on_token,
                should_stop=should_stop,
                mode="revise_from_interruption",
            )


    # =========================
    # Orchestrator
    # =========================

    class WritingOrchestrator:
        """
        Clear flow:

        User -> Orchestrator -> StreamingWriterAgent
                                 |
                                 v
                              interruption
                                 |
                                 v
                            MonitorAgent
                                 |
                                 v
                     user selects / types dissatisfaction
                                 |
                                 v
                           MemoryAgent + StyleAgent
                                 |
                                 v
                            RevisionAgent
                                 |
                                 v
                          back to StreamingWriterAgent

        This is intentionally explicit so you can later replace the internals
        without changing the UI wiring.
        """

        def __init__(self, ui_event_queue: "queue.Queue[tuple[str, Any]]", model: str = "gpt-4o-mini"):
            self.ui_event_queue = ui_event_queue
            self.state = SessionState()

            self.monitor_agent = MonitorAgent()
            self.memory_agent = MemoryAgent()
            self.style_agent = WritingStyleAgent()
            self.writer_agent = StreamingWriterAgent(model=model)
            self.revision_agent = RevisionAgent(self.writer_agent)

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

            # Wait briefly for loop creation.
            while self._loop is None:
                time.sleep(0.02)

        def shutdown(self) -> None:
            async def _close() -> None:
                await self.writer_agent.close()

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

        def start_new_task(self, task: str, style_goal: str) -> None:
            if not self.can_start():
                return

            self.state.task = task.strip()
            self.state.style_goal = style_goal.strip()
            self.state.live_text = ""
            self.state.accepted_text = ""
            self.state.last_interruption_tail = ""
            self.state.dissatisfaction_options = []
            self._stop_flag.clear()

            self._emit("guidance", (
                "Guidance before use:\n"
                "If the current content becomes unsatisfactory, click 'Stop Streaming' immediately.\n"
                "After that, choose the reason or type your own reason."
            ))
            self.state.user_guidance_shown = True

            self._submit_coroutine(self._run_main_stream())

        def continue_generation(self) -> None:
            if not self.can_start():
                return
            self._stop_flag.clear()
            self._submit_coroutine(self._run_main_stream())

        def accept_current_text(self) -> None:
            self.state.accepted_text = self.state.live_text
            self._emit("accepted_text", self.state.accepted_text)

        def submit_feedback(self, selected_reason: str, custom_reason: str) -> None:
            if not self.can_start():
                return

            restrictions = self.memory_agent.build_restrictions(selected_reason, custom_reason, self.state)
            style_notes = self.style_agent.infer_style_notes(self.state, selected_reason, custom_reason)

            self.state.restrictions = restrictions
            self.state.style_notes = style_notes

            event = RevisionEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                task=self.state.task,
                style_goal=self.state.style_goal,
                partial_text=self.state.live_text,
                interruption_tail=self.state.last_interruption_tail,
                selected_reason=selected_reason,
                custom_reason=custom_reason,
                inferred_style_notes=list(style_notes),
                added_restrictions=list(restrictions),
            )
            self.state.revision_log.append(event)

            self._emit("memory_update", {
                "selected_reason": selected_reason,
                "custom_reason": custom_reason,
                "restrictions": restrictions,
                "style_notes": style_notes,
                "revision_count": len(self.state.revision_log),
            })

            self._stop_flag.clear()
            self._submit_coroutine(self._run_revision_stream())

        def export_session_json(self) -> str:
            payload = {
                "state": self.state.context_snapshot(),
                "revision_log": [
                    {
                        "event_id": e.event_id,
                        "timestamp": e.timestamp,
                        "task": e.task,
                        "style_goal": e.style_goal,
                        "partial_text": e.partial_text,
                        "interruption_tail": e.interruption_tail,
                        "selected_reason": e.selected_reason,
                        "custom_reason": e.custom_reason,
                        "inferred_style_notes": e.inferred_style_notes,
                        "added_restrictions": e.added_restrictions,
                    }
                    for e in self.state.revision_log
                ],
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
                self.state.last_interruption_tail = ""

                def on_token(token: str) -> None:
                    self.state.live_text += token
                    self.state.last_interruption_tail = self.state.live_text[-220:]
                    self._emit("append_text", token)

                await self.writer_agent.stream_generate(
                    state=self.state,
                    on_token=on_token,
                    should_stop=self._stop_flag.is_set,
                    mode="continue",
                )

                if self._stop_flag.is_set():
                    options = self.monitor_agent.analyze_interruption(self.state)
                    self.state.dissatisfaction_options = options
                    self._emit("status", "Streaming stopped. Monitor agent prepared feedback options.")
                    self._emit("feedback_options", options)
                else:
                    self._emit("status", "Streaming finished.")
            except Exception as e:
                self._emit("error", f"{type(e).__name__}: {e}")
            finally:
                self._set_busy(False)

        async def _run_revision_stream(self) -> None:
            self._set_busy(True)
            try:
                self._emit("stream_mode", "revision")
                self._emit("status", "Generating revised continuation...")
                self.state.live_text += "\n"

                def on_token(token: str) -> None:
                    self.state.live_text += token
                    self.state.last_interruption_tail = self.state.live_text[-220:]
                    self._emit("append_text", token)

                await self.revision_agent.stream_revision(
                    state=self.state,
                    on_token=on_token,
                    should_stop=self._stop_flag.is_set,
                )

                if self._stop_flag.is_set():
                    options = self.monitor_agent.analyze_interruption(self.state)
                    self.state.dissatisfaction_options = options
                    self._emit("status", "Revision stream stopped. Updated options are ready.")
                    self._emit("feedback_options", options)
                else:
                    self._emit("status", "Revision finished. You can accept or continue.")
            except Exception as e:
                self._emit("error", f"{type(e).__name__}: {e}")
            finally:
                self._set_busy(False)


    # =========================
    # Tkinter frontend
    # =========================

    class WritingHelperApp:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("Interruption-Based Multi-Agent Writing Helper")
            self.root.geometry("1280x860")

            self.ui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
            self.orchestrator = WritingOrchestrator(self.ui_queue)

            self.selected_reason_var = tk.StringVar(value="")
            self.status_var = tk.StringVar(value="Ready.")
            self.busy_var = tk.StringVar(value="Idle")
            self.stream_mode_var = tk.StringVar(value="-")

            self._build_layout()
            self._poll_events()

            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        def _build_layout(self) -> None:
            top = ttk.Frame(self.root, padding=10)
            top.pack(fill="x")

            ttk.Label(top, text="Task / Purpose").grid(row=0, column=0, sticky="w")
            self.task_entry = tk.Text(top, height=4, width=75, wrap="word")
            self.task_entry.grid(row=1, column=0, padx=(0, 12), pady=(4, 10), sticky="ew")

            ttk.Label(top, text="Broad Style Goal").grid(row=0, column=1, sticky="w")
            self.style_entry = tk.Text(top, height=4, width=45, wrap="word")
            self.style_entry.grid(row=1, column=1, pady=(4, 10), sticky="ew")

            top.columnconfigure(0, weight=3)
            top.columnconfigure(1, weight=2)

            button_row = ttk.Frame(self.root, padding=(10, 0))
            button_row.pack(fill="x")

            self.start_btn = ttk.Button(button_row, text="Start Streaming", command=self._start_streaming)
            self.start_btn.pack(side="left", padx=(0, 8))

            self.stop_btn = ttk.Button(button_row, text="Stop Streaming", command=self._stop_streaming)
            self.stop_btn.pack(side="left", padx=(0, 8))

            self.accept_btn = ttk.Button(button_row, text="Accept Current Text", command=self._accept_text)
            self.accept_btn.pack(side="left", padx=(0, 8))

            self.continue_btn = ttk.Button(button_row, text="Continue Generation", command=self._continue_generation)
            self.continue_btn.pack(side="left", padx=(0, 8))

            self.export_btn = ttk.Button(button_row, text="Export Session JSON", command=self._export_json)
            self.export_btn.pack(side="left")

            center = ttk.Panedwindow(self.root, orient="horizontal")
            center.pack(fill="both", expand=True, padx=10, pady=10)

            left_frame = ttk.Frame(center)
            right_frame = ttk.Frame(center)
            center.add(left_frame, weight=3)
            center.add(right_frame, weight=2)

            ttk.Label(left_frame, text="Live Document").pack(anchor="w")
            self.output_text = tk.Text(left_frame, wrap="word", font=("Segoe UI", 11))
            self.output_text.pack(fill="both", expand=True, pady=(4, 8))

            ttk.Label(left_frame, text="System / Memory / State Log").pack(anchor="w")
            self.log_text = tk.Text(left_frame, wrap="word", height=12, font=("Consolas", 10))
            self.log_text.pack(fill="x", expand=False)

            ttk.Label(right_frame, text="Feedback Options from Monitor Agent").pack(anchor="w")
            self.options_frame = ttk.Frame(right_frame)
            self.options_frame.pack(fill="x", pady=(6, 10))

            ttk.Label(right_frame, text="Custom Reason (if options do not fit)").pack(anchor="w")
            self.custom_reason_text = tk.Text(right_frame, height=5, wrap="word")
            self.custom_reason_text.pack(fill="x", pady=(4, 8))

            self.submit_feedback_btn = ttk.Button(
                right_frame,
                text="Submit Feedback and Revise",
                command=self._submit_feedback,
            )
            self.submit_feedback_btn.pack(anchor="w", pady=(0, 10))

            ttk.Label(right_frame, text="Current Restrictions").pack(anchor="w")
            self.restrictions_text = tk.Text(right_frame, height=10, wrap="word")
            self.restrictions_text.pack(fill="x", pady=(4, 10))

            ttk.Label(right_frame, text="Current Style Notes").pack(anchor="w")
            self.style_notes_text = tk.Text(right_frame, height=8, wrap="word")
            self.style_notes_text.pack(fill="x", pady=(4, 10))

            status = ttk.Frame(self.root, padding=(10, 0, 10, 10))
            status.pack(fill="x")
            ttk.Label(status, text="Status:").pack(side="left")
            ttk.Label(status, textvariable=self.status_var).pack(side="left", padx=(6, 20))
            ttk.Label(status, text="Busy:").pack(side="left")
            ttk.Label(status, textvariable=self.busy_var).pack(side="left", padx=(6, 20))
            ttk.Label(status, text="Mode:").pack(side="left")
            ttk.Label(status, textvariable=self.stream_mode_var).pack(side="left", padx=(6, 0))

        def _start_streaming(self) -> None:
            task = self.task_entry.get("1.0", "end").strip()
            style_goal = self.style_entry.get("1.0", "end").strip()
            if not task:
                messagebox.showwarning("Missing task", "Please enter a writing task or purpose.")
                return

            self.output_text.delete("1.0", "end")
            self.log_text.delete("1.0", "end")
            self.restrictions_text.delete("1.0", "end")
            self.style_notes_text.delete("1.0", "end")
            self._clear_feedback_options()

            self.orchestrator.start_new_task(task, style_goal)

        def _stop_streaming(self) -> None:
            self.orchestrator.stop_streaming()
            self._append_log("[UI] Stop requested.\n")

        def _accept_text(self) -> None:
            self.orchestrator.accept_current_text()
            self._append_log("[UI] Current text accepted as baseline.\n")

        def _continue_generation(self) -> None:
            self.orchestrator.continue_generation()

        def _submit_feedback(self) -> None:
            selected = self.selected_reason_var.get().strip()
            custom = self.custom_reason_text.get("1.0", "end").strip()

            if not selected and not custom:
                messagebox.showwarning("Missing feedback", "Choose an option or type a custom reason.")
                return

            self.orchestrator.submit_feedback(selected_reason=selected, custom_reason=custom)

        def _export_json(self) -> None:
            payload = self.orchestrator.export_session_json()
            popup = tk.Toplevel(self.root)
            popup.title("Session JSON")
            popup.geometry("900x650")
            txt = tk.Text(popup, wrap="word")
            txt.pack(fill="both", expand=True)
            txt.insert("1.0", payload)

        def _clear_feedback_options(self) -> None:
            for child in self.options_frame.winfo_children():
                child.destroy()
            self.selected_reason_var.set("")

        def _set_feedback_options(self, options: List[str]) -> None:
            self._clear_feedback_options()
            for option in options:
                rb = ttk.Radiobutton(
                    self.options_frame,
                    text=option,
                    value=option,
                    variable=self.selected_reason_var,
                )
                rb.pack(anchor="w", pady=2)

        def _append_log(self, text: str) -> None:
            self.log_text.insert("end", text)
            self.log_text.see("end")

        def _poll_events(self) -> None:
            try:
                while True:
                    event_type, payload = self.ui_queue.get_nowait()

                    if event_type == "guidance":
                        self._append_log(f"[Guidance]\n{payload}\n\n")
                    elif event_type == "status":
                        self.status_var.set(payload)
                        self._append_log(f"[Status] {payload}\n")
                    elif event_type == "append_text":
                        self.output_text.insert("end", payload)
                        self.output_text.see("end")
                    elif event_type == "feedback_options":
                        self._set_feedback_options(payload)
                        self._append_log("[MonitorAgent] Feedback options updated.\n")
                    elif event_type == "memory_update":
                        self.restrictions_text.delete("1.0", "end")
                        self.restrictions_text.insert("1.0", "\n".join(payload["restrictions"]))
                        self.style_notes_text.delete("1.0", "end")
                        self.style_notes_text.insert("1.0", "\n".join(payload["style_notes"]))
                        self._append_log("[MemoryAgent/StyleAgent] State updated:\n")
                        self._append_log(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                    elif event_type == "accepted_text":
                        self._append_log("[Accepted Baseline Updated]\n")
                    elif event_type == "busy":
                        self.busy_var.set("Yes" if payload else "No")
                    elif event_type == "stream_mode":
                        self.stream_mode_var.set(payload)
                    elif event_type == "error":
                        self._append_log(f"[Error] {payload}\n")
                        messagebox.showerror("Error", payload)
                    else:
                        self._append_log(f"[Unknown Event] {event_type}: {payload}\n")
            except queue.Empty:
                pass

            self.root.after(80, self._poll_events)

        def _on_close(self) -> None:
            try:
                self.orchestrator.shutdown()
            finally:
                self.root.destroy()


    def main() -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Set it in your environment before running this script."
            )

        root = tk.Tk()
        app = WritingHelperApp(root)
        root.mainloop()


    if __name__ == "__main__":
        main()
