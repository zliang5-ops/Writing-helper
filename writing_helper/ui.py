import json
import queue
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List

from .orchestrator import WritingOrchestrator


class WritingHelperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Interruption-Based Multi-Agent Writing Helper")
        self.root.geometry("1440x980")

        self.ui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.orchestrator = WritingOrchestrator(self.ui_queue)

        self.selected_option_var = tk.StringVar(value="")
        self.other_mode_var = tk.StringVar(value="describe_revision")
        self.status_var = tk.StringVar(value="Ready.")
        self.busy_var = tk.StringVar(value="Idle")
        self.stream_mode_var = tk.StringVar(value="-")
        self.credential_var = tk.StringVar(value="No user loaded.")

        self._build_layout()
        self._poll_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="User Name").grid(row=0, column=0, sticky="w")
        self.username_entry = ttk.Entry(top, width=24)
        self.username_entry.grid(row=1, column=0, padx=(0, 12), pady=(4, 10), sticky="ew")

        ttk.Label(top, text="Task / Purpose").grid(row=0, column=1, sticky="w")
        self.task_entry = tk.Text(top, height=4, width=62, wrap="word")
        self.task_entry.grid(row=1, column=1, padx=(0, 12), pady=(4, 10), sticky="ew")

        ttk.Label(top, text="Credential Log").grid(row=0, column=2, sticky="w")
        ttk.Label(top, textvariable=self.credential_var, wraplength=340).grid(row=1, column=2, sticky="nw")

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=3)
        top.columnconfigure(2, weight=2)

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

        ttk.Label(left_frame, text="System Log").pack(anchor="w")
        self.log_text = tk.Text(left_frame, wrap="word", height=12, font=("Consolas", 10))
        self.log_text.pack(fill="x", expand=False)

        ttk.Label(right_frame, text="Interpreter Output").pack(anchor="w")
        self.interpreter_text = tk.Text(right_frame, height=16, wrap="word")
        self.interpreter_text.pack(fill="x", pady=(4, 10))

        ttk.Label(right_frame, text="Replacement Options").pack(anchor="w")
        self.options_frame = ttk.Frame(right_frame)
        self.options_frame.pack(fill="x", pady=(6, 10))

        ttk.Label(right_frame, text="Other Action").pack(anchor="w")
        other_mode_frame = ttk.Frame(right_frame)
        other_mode_frame.pack(fill="x", pady=(4, 6))
        ttk.Radiobutton(
            other_mode_frame,
            text="Provide What To Revise",
            value="describe_revision",
            variable=self.other_mode_var,
        ).pack(anchor="w")
        ttk.Radiobutton(
            other_mode_frame,
            text="Write My Own Text",
            value="write_own_text",
            variable=self.other_mode_var,
        ).pack(anchor="w")

        ttk.Label(right_frame, text="Other Input").pack(anchor="w")
        self.other_text = tk.Text(right_frame, height=6, wrap="word")
        self.other_text.pack(fill="x", pady=(4, 8))
        ttk.Label(
            right_frame,
            text="If none of the generated options fit, select a custom-input option below and type here.",
            wraplength=420,
        ).pack(anchor="w", pady=(0, 8))

        self.apply_replacement_btn = ttk.Button(
            right_frame,
            text="Apply Selected Option and Continue",
            command=self._apply_option,
        )
        self.apply_replacement_btn.pack(anchor="w", pady=(0, 10))

        ttk.Label(right_frame, text="User Preference Profile").pack(anchor="w")
        self.profile_text = tk.Text(right_frame, height=10, wrap="word")
        self.profile_text.pack(fill="x", pady=(4, 10))

        status = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        status.pack(fill="x")
        ttk.Label(status, text="Status:").pack(side="left")
        ttk.Label(status, textvariable=self.status_var).pack(side="left", padx=(6, 20))
        ttk.Label(status, text="Busy:").pack(side="left")
        ttk.Label(status, textvariable=self.busy_var).pack(side="left", padx=(6, 20))
        ttk.Label(status, text="Mode:").pack(side="left")
        ttk.Label(status, textvariable=self.stream_mode_var).pack(side="left", padx=(6, 0))

    def _start_streaming(self) -> None:
        username = self.username_entry.get().strip()
        task = self.task_entry.get("1.0", "end").strip()
        if not username:
            messagebox.showwarning("Missing user name", "Please enter a user name.")
            return
        if not task:
            messagebox.showwarning("Missing task", "Please enter a writing task or purpose.")
            return

        self.output_text.delete("1.0", "end")
        self.log_text.delete("1.0", "end")
        self.interpreter_text.delete("1.0", "end")
        self.profile_text.delete("1.0", "end")
        self.other_text.delete("1.0", "end")
        self._clear_replacement_options()

        self.orchestrator.start_new_task(username=username, task=task)

    def _stop_streaming(self) -> None:
        self.orchestrator.stop_streaming()
        self._append_log("[UI] Stop requested.\n")

    def _accept_text(self) -> None:
        self.orchestrator.accept_current_text()
        self._append_log("[UI] Current text accepted as baseline.\n")

    def _continue_generation(self) -> None:
        self.orchestrator.continue_generation()

    def _apply_option(self) -> None:
        option_id = self.selected_option_var.get().strip()
        other_text = self.other_text.get("1.0", "end").strip()
        self.orchestrator.apply_selected_option(
            option_id=option_id,
            other_mode=self.other_mode_var.get().strip(),
            other_text=other_text,
        )

    def _export_json(self) -> None:
        payload = self.orchestrator.export_session_json()
        popup = tk.Toplevel(self.root)
        popup.title("Session JSON")
        popup.geometry("900x650")
        txt = tk.Text(popup, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", payload)

    def _clear_replacement_options(self) -> None:
        for child in self.options_frame.winfo_children():
            child.destroy()
        self.selected_option_var.set("")

    def _set_replacement_options(self, options: List[Dict[str, Any]]) -> None:
        self._clear_replacement_options()
        first_selectable_option = ""
        for option in options:
            title = f"{option['reason_id']} - {option['reason']}"
            if len(title) > 120:
                title = title[:117] + "..."
            explanation = option["explanation"]
            if len(explanation) > 140:
                explanation = explanation[:137] + "..."
            text = f"{title}\n{explanation}"
            if option["replacement_text"]:
                preview = option["replacement_text"]
                if len(preview) > 220:
                    preview = preview[:217] + "..."
                text += f"\nReplacement: {preview}"
            elif option.get("option_kind") in {"other_describe", "other_write"}:
                text += "\nUse the Other Input box below."
            rb = ttk.Radiobutton(
                self.options_frame,
                text=text,
                value=option["option_id"],
                variable=self.selected_option_var,
            )
            rb.pack(anchor="w", pady=4, fill="x")
            if not first_selectable_option and option.get("option_kind") != "other":
                first_selectable_option = option["option_id"]
        if first_selectable_option:
            self.selected_option_var.set(first_selectable_option)

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _poll_events(self) -> None:
        try:
            while True:
                event_type, payload = self.ui_queue.get_nowait()

                if event_type == "guidance":
                    self._append_log(f"[Guidance]\n{payload}\n\n")
                elif event_type == "credential_status":
                    self.credential_var.set(payload)
                    self._append_log(f"[Credential] {payload}\n")
                elif event_type == "status":
                    self.status_var.set(payload)
                    self._append_log(f"[Status] {payload}\n")
                elif event_type == "append_text":
                    self.output_text.insert("end", payload)
                    self.output_text.see("end")
                elif event_type == "set_text":
                    self.output_text.delete("1.0", "end")
                    self.output_text.insert("1.0", payload)
                    self.output_text.see("end")
                elif event_type == "interpreter_result":
                    self.interpreter_text.delete("1.0", "end")
                    self.interpreter_text.insert("1.0", self._format_interpreter_result(payload))
                elif event_type == "replacement_options":
                    self._set_replacement_options(payload)
                    self._append_log("[ReplacementHelper] Options updated.\n")
                elif event_type == "profile_update":
                    self.profile_text.delete("1.0", "end")
                    self.profile_text.insert("1.0", "\n".join(payload))
                elif event_type == "accepted_text":
                    self._append_log("[Accepted Baseline Updated]\n")
                elif event_type == "revision_applied":
                    self._append_log("[Revision Applied]\n")
                    self._append_log(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
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

    def _format_interpreter_result(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        stop_point = payload.get("stop_point", {})
        likely_intent = payload.get("likely_user_intent", "").strip()
        if likely_intent:
            lines.append(f"Likely intent: {likely_intent}")
        current_sentence = stop_point.get("current_sentence", "").strip()
        if current_sentence:
            lines.append(f"Interrupted sentence: {current_sentence}")
        last_sentence = stop_point.get("last_sentence", "").strip()
        if last_sentence:
            lines.append(f"Previous sentence: {last_sentence}")

        reasons = payload.get("reason_candidates", [])
        if reasons:
            lines.append("")
            lines.append("Interpreted revision reasons:")
            for item in reasons:
                reason_id = item.get("id", "").strip()
                reason = item.get("reason", "").strip()
                lines.append(f"{reason_id}: {reason}")

        guidance = payload.get("replacement_guidance", {})
        goal = guidance.get("goal", "").strip()
        if goal:
            lines.append("")
            lines.append(f"Replacement goal: {goal}")

        return "\n".join(lines)
