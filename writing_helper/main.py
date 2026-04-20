import os
import tkinter as tk

from .ui import WritingHelperApp


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment before running this script."
        )

    root = tk.Tk()
    WritingHelperApp(root)
    root.mainloop()
