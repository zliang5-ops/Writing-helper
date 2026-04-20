import json

from .constants import SENTENCE_PATTERN
from .models import InterruptionContext


def extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def extract_interruption_context(text: str) -> InterruptionContext:
    matches = list(SENTENCE_PATTERN.finditer(text))
    spans: list[tuple[int, int, str]] = []
    for match in matches:
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        offset = raw.find(stripped)
        start = match.start() + offset
        end = start + len(stripped)
        spans.append((start, end, stripped))

    if not spans:
        stripped = text.strip()
        return InterruptionContext(
            termination_point=stripped,
            last_sentence="",
            current_sentence=stripped,
            replacement_start=max(0, len(text) - len(stripped)),
        )

    current_start, _, current_sentence = spans[-1]
    last_sentence = spans[-2][2] if len(spans) > 1 else ""
    termination_point = text[current_start:].strip() or current_sentence
    return InterruptionContext(
        termination_point=termination_point,
        last_sentence=last_sentence,
        current_sentence=current_sentence,
        replacement_start=current_start,
    )
