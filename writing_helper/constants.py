import re


SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|$)", re.S)
MAX_REASON_OPTIONS = 10
TARGET_REASON_OPTIONS = 5
