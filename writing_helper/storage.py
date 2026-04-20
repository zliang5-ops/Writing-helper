import json
import re
import time
from pathlib import Path
from typing import Dict, Tuple

from .models import InterruptionContext, RevisionEvent, UserProfile


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "local_data"
PROFILES_DIR = DATA_DIR / "profiles"
CREDENTIAL_LOG_PATH = DATA_DIR / "credential_log.json"


def _ensure_dirs() -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def _slugify_username(username: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", username.strip().lower()).strip("_")
    return slug or "user"


def _profile_path(username: str) -> Path:
    return PROFILES_DIR / f"{_slugify_username(username)}.json"


def _revision_event_from_dict(payload: Dict) -> RevisionEvent:
    stop_point_payload = payload.get("stop_point", {})
    stop_point = InterruptionContext(
        termination_point=stop_point_payload.get("termination_point", ""),
        last_sentence=stop_point_payload.get("last_sentence", ""),
        current_sentence=stop_point_payload.get("current_sentence", ""),
        replacement_start=stop_point_payload.get("replacement_start", 0),
    )
    return RevisionEvent(
        event_id=payload.get("event_id", ""),
        timestamp=payload.get("timestamp", 0.0),
        username=payload.get("username", ""),
        task=payload.get("task", ""),
        stop_point=stop_point,
        interpreter_result=payload.get("interpreter_result", {}),
        selected_reason_id=payload.get("selected_reason_id", ""),
        selected_reason=payload.get("selected_reason", ""),
        selected_revision=payload.get("selected_revision", ""),
        selection_kind=payload.get("selection_kind", ""),
        custom_input=payload.get("custom_input", ""),
        updated_preference_profile=list(payload.get("updated_preference_profile", [])),
    )


def load_or_create_user_profile(username: str) -> Tuple[UserProfile, bool]:
    _ensure_dirs()
    path = _profile_path(username)
    now = time.time()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = UserProfile(
            username=payload.get("username", username),
            preference_profile=list(payload.get("preference_profile", [])),
            revision_log=[_revision_event_from_dict(item) for item in payload.get("revision_log", [])],
            created_at=payload.get("created_at", now),
            updated_at=payload.get("updated_at", now),
        )
        _update_credential_log(profile.username, path, created_new=False)
        return profile, False

    profile = UserProfile(username=username.strip(), created_at=now, updated_at=now)
    save_user_profile(profile)
    _update_credential_log(profile.username, path, created_new=True)
    return profile, True


def save_user_profile(profile: UserProfile) -> None:
    _ensure_dirs()
    profile.updated_at = time.time()
    payload = {
        "username": profile.username,
        "preference_profile": list(profile.preference_profile),
        "revision_log": [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "username": event.username,
                "task": event.task,
                "stop_point": {
                    "termination_point": event.stop_point.termination_point,
                    "last_sentence": event.stop_point.last_sentence,
                    "current_sentence": event.stop_point.current_sentence,
                    "replacement_start": event.stop_point.replacement_start,
                },
                "interpreter_result": event.interpreter_result,
                "selected_reason_id": event.selected_reason_id,
                "selected_reason": event.selected_reason,
                "selected_revision": event.selected_revision,
                "selection_kind": event.selection_kind,
                "custom_input": event.custom_input,
                "updated_preference_profile": event.updated_preference_profile,
            }
            for event in profile.revision_log
        ],
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }
    _profile_path(profile.username).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _update_credential_log(profile.username, _profile_path(profile.username), created_new=False)


def _update_credential_log(username: str, profile_path: Path, created_new: bool) -> None:
    _ensure_dirs()
    payload = {"users": {}}
    if CREDENTIAL_LOG_PATH.exists():
        payload = json.loads(CREDENTIAL_LOG_PATH.read_text(encoding="utf-8"))
    users = payload.setdefault("users", {})
    users[username.strip().lower()] = {
        "username": username.strip(),
        "profile_path": str(profile_path),
        "last_seen": time.time(),
        "created_new": created_new,
    }
    CREDENTIAL_LOG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
