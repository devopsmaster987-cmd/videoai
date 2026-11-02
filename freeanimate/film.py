"""Film mode — structured, resumable long-video production (5-25 minutes).

This is a separate production path from ``create``/``make``:

``freeanimate film plan``      prompt or script -> reviewed film plan
``freeanimate film preview``   one representative scene at preview settings
``freeanimate film render``    full film, resumable at scene granularity
``freeanimate film resume``    continue an interrupted film render
``freeanimate film status``    show plan, scene progress and estimates
``freeanimate film validate``  run output validation + automated QA checks

Everything is local and honest.  The spoken track is real measured speech from
installed offline TTS voices, the picture is the procedural 2D rig renderer, and
any capability that is unavailable (3D, neural video) is reported rather than
faked.  Scene boundaries are derived from *measured* audio timings, not from
guessed word counts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .animation2d import (ActorCue, CameraCue, CharacterPlacement,
                          DialogueSpan, SceneAnimation, SceneProp,
                          METHOD_NAME as ANIM_2D_METHOD, render_scene_animation)
from .audio_mix import DEFAULT_SAMPLE_RATE, mix_final_audio, wav_duration_seconds
from .backgrounds import EnvironmentSpec, render_cached
from .characters import Character, CharacterLibrary, ensure_project_characters
from .config import Config
from .constants import (DEFAULT_MUSIC_VOLUME, FILM_DEFAULT_DURATION_MIN,
                        FILM_DIALOGUE_AUDIO_SHARE, FILM_MAX_DURATION_MIN,
                        FILM_MAX_SCENES, FILM_MIN_DURATION_MIN,
                        FILM_PREVIEW_DEFAULT_SECONDS, FILM_TARGET_SCENE_SECONDS)
from .conversation import ConversationResult, RenderedLine, render_conversation
from .ffmpeg_utils import FFmpegError, has_filter, probe, run_ffmpeg
from .hardware import detect_hardware
from .project import (Project, STATUS_EXPORTED, STATUS_NARRATED,
                      STATUS_RENDERED, STATUS_SCRIPT_READY, resolve_project)
from .prompt_understanding import (CharacterSpec, DialogueLine, PromptPlan,
                                   _detect_environment, understand_prompt)
from .renderer import concat_segments
from .script_engine import (DEFAULT_WPM, Script, ScriptSection,
                            TemplateScriptEngine, _split_sentences, _words,
                            compute_word_budget, extract_topic, parse_user_script)
from .studio import Studio, _resolution_dims, l_visemes
from .subtitles import (SubtitleCue, SubtitleTrack, wrap_lines, write_subtitles)
from .validation import validate_output, write_report

ProgressFn = Callable[[str, float, str | None], None]
CancelFn = Callable[[], bool]

NARRATOR_ID = "narrator"
NARRATOR_NAME = "Narrator"

FILM_QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "pc_friendly": {
        "resolution": "480p", "crf": 30, "preset": "ultrafast",
        "description": "Lowest resource use: 480p, ultrafast x264, fast previews.",
    },
    "balanced": {
        "resolution": "720p", "crf": 26, "preset": "veryfast",
        "description": "720p with veryfast x264: the practical default for an 8 GB CPU laptop.",
    },
    "high": {
        "resolution": "1080p", "crf": 22, "preset": "faster",
        "description": "1080p with faster x264. Slower on CPU; still 2D rig animation, "
                       "not higher character detail.",
    },
}


# ---------------------------------------------------------------------------
# Duration limits (section 15)
# ---------------------------------------------------------------------------
def clamp_film_minutes(minutes: float | None) -> tuple[float, list[str]]:
    """Clamp a requested film length into [5, 25] minutes with honest notes.

    ``None`` selects the default of 15 minutes.  Values below the minimum are
    raised and values above the maximum are lowered; both are reported so the
    caller never silently produces a 40-minute video.
    """
    notes: list[str] = []
    if minutes is None:
        return float(FILM_DEFAULT_DURATION_MIN), notes
    try:
        value = float(minutes)
    except (TypeError, ValueError):
        notes.append(f"Duration {minutes!r} is not a number; the default "
                     f"{FILM_DEFAULT_DURATION_MIN:g} minutes is used.")
        return float(FILM_DEFAULT_DURATION_MIN), notes
    if value < FILM_MIN_DURATION_MIN:
        notes.append(f"Requested {value:g} min is below the long-video minimum; "
                     f"it was raised to {FILM_MIN_DURATION_MIN:g} min.")
        return float(FILM_MIN_DURATION_MIN), notes
    if value > FILM_MAX_DURATION_MIN:
        notes.append(f"Requested {value:g} min exceeds the supported maximum; "
                     f"it was lowered to {FILM_MAX_DURATION_MIN:g} min. Longer films are "
                     "not generated under the default configuration.")
        return float(FILM_MAX_DURATION_MIN), notes
    return value, notes


def _estimate_line_seconds(text: str, wpm: int) -> float:
    return _words(text) / max(60, int(wpm)) * 60.0 + 0.45


def _film_subtitle_track(lines: list[RenderedLine], language: str,
                         max_chars: int, max_lines: int) -> SubtitleTrack:
    """Subtitles for a film: label on-screen speakers, never the narrator."""
    speakers = {l.speaker_id for l in lines}
    cues: list[SubtitleCue] = []
    for i, line in enumerate(lines):
        label = ""
        if len(speakers) > 1 and line.speaker_id != NARRATOR_ID:
            label = f"{line.speaker_name}: "
        cues.append(SubtitleCue(
            i, line.start, line.end,
            wrap_lines(label + line.text, max_chars=max_chars, max_lines=max_lines)))
    return SubtitleTrack(cues=cues, language=language,
                         timing_source="audio_measured")


def _fingerprint(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
@dataclass
class FilmOptions:
    title: str | None = None
    duration_minutes: float = FILM_DEFAULT_DURATION_MIN
    quality: str = "balanced"                 # pc_friendly | balanced | high
    resolution: str | None = None             # explicit override
    fps: int = 24
    language: str = "en"
    script_path: str | None = None
    script_text: str | None = None
    music: str = "none"                       # none | procedural | <path>
    music_volume: float = DEFAULT_MUSIC_VOLUME
    burn_subtitles: bool = True
    crf: int | None = None
    preset: str | None = None
    tts_engine: str = "auto"
    tts_voice: str | None = None
    tts_speed: float = 1.0
    aspect_ratio: str = "16:9"
    seed: str = ""
    characters_library: str | None = None
    preview_seconds: float = FILM_PREVIEW_DEFAULT_SECONDS
    target_scene_seconds: float = FILM_TARGET_SCENE_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render_settings(self) -> tuple[str, int, str]:
        base = FILM_QUALITY_PRESETS.get(self.quality, FILM_QUALITY_PRESETS["balanced"])
        resolution = self.resolution or str(base["resolution"])
        crf = self.crf if self.crf is not None else int(base["crf"])
        preset = self.preset or str(base["preset"])
        return resolution, crf, preset


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------
@dataclass
class FilmScene:
    index: int
    scene_id: str
    title: str
    purpose: str = ""
    environment: str = ""
    time_of_day: str = ""
    duration_planned: float = 0.0
    characters: list[str] = field(default_factory=list)
    lines: list[DialogueLine] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    props: list[dict[str, Any]] = field(default_factory=list)
    continuity: list[str] = field(default_factory=list)
    start: float = -1.0
    end: float = -1.0
    duration_actual: float = 0.0
    status: str = "planned"
    segment: str = ""
    background: str = ""
    frames: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lines"] = [ln.model_dump() for ln in self.lines]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilmScene":
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in known}
        values["lines"] = [DialogueLine.model_validate(x) for x in data.get("lines", [])]
        return cls(**values)


@dataclass
class FilmPlan:
    prompt: str
    title: str
    project_type: str = "story"
    environment_type: str = "generic"
    aspect_ratio: str = "16:9"
    language: str = "en"
    quality: str = "balanced"
    target_minutes: float = FILM_DEFAULT_DURATION_MIN
    target_seconds: float = FILM_DEFAULT_DURATION_MIN * 60.0
    wpm: int = DEFAULT_WPM
    estimated_seconds: float = 0.0
    characters: list[CharacterSpec] = field(default_factory=list)
    scenes: list[FilmScene] = field(default_factory=list)
    script: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "title": self.title,
            "project_type": self.project_type,
            "environment_type": self.environment_type,
            "aspect_ratio": self.aspect_ratio,
            "language": self.language,
            "quality": self.quality,
            "target_minutes": self.target_minutes,
            "target_seconds": self.target_seconds,
            "wpm": self.wpm,
            "estimated_seconds": self.estimated_seconds,
            "characters": [c.model_dump() for c in self.characters],
            "scenes": [s.to_dict() for s in self.scenes],
            "script": self.script,
            "notes": self.notes,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilmPlan":
        return cls(
            prompt=data.get("prompt", ""),
            title=data.get("title", "Untitled"),
            project_type=data.get("project_type", "story"),
            environment_type=data.get("environment_type", "generic"),
            aspect_ratio=data.get("aspect_ratio", "16:9"),
            language=data.get("language", "en"),
            quality=data.get("quality", "balanced"),
            target_minutes=float(data.get("target_minutes") or FILM_DEFAULT_DURATION_MIN),
            target_seconds=float(data.get("target_seconds") or 15 * 60),
            wpm=int(data.get("wpm") or DEFAULT_WPM),
            estimated_seconds=float(data.get("estimated_seconds") or 0.0),
            characters=[CharacterSpec.model_validate(c) for c in data.get("characters", [])],
            scenes=[FilmScene.from_dict(s) for s in data.get("scenes", [])],
            script=dict(data.get("script") or {}),
            notes=list(data.get("notes") or []),
            warnings=list(data.get("warnings") or []),
            limitations=list(data.get("limitations") or []),
        )

    @property
    def scene_count(self) -> int:
        return len(self.scenes)


@dataclass
class FilmResult:
    ok: bool
    project: Project | None = None
    output: Path | None = None
    thumbnail: Path | None = None
    plan: FilmPlan | None = None
    conversation: ConversationResult | None = None
    validation: Any = None
    qa: Any = None
    report_json: Path | None = None
    report_txt: Path | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    perception: dict[str, Any] | None = None
    duration: float = 0.0
    scene_count: int = 0
    estimate_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project": str(self.project.root) if self.project else None,
            "project_slug": self.project.manifest.get("slug") if self.project else None,
            "output": str(self.output) if self.output else None,
            "thumbnail": str(self.thumbnail) if self.thumbnail else None,
            "duration": round(self.duration, 2),
            "scene_count": self.scene_count,
            "estimate_seconds": round(self.estimate_seconds, 1),
            "notes": self.notes,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "error": self.error,
            "validation_ok": getattr(self.validation, "ok", None),
            "qa_ok": getattr(self.qa, "ok", None),
            "report_json": str(self.report_json) if self.report_json else None,
            "report_txt": str(self.report_txt) if self.report_txt else None,
        }


# ---------------------------------------------------------------------------
# Director: text -> actions, props and camera cues
# ---------------------------------------------------------------------------
_ACTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bapproach(es|ed|ing)?\b|\bcomes? (over|closer)\b|\bgoes? to\b", "approach"),
    (r"\bwalk(s|ed|ing)?\b|\bstroll(s|ed|ing)?\b|\bwanders?\b", "walk"),
    (r"\brun(s|ning)?\b|\bsprint(s|ed|ing)?\b|\bchase[sd]?\b|\brates?\b", "run"),
    (r"\bturn(s|ed|ing)?\b", "turn"),
    (r"\bjump(s|ed|ing)?\b|\bleap(s|ed|ing)?\b", "jump"),
    (r"\bwave[sd]?\b|\bwaving\b", "wave"),
    (r"\bpoint(s|ed|ing)?\b", "point"),
    (r"\blaugh(s|ed|ing)?\b|\bgiggle[sd]?\b", "laugh"),
    (r"\bclap(s|ped|ping)?\b|\bapplaud(s|ed|ing)?\b", "clap"),
    (r"\bdance[sd]?\b|\bdancing\b", "dance"),
    (r"\bgreet(s|ed|ing)?\b|\bwave hello\b|\bsays? hello\b", "greet"),
    (r"\bleav(e|es|ing)\b|\bwalk(s|ed)? away\b|\bdepart(s|ed|ing)?\b", "leave"),
    (r"\bfollow(s|ed|ing)?\b", "follow"),
    (r"\bsit(s|ting)?\b|\bseated\b", "sit"),
    (r"\bstand(s|ing)?\b|\bgets? up\b", "stand"),
    (r"\breach(es|ed|ing)?\b", "reach"),
    (r"\bpick(s|ed|ing)? up\b|\bgrabs?\b|\blifts?\b", "pick_up"),
    (r"\bput(s|ting)? down\b|\bsets? down\b", "put_down"),
    (r"\bthrow(s|ing)?\b|\btoss(es|ed|ing)?\b", "throw"),
    (r"\bcatch(es|ing)?\b", "catch"),
    (r"\bkick(s|ed|ing)?\b|\broll(s|ed|ing)?\b|\bbounc(e|es|ed|ing)\b", "play"),
    (r"\bplay(s|ed|ing)?\b", "play"),
    (r"\breact(s|ed|ing)?\b|\bsurprised\b|\bamazed\b", "react"),
    (r"\blisten(s|ed|ing)?\b", "listen"),
    (r"\bspeak(s|ing)?\b|\btalk(s|ing)?\b|\bsays?\b", "speak"),
)

_PROP_NOUNS: dict[str, str] = {
    "soccer ball": "ball", "beach ball": "ball", "football": "ball", "ball": "ball",
    "book": "book", "storybook": "book", "chair": "chair", "stool": "stool",
    "table": "table", "desk": "table", "tree": "tree", "flower": "flower",
    "plant": "plant", "bush": "plant", "phone": "phone", "cup": "cup", "mug": "cup",
    "gift": "gift", "present": "gift", "box": "box", "crate": "box",
    "bottle": "bottle", "car": "vehicle", "bike": "vehicle", "bicycle": "vehicle",
    "door": "door", "window": "window", "lantern": "lantern", "lamp": "light",
}

_PROP_LAYOUT: dict[str, tuple[float, float, float]] = {
    # kind -> (x fraction, y fraction, scale)
    "tree": (0.86, 0.82, 1.8), "plant": (0.14, 0.90, 0.9),
    "flower": (0.20, 0.93, 0.55), "ball": (0.50, 0.94, 0.42),
    "book": (0.66, 0.92, 0.45), "chair": (0.82, 0.93, 0.95),
    "stool": (0.18, 0.93, 0.75), "table": (0.52, 0.92, 1.25),
    "phone": (0.62, 0.92, 0.4), "cup": (0.70, 0.92, 0.38),
    "gift": (0.30, 0.93, 0.55), "box": (0.74, 0.93, 0.6),
    "bottle": (0.26, 0.93, 0.5), "vehicle": (0.80, 0.90, 1.4),
    "door": (0.88, 0.85, 1.5), "window": (0.12, 0.60, 1.2),
    "lantern": (0.10, 0.78, 0.6), "light": (0.90, 0.72, 0.6),
}

_OBJECT_VERBS = re.compile(
    r"\b(roll(s|ed|ing)?|kick(s|ed|ing)?|bounc(e|es|ed|ing)|throw(s|ing)?|"
    r"toss(es|ed|ing)?|catch(es|ing)?|pass(es|ed|ing)?|play(s|ed|ing)?)\b",
    re.IGNORECASE)

_ALLOWED_ACTIONS = {
    "idle", "walk", "run", "approach", "leave", "follow", "chase", "play", "turn",
    "jump", "wave", "point", "laugh", "clap", "dance", "greet", "sit", "stand",
    "reach", "pick_up", "put_down", "throw", "catch", "react", "listen", "speak",
}


def _find_actions(text: str) -> list[str]:
    """Return action verbs detected in *text*, in order of first appearance."""
    found: list[tuple[int, str]] = []
    low = (text or "").lower()
    for pattern, action in _ACTION_PATTERNS:
        m = re.search(pattern, low)
        if m:
            found.append((m.start(), action))
    found.sort()
    out: list[str] = []
    for _pos, action in found:
        if action not in out:
            out.append(action)
    return out


def _find_props(text: str) -> list[str]:
    low = (text or "").lower()
    found: list[tuple[int, str]] = []
    for noun in sorted(_PROP_NOUNS, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(noun)}s?\b", low)
        if m:
            found.append((m.start(), _PROP_NOUNS[noun]))
    found.sort()
    out: list[str] = []
    for _pos, kind in found:
        if kind not in out:
            out.append(kind)
    return out


def _character_x(i: int, n: int) -> float:
    return 0.5 if n <= 1 else 0.32 + 0.36 * i / (n - 1)


def direct_scene(
    index: int,
    text: str,
    scene_characters: list[str],
    *,
    on_screen: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn scene text into fractional action cues and prop placements.

    All times are fractions of the final (measured) scene duration so the cues
    stay aligned when the real speech length differs from the estimate.
    """
    actors = [c for c in scene_characters if c in on_screen] or list(on_screen)
    if not actors:
        return [], []
    n = len(actors)
    actions = _find_actions(text)
    cues: list[dict[str, Any]] = []
    slots = max(1, min(6, len(actions)))
    per_slot = 0.72 / slots
    for a_i, action in enumerate(actions[:12]):
        actor = actors[a_i % n]
        slot = a_i // n
        t0 = 0.12 + slot * per_slot + (a_i % n) * 0.015
        t1 = min(0.94, t0 + per_slot * 0.95)
        if t1 <= t0:
            t1 = min(1.0, t0 + 0.08)
        cue: dict[str, Any] = {
            "character_id": actor, "action": action,
            "t0": round(t0, 4), "t1": round(t1, 4),
        }
        base_x = _character_x(actors.index(actor), n)
        other = next((a for a in actors if a != actor), None)
        if action in ("walk", "run", "turn", "approach", "leave", "follow"):
            if action in ("approach", "follow") and other:
                target_x = _character_x(actors.index(other), n)
                # Stop on the actor's own side of the target instead of
                # walking through them.
                if base_x >= target_x:
                    x_to = min(0.96, target_x + 0.12)
                    x_from = min(0.96, base_x + 0.05)
                else:
                    x_to = max(0.04, target_x - 0.12)
                    x_from = max(0.04, base_x - 0.05)
                cue.update({"x_from": round(x_from, 4), "x_to": round(x_to, 4),
                            "target_character": other})
            elif action == "leave":
                cue.update({"x_from": round(base_x, 4),
                            "x_to": round(0.04 if base_x < 0.5 else 0.96, 4)})
            elif action == "walk":
                direction = -1 if base_x >= 0.5 else 1
                cue.update({"x_from": round(base_x, 4),
                            "x_to": round(max(0.05, min(0.95, base_x + direction * 0.14)), 4)})
            else:
                step = 0.12 * (-1 if base_x >= 0.5 else 1)
                cue.update({"x_from": round(base_x - step, 4),
                            "x_to": round(base_x, 4)})
        if action in ("greet", "wave", "point", "throw", "catch", "play") and other:
            cue["target_character"] = other
        cues.append(cue)

    props: list[dict[str, Any]] = []
    prop_kinds = _find_props(text)
    object_motion = bool(_OBJECT_VERBS.search(text or ""))
    for p_i, kind in enumerate(prop_kinds[:5]):
        x, y, scale = _PROP_LAYOUT.get(kind, (0.22 + 0.16 * p_i, 0.93, 0.6))
        entry: dict[str, Any] = {
            "id": f"{kind}_{p_i + 1}", "kind": kind,
            "x": round(x, 3), "y": round(y, 3), "scale": round(scale, 2),
            "t0": 0.0, "t1": 0.0, "action": "static",
        }
        if kind == "ball" and object_motion:
            entry.update({"action": "roll", "t0": 0.15, "t1": 0.85,
                          "x": 0.30, "x_to": 0.72, "y": 0.94, "y_to": 0.94})
            carrier = next((c for c in cues if c.get("action") == "play"), None)
            if carrier:
                carrier["prop"] = f"ball_{p_i + 1}"
        elif object_motion and kind in ("box", "gift", "book"):
            entry.update({"t0": 0.2, "t1": 0.8, "action": "static"})
        props.append(entry)
    if index == 0 and not cues:
        # Opening scene: a gentle entrance keeps the first frames alive.
        for a_i, actor in enumerate(actors[:3]):
            base_x = _character_x(a_i, max(3, n))
            cues.append({
                "character_id": actor, "action": "walk",
                "t0": 0.02 + a_i * 0.03, "t1": 0.16 + a_i * 0.03,
                "x_from": round(max(0.04, base_x - 0.18), 4),
                "x_to": round(base_x, 4),
            })
    return cues, props


def build_camera_cues(
    lines: list[tuple[float, float, str]],
    duration: float,
    *,
    n_chars: int,
    movement: list[tuple[float, float, str]] | None = None,
) -> list[CameraCue]:
    """Build a non-overlapping shot list that covers the whole scene.

    ``lines`` are ``(start, end, speaker_id)`` in scene-local seconds.  The
    sequence opens with an establishing shot, follows the speaker, inserts
    reaction shots in longer pauses, tracks real movement, and closes wide.
    """
    if duration <= 0.05:
        return []
    cues: list[CameraCue] = []
    open_end = min(duration, 3.2 if duration > 8.0 else duration * 0.35)
    cues.append(CameraCue("establishing", "push_in", 0.0, open_end,
                          focus=None, cut=True))
    cursor = open_end
    movement_cycle = ("push_in", "pan_right", "pan_left", "push_in", "slow_zoom_center")

    ordered = sorted((l for l in lines if l[1] > l[0]), key=lambda l: l[0])
    for i, (start, end, speaker) in enumerate(ordered):
        start = max(cursor, start)
        end = min(duration, end)
        if end - start < 0.6:
            continue
        if start > cursor + 1.0:
            cues.append(CameraCue("wide", "pan_right", cursor, start,
                                  focus=None, cut=True))
        focus = None if speaker == NARRATOR_ID else speaker
        if n_chars >= 3:
            kind = "medium"
        else:
            kind = "close_up" if (end - start) > 3.4 else "medium"
        cues.append(CameraCue(kind, movement_cycle[i % len(movement_cycle)],
                              start, end, focus=focus, cut=True))
        cursor = max(cursor, end)

    for start, end, actor in (movement or []):
        start = max(0.0, min(duration - 0.4, start))
        end = max(start + 0.4, min(duration, end))
        cues.append(CameraCue("tracking", "pan_right", start, end,
                              focus=actor, cut=True))

    if cursor < duration - 0.25:
        cues.append(CameraCue("wide", "pull_out", cursor, duration,
                              focus=None, cut=True))

    cues.sort(key=lambda c: c.start)
    filled: list[CameraCue] = []
    cursor = 0.0
    for cue in cues:
        if cue.end <= cursor + 0.02:
            continue
        if cue.start > cursor + 0.02:
            # Short pauses hold the current shot instead of flashing to a
            # one-frame wide shot between lines.
            if cue.start - cursor < 1.0 and filled:
                filled[-1].end = cue.start
            else:
                filled.append(CameraCue("wide", "pan_right", cursor, cue.start,
                                        focus=None, cut=True))
        cue.start = max(cue.start, cursor)
        cue.end = min(duration, max(cue.end, cue.start + 0.1))
        if filled and abs(filled[-1].start - cue.start) < 0.02:
            # Same start: the later cue wins for the whole span.
            filled[-1] = cue
        else:
            filled.append(cue)
        cursor = cue.end
    if cursor < duration - 0.02:
        filled.append(CameraCue("wide", "pull_out", cursor, duration,
                                focus=None, cut=True))
    return filled


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def _calibrate_wpm(language: str, config: Config, cache_root: Path) -> tuple[int, str | None]:
    """Measure the installed voice's real speaking rate with a short synthesis.

    Returns ``(wpm, note)``.  Falls back to the configured default when no
    local voice can synthesize the calibration line.
    """
    fallback = int(config.data.get("defaults", {}).get("wpm", DEFAULT_WPM))
    cache_dir = cache_root / "film_calibration"
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe_wav = cache_dir / f"calibration_{language}.wav"
    sentence = ("This sentence measures the real speaking pace of the installed voice "
                "so the film reaches its requested length honestly.")
    if not probe_wav.exists() or probe_wav.stat().st_size < 1024:
        try:
            from .tts import synthesize_with_fallback
            res = synthesize_with_fallback(sentence, probe_wav, preferred="auto",
                                           language=language, config={})
            if not res.ok or res.audio_path is None:
                return fallback, None
        except Exception:  # noqa: BLE001 — calibration is best-effort
            return fallback, None
    try:
        seconds = wav_duration_seconds(probe_wav)
    except Exception:  # noqa: BLE001
        return fallback, None
    if seconds <= 0.2:
        return fallback, None
    wpm = int(round(_words(sentence) / seconds * 60.0))
    wpm = max(80, min(230, wpm))
    if abs(wpm - fallback) > 8:
        return wpm, (f"Speech pace was calibrated to about {wpm} words/minute with the "
                     f"installed voice (configured default is {fallback}); the script word "
                     "budget uses the measured pace so the film length is realistic.")
    return wpm, None


def _split_section_text(text: str, max_words: int = 26) -> list[str]:
    """Split narration into speakable lines of at most ~``max_words`` words."""
    sentences = _split_sentences(text)
    out: list[str] = []
    buf: list[str] = []
    for sentence in sentences:
        buf.append(sentence)
        if _words(" ".join(buf)) >= max_words or _words(sentence) > max_words:
            out.append(" ".join(buf).strip())
            buf = []
    if buf:
        out.append(" ".join(buf).strip())
    return [s for s in out if s]


# ---------------------------------------------------------------------------
# Offline story writing
# ---------------------------------------------------------------------------
# Film prompts are usually stories ("Lily and Ben explore a garden..."), not
# documentary topics.  The documentary template engine would ignore the plot
# and narrate generic filler about a single noun, so film mode composes an
# on-topic story offline instead: the prompt supplies the plot beats, and
# deterministic prose expands them with the detected characters, actions,
# props and setting.  No external model is called and no facts are invented.

_STORY_VERB_RE = re.compile(
    r"\b(?:explore|play|find|chase|sit|run|walk|jump|dance|swim|build|help|meet|"
    r"wave|laugh|sing|climb|hide|search|discover|watch|eat|cook|read|paint|draw|"
    r"fly|flies|ride|visit|travel|look|talk|speak|smile|open|give|take|hold|carry|"
    r"bring|follow|escape|rescue|celebrate|wait|listen|cry|fall|stand|return|"
    r"wake|sleep|start|stop)(?:s|ed|ing)?\b",
    re.IGNORECASE,
)

_NAME_VERB_RE = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:[a-z]+\s+)?(?:walks?|runs?|plays?|waves?|laughs?|"
    r"dances?|sits?|smiles?|jumps?|sings?|swims?|looks?|says?|shouts?|explores?|"
    r"finds?|meets?|lives?|wants?|loves?|holds?|carries?|eats?|reads?|builds?|"
    r"chases?|climbs?|falls?|stands?|watches?|talks?|speaks?|approaches?|leaves?|"
    r"gives?|takes?|helps?|asks?|answers?|listens?|cries?|sleeps?|wakes?|opens?|"
    r"closes?|throws?|catches?|searches?|discovers?|flies?|rides?|visits?|"
    r"travels?|follows?|returns?)\b"
)

_NAME_STOPWORDS = {
    "They", "Them", "Their", "There", "Then", "Than", "That", "This", "These",
    "Those", "When", "While", "After", "Before", "Once", "Soon", "Later",
    "Meanwhile", "Finally", "Everyone", "Someone", "Nobody", "Inside", "Outside",
    "Under", "Over", "Around", "Through", "Across", "Hello", "Goodbye", "And",
    "But", "For", "With", "From", "Into", "Onto", "Today", "Tonight", "Tomorrow",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Once", "Some", "Many", "All", "Most", "Each", "Every",
}

_STORY_DETAIL: dict[str, list[str]] = {
    "find": [
        "{a} stops mid-step and points at {prop}.",
        "{a} kneels down to look more closely at {prop}.",
    ],
    "explore": [
        "Every path through the {place} feels new to {them}.",
        "{they} wander deeper into the {place}, curious about what waits ahead.",
    ],
    "play": [
        "{they} take turns, laughing louder each time.",
        "The game swells and slows, and neither of them wants it to end.",
    ],
    "chase": [
        "{a} darts ahead while {b} follows close behind.",
        "Their footsteps scatter across the {place} as the chase loops back around.",
    ],
    "run": [
        "{a} races ahead, arms pumping, and {b} gives chase.",
        "The run carries {them} deeper into the {place}.",
    ],
    "walk": [
        "{they} walk side by side, unhurried.",
        "The walk takes {them} deeper into the {place}.",
    ],
    "sit": [
        "{they} settle down together and let the quiet catch up.",
        "Sitting still, {they} notice details they had hurried past before.",
    ],
    "dance": [
        "{they} spin and sway to a rhythm only they can hear.",
        "The dance loosens into laughter, then into another round.",
    ],
    "wave": [
        "{a} lifts a hand in a wide, cheerful wave.",
        "The wave turns into a grin that spreads to {b}.",
    ],
    "laugh": [
        "Laughter rings out across the {place}.",
        "{names} laugh until their sides ache, then start all over again.",
    ],
    "look": [
        "{a} tilts their head and studies {prop}.",
        "For a moment {they} simply look, saying nothing.",
    ],
    "watch": [
        "{they} watch as the light shifts over the {place}.",
        "Nobody speaks; watching is enough.",
    ],
    "jump": [
        "{a} leaps, and lands with a satisfied bounce.",
        "{they} jump again and again, each landing a small celebration.",
    ],
    "swim": [
        "{they} cut through the water, leaving bright ripples behind.",
        "The water carries {them} along, cool and steady.",
    ],
    "sing": [
        "{a} starts humming, and the tune spreads to {b}.",
        "The song drifts over the {place}, half melody and half laughter.",
    ],
    "climb": [
        "{a} scrambles up first and reaches back for {b}.",
        "The climb is harder than it looked, which makes the view better.",
    ],
    "hide": [
        "{a} ducks behind cover and goes quiet.",
        "The hiding spot is not very good, which is the point.",
    ],
    "search": [
        "{they} search high and low, checking every corner.",
        "The search is slow, but nobody suggests giving up.",
    ],
    "discover": [
        "{a} pushes forward and discovers something no one expected.",
        "What {they} discover changes the shape of the afternoon.",
    ],
    "eat": [
        "{they} share what they have, passing it back and forth.",
        "The food is simple, and it tastes better for being shared.",
    ],
    "read": [
        "{a} reads aloud while {b} leans in to listen.",
        "The page turns, and the story pulls them both along.",
    ],
    "build": [
        "{they} fit the pieces together, adjusting and rebuilding.",
        "The build wobbles, holds, and earns a cheer.",
    ],
    "help": [
        "{a} steps in to help without being asked.",
        "{b} nods thanks, and the work goes faster.",
    ],
    "meet": [
        "{a} and {b} meet in the middle of the {place}.",
        "The meeting starts shyly and ends in easy conversation.",
    ],
    "fly": [
        "{they} watch the wind lift everything light enough to rise.",
        "For a moment it feels as if they could rise too.",
    ],
    "ride": [
        "{they} ride on, leaning into every turn.",
        "The ride smooths out, and the {place} slides past.",
    ],
    "talk": [
        "{a} and {b} fall into an easy conversation.",
        "They talk the way old friends do, skipping the small talk.",
    ],
    "speak": [
        "{a} speaks first, then waits.",
        "{b} answers, and the conversation finds its rhythm.",
    ],
    "cook": [
        "{they} work side by side, tasting and adjusting.",
        "The kitchen fills with a smell that makes the wait worth it.",
    ],
    "draw": [
        "{a} draws while {b} offers suggestions.",
        "The drawing takes shape slowly, one confident line at a time.",
    ],
    "paint": [
        "{they} paint with more enthusiasm than precision.",
        "The colours run together, and nobody minds.",
    ],
    "travel": [
        "{they} keep moving, trading stories as the way unfolds.",
        "The road bends, and the {place} opens up ahead.",
    ],
    "visit": [
        "{they} arrive and are welcomed in.",
        "The visit stretches on comfortably, the way good visits do.",
    ],
    "follow": [
        "{a} follows, keeping a few steps behind.",
        "Following turns into walking together.",
    ],
    "celebrate": [
        "{they} cheer, and the sound carries across the {place}.",
        "The celebration is small, and it is entirely theirs.",
    ],
    "return": [
        "{they} turn back toward where they started.",
        "The way home feels shorter than the way out.",
    ],
    "_default": [
        "{they} move deeper into the {place}, taking their time.",
        "The moment holds, quiet and full.",
    ],
}

_STORY_SENSORY = (
    "Around them, {sight}.",
    "Not far away, {sound}.",
    "The air is {air}.",
    "Every so often, {sound}.",
)

_STORY_TRANSITIONS = (
    "The afternoon slips by without anyone counting the minutes.",
    "One moment folds into the next, easy and unhurried.",
    "Time passes the way it does when nothing needs to be rushed.",
    "Soon the light begins to change, and the day leans toward evening.",
)

_STORY_DIALOGUE: dict[str, list[tuple[str, str]]] = {
    "find": [("Look! Over here!", "Oh, it is wonderful."),
             ("Did you see that?", "I saw it too.")],
    "explore": [("What is past those trees?", "Only one way to find out."),
                ("This place goes on forever.", "Let us keep going.")],
    "play": [("This is the best game.", "Do not stop now!"),
             ("Your turn!", "Watch this.")],
    "chase": [("You cannot catch me!", "Watch me try!"),
              ("Not so fast!", "Too late!")],
    "run": [("Wait for me!", "Keep up!"),
            ("I am winning!", "Not for long!")],
    "walk": [("It is so quiet here.", "I like it that way.")],
    "sit": [("The sky is beautiful.", "Let us stay a little longer."),
            ("I could sit here all evening.", "Me too.")],
    "dance": [("Dance with me!", "I am trying!"),
              ("Spin around!", "I am dizzy!")],
    "wave": [("Over here!", "I see you!"),
             ("Hello!", "Hello back!")],
    "laugh": [("I cannot breathe!", "That was too funny.")],
    "look": [("Do you see it?", "Right there.")],
    "watch": [("It is changing colour.", "I know. Look at that.")],
    "swim": [("The water is perfect.", "Race you to the far side!")],
    "sing": [("Sing it again!", "Only if you join in.")],
    "climb": [("Almost there!", "Do not look down!")],
    "hide": [("Ready or not!", "Not yet!")],
    "search": [("Any luck?", "Not yet. Keep looking.")],
    "discover": [("What is that?", "I have never seen anything like it.")],
    "eat": [("Try this.", "That is delicious.")],
    "read": [("One more page?", "Just one.")],
    "build": [("It is standing!", "Careful, do not touch it.")],
    "help": [("Let me help.", "Thanks, that is better.")],
    "meet": [("I am glad you came.", "I would not miss it.")],
    "talk": [("Tell me everything.", "Where do I start?")],
    "celebrate": [("We did it!", "We really did!")],
    "_default": [("Come on, let us go!", "I am right behind you."),
                 ("Look at this!", "I am looking!")],
}

_ENV_STORY: dict[str, dict[str, Any]] = {
    "garden": {
        "place": "garden",
        "sights": ("flowers nod in the breeze", "bees drift between the blossoms",
                   "sunlight falls through the leaves", "petals scatter on the grass"),
        "sounds": ("birds call from the trees", "leaves rustle softly",
                   "water trickles over the stones"),
        "air": "warm and sweet",
    },
    "forest": {
        "place": "forest",
        "sights": ("light filters down through the canopy", "ferns uncurl in the shade",
                   "moss covers the old stones", "branches creak high overhead"),
        "sounds": ("birds answer each other", "the wind moves through the pines",
                   "twigs snap underfoot"),
        "air": "cool and green",
    },
    "beach": {
        "place": "beach",
        "sights": ("waves fold onto the sand", "gulls wheel above the water",
                   "the tide leaves bright lines behind", "shells glint in the sun"),
        "sounds": ("the surf rolls in and out", "gulls call across the water",
                   "the water hisses over the sand"),
        "air": "salty and bright",
    },
    "park": {
        "place": "park",
        "sights": ("trees throw long shadows", "a path curves past the benches",
                   "clouds drift overhead", "grass sways in the wind"),
        "sounds": ("people talk on the path", "birds chatter in the branches",
                   "leaves whisper overhead"),
        "air": "fresh and easy",
    },
    "mountains": {
        "place": "mountains",
        "sights": ("peaks rise against the sky", "clouds catch on the ridges",
                   "the valley spreads out below", "stone glows in the light"),
        "sounds": ("the wind sweeps across the slopes", "a distant bird calls",
                   "loose stones clatter downhill"),
        "air": "thin and cold",
    },
    "city": {
        "place": "city",
        "sights": ("windows catch the light", "traffic moves in slow waves",
                   "signs hum above the street", "people cross in every direction"),
        "sounds": ("buses sigh at the corner", "footsteps echo on the pavement",
                   "a radio plays from an open window"),
        "air": "busy and warm",
    },
    "room": {
        "place": "room",
        "sights": ("light falls across the floor", "curtains move at the window",
                   "dust turns slowly in the sun", "the furniture holds the quiet"),
        "sounds": ("a clock ticks steadily", "the floorboards settle",
                   "traffic murmurs beyond the glass"),
        "air": "still and warm",
    },
    "kitchen": {
        "place": "kitchen",
        "sights": ("steam curls above the counter", "utensils catch the light",
                   "the window fogs at the edges", "bowls wait in a row"),
        "sounds": ("something simmers on the stove", "a tap drips once",
                   "drawers slide open and shut"),
        "air": "warm and fragrant",
    },
    "snow": {
        "place": "snowfield",
        "sights": ("snow fills every footprint", "breath hangs in the air",
                   "the light is white and wide", "branches carry thin lines of snow"),
        "sounds": ("snow creaks underfoot", "the wind stays very far away",
                   "everything is muffled and quiet"),
        "air": "sharp and cold",
    },
    "space": {
        "place": "open space",
        "sights": ("stars hold perfectly still", "a planet turns slowly below",
                   "light bends around the hull", "the dark goes on forever"),
        "sounds": ("only the ship hums", "the radio clicks softly",
                   "silence presses in"),
        "air": "recycled and cool",
    },
    "underwater": {
        "place": "water",
        "sights": ("light ripples across the sand", "fish flick between the rocks",
                   "bubbles rise in slow columns", "the surface shines far above"),
        "sounds": ("water moves in long pulses", "something clicks in the distance",
                   "the deep hums quietly"),
        "air": "cool and blue",
    },
}

_STORY_OPENERS = (
    "{names} set out together, and the {place} opens up around them.",
    "It begins with {names}, in a {place} full of possibility.",
    "{names} start the day with no plan at all, which suits them fine.",
)

_STORY_BASE_VERBS = re.compile(
    r"^(?:explore|play|find|chase|sit|run|walk|jump|dance|swim|build|help|meet|"
    r"wave|laugh|sing|climb|hide|search|discover|watch|eat|cook|read|paint|draw|"
    r"fly|ride|visit|travel|look|talk|speak|smile|open|give|take|hold|carry|bring|"
    r"follow|escape|rescue|celebrate|wait|listen|cry|fall|stand|return|wake|sleep|"
    r"start|stop)\b",
    re.IGNORECASE,
)


def _looks_narrative(prompt: str, characters: list[CharacterSpec]) -> bool:
    """True when a prompt describes a story rather than a documentary topic."""
    text = (prompt or "").strip()
    if not text:
        return False
    low = text.lower()
    if any(cue in low for cue in ("once upon", "story about", "adventure", "tale of")):
        return True
    if not _STORY_VERB_RE.search(text):
        return False
    if len(_split_sentences(text)) >= 2:
        return True
    return len([c for c in characters if c.role != "narrator"]) >= 2


def _story_names(characters: list[CharacterSpec]) -> list[str]:
    return [c.name for c in characters if c.role != "narrator"]


def _join_names(names: list[str]) -> str:
    names = [n for n in names if n]
    if not names:
        return "The friends"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _story_env(env_type: str | None) -> dict[str, Any]:
    return _ENV_STORY.get((env_type or "generic").lower()) or _ENV_STORY["garden"]


def _prompt_beats(prompt: str) -> list[str]:
    beats: list[str] = []
    for sentence in _split_sentences(prompt):
        parts = re.split(r"(?:,\s*and then|,\s*then|;|\band then)\s+", sentence,
                         flags=re.IGNORECASE)
        for part in parts:
            part = part.strip(" .,;:")
            if _words(part) >= 2:
                beats.append(part)
    return beats or [prompt.strip()]


def _normalize_beat(beat: str, names: list[str]) -> str:
    text = " ".join(beat.strip().split())
    text = re.sub(r"^(and then|then|and)\s+", "", text, flags=re.IGNORECASE)
    joined = _join_names(names)
    text = re.sub(r"^(they|the friends)\s+", f"{joined} ", text, flags=re.IGNORECASE)
    # A clause split from "..., and then sit ..." starts with a bare verb.
    if _STORY_BASE_VERBS.match(text):
        text = f"{joined} {text[:1].lower()}{text[1:]}"
    text = text[:1].upper() + text[1:]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


_PURPOSE_CLAUSE_RE = re.compile(
    r"\s+to\s+(?:watch|see|look|find|chase|play|sit|run|walk|talk|speak|eat|"
    r"read|build|climb|swim|sing|dance|celebrate|help|meet|visit|travel|return|"
    r"wake|wait|hide|search|discover|carry|hold|give|take)",
    re.IGNORECASE,
)


def _beat_title(text: str, max_words: int = 9) -> str:
    text = text.rstrip(" .!?")
    text = _PURPOSE_CLAUSE_RE.split(text, maxsplit=1)[0]
    text = re.split(r"\s+(?:where|when|while|because|so that|until)\s+", text,
                    maxsplit=1, flags=re.IGNORECASE)[0]
    if ", " in text and len(text.split(", ")[0].split()) >= 3:
        text = text.split(", ")[0]
    words = text.split()[:max_words]
    while words and words[-1].lower() in ("to", "and", "the", "a", "an", "of",
                                          "with", "by", "in", "on", "at", "for"):
        words.pop()
    title = " ".join(words)
    return title[:1].upper() + title[1:]


def _story_dialogue_for(action: str, characters: list[CharacterSpec],
                         cycle: int) -> list[DialogueLine]:
    if not characters:
        return []
    pairs = _STORY_DIALOGUE.get(action, _STORY_DIALOGUE["_default"])
    pair = pairs[cycle % len(pairs)]
    first = characters[cycle % len(characters)]
    others = [c for c in characters if c.id != first.id]
    lines = [DialogueLine(speaker=first.id, text=pair[0], source="generated")]
    if others and pair[1]:
        second = others[cycle % len(others)]
        lines.append(DialogueLine(speaker=second.id, text=pair[1],
                                  source="generated"))
    return lines


def _story_script(
    prompt: str,
    characters: list[CharacterSpec],
    env_type: str | None,
    target_words: int,
    wpm: int,
    language: str,
) -> tuple[Script, dict[str, list[DialogueLine]]]:
    """Compose an on-topic story script offline from the prompt's own beats."""
    story_chars = [c for c in characters if c.role != "narrator"]
    names = _story_names(story_chars)
    joined = _join_names(names)
    env = _story_env(env_type)
    place = env["place"]
    beats = _prompt_beats(prompt)
    dialogue_map: dict[str, list[DialogueLine]] = {}
    sections: list[ScriptSection] = []
    budget = max(60, int(target_words))
    total = 0

    def compose(index: int, beat: str, cycle: int) -> None:
        nonlocal total
        actions = _find_actions(beat)
        action = actions[0] if actions else ""
        props = _find_props(beat)
        prop = props[0].replace("_", " ") if props else "something half-hidden"
        sight = env["sights"][index % len(env["sights"])]
        sound = env["sounds"][index % len(env["sounds"])]
        sight2 = env["sights"][(index + 1) % len(env["sights"])]
        a = names[0] if names else "The first friend"
        b = names[1] if len(names) > 1 else "the others"
        slots = {"a": a, "b": b, "names": joined, "they": "They",
                 "them": "them", "place": place, "prop": prop,
                 "sight": sight2, "sound": sound, "air": env["air"]}
        details = _STORY_DETAIL.get(action, _STORY_DETAIL["_default"])
        body = [_normalize_beat(beat, names)]
        if cycle == 0 and index < len(_STORY_OPENERS):
            body.insert(0, _STORY_OPENERS[index].format(
                names=joined, place=place, sight=sight, sound=sound))
        body.append(details[index % len(details)].format(**slots))
        body.append(_STORY_SENSORY[index % len(_STORY_SENSORY)].format(**slots))
        if index and index % 2 == 0:
            body.append(_STORY_TRANSITIONS[index % len(_STORY_TRANSITIONS)])
        narration = " ".join(body)
        section = ScriptSection(
            id=f"story_{len(sections) + 1:03d}",
            beat=action or "story",
            title=_short_title(_beat_title(_normalize_beat(beat, names))
                               + (f" ({cycle + 1})" if cycle else ""), 58),
            narration=narration,
            word_count=_words(narration),
            visual_hint=f"{env_type or 'story'} scene, {action or 'idle'} action",
        )
        sections.append(section)
        total += section.word_count
        if action or index % 3 == 0:
            dialogue_map[section.id] = _story_dialogue_for(action, story_chars, cycle)

    index = 0
    while total < budget and len(sections) < FILM_MAX_SCENES:
        compose(index, beats[index % len(beats)], index // len(beats))
        index += 1
        if index > FILM_MAX_SCENES * 2:  # pragma: no cover - safety valve
            break

    closing = (
        f"{joined} will remember this day in the {place}. "
        "The light softens, the story slows, and everything comes gently to a close."
    )
    sections.append(ScriptSection(
        id=f"story_{len(sections) + 1:03d}", beat="closing",
        title="Closing Thoughts", narration=closing, word_count=_words(closing),
        visual_hint=f"wide {env_type or 'story'} scene, calm closing shot",
    ))

    actual_words = sum(s.word_count for s in sections)
    dialogue_seconds = sum(_estimate_line_seconds(l.text, wpm)
                           for lines in dialogue_map.values() for l in lines)
    warnings: list[str] = []
    if (language or "en").lower()[:2] not in ("en", ""):
        warnings.append("The offline story writer only composes English narration; "
                        "supply a translated --script for other languages.")
    script = Script(
        topic=extract_topic(prompt),
        kind="story",
        language="en",
        target_minutes=round((actual_words / max(1, wpm) * 60 + dialogue_seconds) / 60, 2),
        target_wpm=wpm,
        target_words=budget,
        actual_words=actual_words,
        estimated_seconds=round(actual_words / max(1, wpm) * 60 + dialogue_seconds, 1),
        sections=sections,
        script_source="prompt_story",
        warnings=warnings,
    )
    return script, dialogue_map


def _apply_prompt_names(plan: PromptPlan, prompt: str) -> list[str]:
    """Give generic "Character N" specs the names the prompt actually uses.

    Only names that appear directly before a story verb ("Lily runs", "Ben
    waves") are accepted, so random capitalised words are not mistaken for
    people.  Existing ids are kept, so dialogue bookkeeping stays intact.
    """
    generic = [c for c in plan.characters
               if c.role != "narrator" and re.fullmatch(r"Character \d+", c.name)]
    if not generic:
        return []
    found: list[str] = []
    for match in _NAME_VERB_RE.finditer(prompt):
        name = match.group(1)
        if name in _NAME_STOPWORDS or name in found:
            continue
        found.append(name)
    if len(found) < 2 or len(found) != len(generic):
        return []
    for spec, name in zip(generic, found):
        spec.name = name
        spec.description = (spec.description or "").strip() or name
    return found


def plan_film(
    prompt: str,
    options: FilmOptions | None = None,
    *,
    available: dict[str, bool] | None = None,
    config: Config | None = None,
    cache_root: Path | None = None,
) -> tuple[FilmPlan, PromptPlan, Script]:
    """Build a structured film plan (scenes, actions, props, camera, lines)."""
    opts = options or FilmOptions()
    cfg = config or Config()
    root = Path(cache_root or cfg.data.get("paths", {}).get("cache_dir", "cache"))
    root.mkdir(parents=True, exist_ok=True)

    minutes, notes = clamp_film_minutes(opts.duration_minutes)
    warnings: list[str] = []
    limitations: list[str] = []
    target_seconds = minutes * 60.0

    caps = dict(available or {})
    plan = understand_prompt(prompt, available=caps, default_duration=target_seconds)
    notes.extend(n for n in plan.parse_notes
                 if "placeholder lines were generated" not in n)
    # The short-conversation warning does not apply: film mode fills the runtime
    # with the generated narration script, not by padding the dialogue.
    warnings.extend(w for w in plan.warnings
                    if "longer than the measured dialogue" not in w)
    plan.requested_method = "animated_film_2d"
    plan.supported_method = "animated_story_2d"
    plan.method_note = ("Film mode renders real rigged 2D character animation with measured "
                        "offline speech. It is not 3D and not neural video.")
    if opts.aspect_ratio:
        plan.aspect_ratio = opts.aspect_ratio
    detected_names = _apply_prompt_names(plan, prompt)
    if detected_names:
        notes.append("Prompt names were matched to the cast: "
                     + ", ".join(detected_names) + ".")
    # Respect the age the prompt implies: children should not stand adult-tall.
    for spec in plan.characters:
        if abs(spec.height_scale - 1.0) < 1e-6:
            spec.height_scale = {"child": 0.8, "teen": 0.9, "elderly": 0.95}.get(
                spec.age_group, 1.0)

    # ---- Script ----------------------------------------------------------
    script: Script
    story_dialogue: dict[str, list[DialogueLine]] = {}
    story_mode = _looks_narrative(prompt, list(plan.characters))
    if story_mode:
        # The story writer supplies on-topic dialogue; the planner's generic
        # placeholder exchanges would otherwise be duplicated into the film.
        plan.dialogue = [d for d in plan.dialogue if d.source != "generated"]
    user_text = opts.script_text
    if user_text is None and opts.script_path:
        user_text = Path(opts.script_path).read_text(encoding="utf-8", errors="replace")
    dialogue_seconds = 0.0
    if plan.dialogue:
        dialogue_seconds = sum(_estimate_line_seconds(l.text, DEFAULT_WPM)
                               for l in plan.dialogue)
        dialogue_seconds = min(dialogue_seconds, target_seconds * FILM_DIALOGUE_AUDIO_SHARE)

    if user_text:
        script = parse_user_script(user_text, language=opts.language,
                                   topic=extract_topic(prompt))
        notes.append("Spoken content comes from the supplied script file.")
        if opts.language != "en" and script.language == "en":
            warnings.append("The template engine only writes English narration; a supplied "
                            "script is spoken as written.")
    else:
        wpm, calib_note = _calibrate_wpm(opts.language, cfg, root)
        if calib_note:
            notes.append(calib_note)
        narration_minutes = max(0.5, (target_seconds - dialogue_seconds) / 60.0)
        if story_mode:
            # A story prompt gets a story, not documentary filler about one noun.
            # Leave room for line/scene pause overhead and story dialogue so
            # the measured export lands near the requested runtime.
            budget = max(60, int(compute_word_budget(narration_minutes, wpm) * 0.86) - 70)
            script, story_dialogue = _story_script(
                prompt, list(plan.characters), plan.environment_type,
                budget, wpm, opts.language)
            notes.append("Narration was written offline from the prompt as a story "
                         f"({script.actual_words} words at {wpm} wpm); it paraphrases the "
                         "prompt's own beats and does not invent facts.")
        else:
            engine = TemplateScriptEngine(wpm=wpm)
            script = engine.generate(prompt, target_minutes=narration_minutes,
                                     language=opts.language)
            notes.append(f"Generated a {script.actual_words}-word narration script "
                         f"(~{script.estimated_seconds / 60:.1f} min at {wpm} wpm).")
    warnings.extend(script.warnings)
    actual_words = script.actual_words
    if actual_words < 40 and not user_text:
        warnings.append("The generated script is very short for the requested duration.")
    wpm = max(80, min(230, int(script.target_wpm or DEFAULT_WPM)))

    # ---- Characters ------------------------------------------------------
    characters = list(plan.characters)
    needs_narrator = any(_words(s.narration) > 0 for s in script.sections)
    story_chars = list(characters)
    if needs_narrator:
        characters.append(_narrator_spec(plan))
    if not characters:
        characters = [_narrator_spec(plan)]
        story_chars = []
        warnings.append("No characters were detected or created; the film is a narrated "
                        "visual with camera work and no character animation.")
    on_screen = [c.id for c in story_chars] or [NARRATOR_ID]

    # ---- Scene packing ---------------------------------------------------
    scene_seconds = max(6.0, float(opts.target_scene_seconds or FILM_TARGET_SCENE_SECONDS))
    units: list[dict[str, Any]] = []
    for section in script.sections:
        chunk: list[str] = []
        chunk_seconds = 0.0
        for sentence in _split_sentences(section.narration):
            chunk.append(sentence)
            chunk_seconds += _estimate_line_seconds(sentence, wpm)
            if chunk_seconds >= scene_seconds:
                units.append(_unit(section, chunk, wpm))
                chunk, chunk_seconds = [], 0.0
        if chunk:
            units.append(_unit(section, chunk, wpm))
    if not units:
        units = [{"section": None, "title": plan.title, "purpose": "opening",
                  "sentences": [plan.setting or plan.title], "seconds": scene_seconds}]
    units = units[:FILM_MAX_SCENES]

    # Distribute explicit prompt dialogue across the film in conversational
    # chunks so interactions recur instead of being dumped in one scene.
    dialogue_chunks: dict[int, list[DialogueLine]] = {}
    if plan.dialogue:
        n_scenes = max(1, len(units))
        groups = max(1, min(n_scenes, math.ceil(len(plan.dialogue) / 4)))
        per_group = math.ceil(len(plan.dialogue) / groups)
        for g in range(groups):
            chunk = plan.dialogue[g * per_group:(g + 1) * per_group]
            if not chunk:
                continue
            scene_index = min(n_scenes - 1, int(g * n_scenes / groups))
            dialogue_chunks.setdefault(scene_index, []).extend(chunk)

    scenes: list[FilmScene] = []
    total_planned = 0.0
    used_story_dialogue: set[str] = set()
    for i, unit in enumerate(units):
        lines: list[DialogueLine] = list(dialogue_chunks.get(i, []))
        for sentence in unit["sentences"]:
            lines.append(DialogueLine(speaker=NARRATOR_ID, text=sentence,
                                      source="generated" if not user_text else "script"))
        section = unit.get("section")
        if section is not None and section.id not in used_story_dialogue:
            used_story_dialogue.add(section.id)
            lines.extend(story_dialogue.get(section.id, []))
        scene_text = " ".join([unit["title"], unit.get("purpose", "")] +
                              [ln.text for ln in lines])
        env, _setting, elements, _t = _detect_environment(scene_text)
        # A story lives in one place: do not bounce between environments
        # because an action word ("waves") looks like a setting cue.
        if story_mode:
            environment = plan.environment_type or env
        else:
            environment = env if env != "generic" else plan.environment_type
        actions, props = direct_scene(i, scene_text, list(on_screen),
                                      on_screen=list(on_screen))
        duration = sum(_estimate_line_seconds(ln.text, wpm) for ln in lines) + 0.4
        scenes.append(FilmScene(
            index=i,
            scene_id=f"sc_{i + 1:03d}",
            title=_short_title(unit["title"]),
            purpose=unit.get("purpose", ""),
            environment=environment,
            time_of_day=plan.time_of_day,
            duration_planned=round(duration, 2),
            characters=list(on_screen),
            lines=lines,
            actions=actions,
            props=props,
            continuity=["characters keep the same appearance and clothing",
                        "the spoken track is continuous across scene cuts"],
        ))
        total_planned += duration

    film = FilmPlan(
        prompt=prompt,
        title=opts.title or plan.title,
        project_type=plan.project_type,
        environment_type=plan.environment_type,
        aspect_ratio=plan.aspect_ratio,
        language=opts.language,
        quality=opts.quality,
        target_minutes=minutes,
        target_seconds=target_seconds,
        wpm=wpm,
        estimated_seconds=round(total_planned, 1),
        characters=characters,
        scenes=scenes,
        script=script.to_dict(),
        notes=notes,
        warnings=warnings,
        limitations=limitations,
    )
    if total_planned < target_seconds * 0.75:
        film.warnings.append(
            f"The planned narration is about {total_planned / 60:.1f} min, shorter than the "
            f"{minutes:g}-minute target. The render will not pad speech with repetition; the "
            "final video follows the real measured audio.")
    if total_planned > target_seconds * 1.25:
        film.warnings.append(
            f"The planned narration is about {total_planned / 60:.1f} min, longer than the "
            f"{minutes:g}-minute target; the measured export will follow the actual speech.")
    film.limitations.append(
        "Narration is synthesized by installed offline TTS voices; character acting is "
        "procedural 2D rig animation, not motion-captured or neural performance.")
    return film, plan, script


def _short_title(text: str, limit: int = 62) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:-") + "..."


def _unit(section: ScriptSection | None, sentences: list[str], wpm: int) -> dict[str, Any]:
    seconds = sum(_estimate_line_seconds(s, wpm) for s in sentences)
    return {
        "section": section,
        "title": (section.title if section else "Opening"),
        "purpose": (section.beat if section else "opening"),
        "sentences": sentences,
        "seconds": seconds,
    }


def _narrator_spec(plan: PromptPlan) -> CharacterSpec:
    return CharacterSpec(
        id=NARRATOR_ID, name=NARRATOR_NAME, role="narrator",
        species="human", description="Off-screen narrator voice.",
        appearance="neutral presenter", clothing="smart casual",
        color_palette=["#e8d5c0", "#2f2f3a", "#4a6fa5", "#f2e8cf"],
        personality="warm, clear narrator", age_group="adult", build="average",
        hair_style="short", accessories=[], height_scale=1.0, voice="auto",
    )


# ---------------------------------------------------------------------------
# Film studio
# ---------------------------------------------------------------------------
class FilmStudio(Studio):
    """Long-video production with scene-level resumability."""

    # -- paths ---------------------------------------------------------
    @staticmethod
    def _film_dir(project: Project) -> Path:
        d = project.path("film")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _state_path(project: Project) -> Path:
        return FilmStudio._film_dir(project) / "state.json"

    @staticmethod
    def _plan_path(project: Project) -> Path:
        return project.path("script") / "film_plan.json"

    # -- planning ------------------------------------------------------
    def build_plan(self, prompt: str, options: FilmOptions | None = None) -> FilmPlan:
        opts = options or FilmOptions()
        hw = detect_hardware(self.projects_root)
        caps = {k: (c.status == "available") for k, c in hw.capabilities.items()}
        film, _plan, _script = plan_film(prompt, opts, available=caps,
                                         config=self.config, cache_root=self.cache_root)
        return film

    def create_project(self, prompt: str, plan: FilmPlan,
                       options: FilmOptions | None = None) -> Project:
        opts = options or FilmOptions()
        project = Project.create(
            self.projects_root, name=opts.title or plan.title, prompt=prompt,
            languages={"project": opts.language},
            settings={"film": opts.to_dict(), "film_version": 1},
        )
        project.manifest["project_type"] = plan.project_type
        project.manifest["requested_method"] = "animated_film_2d"
        project.manifest["method_note"] = ("Rigged 2D character animation with measured "
                                           "offline speech; not 3D and not neural video.")
        project.manifest["film"] = {
            "target_minutes": plan.target_minutes,
            "target_seconds": plan.target_seconds,
            "scene_count": plan.scene_count,
            "quality": plan.quality,
        }
        project.save()
        project.write_json("script", "script.json", plan.script)
        project.write_json("script", "film_plan.json", plan.to_dict())
        project.mark_stage(STATUS_SCRIPT_READY)
        return project

    def load_options(self, project: Project) -> FilmOptions:
        data = dict((project.manifest.get("settings") or {}).get("film") or {})
        known = FilmOptions.__dataclass_fields__
        return FilmOptions(**{k: v for k, v in data.items() if k in known})

    def load_plan(self, project: Project) -> FilmPlan:
        data = project.read_json("script", "film_plan.json")
        if not data:
            raise RuntimeError("project has no film plan; run 'freeanimate film plan' first")
        return FilmPlan.from_dict(data)

    # -- conversation persistence --------------------------------------
    def _load_conversation(self, project: Project) -> ConversationResult | None:
        data = project.read_json("narration", "conversation.json")
        voice = project.path("narration", "conversation_voice.wav")
        if not data or not voice.exists():
            return None
        lines: list[RenderedLine] = []
        for d in data.get("lines", []):
            kw = {k: v for k, v in d.items() if k in RenderedLine.__dataclass_fields__}
            kw["wav"] = Path(kw["wav"]) if kw.get("wav") else None
            lines.append(RenderedLine(**kw))
        convo = ConversationResult(ok=True, lines=lines,
                                   total_seconds=float(data.get("total_seconds") or 0.0),
                                   notes=list(data.get("notes") or []))
        convo.voice_track = voice
        convo.spans = [DialogueSpan(character_id=l.speaker_id, text=l.text,
                                    start=l.start, end=l.end, emotion=l.emotion,
                                    action=l.action, visemes=l_visemes(convo, l))
                       for l in lines]
        # Rebuild the subtitle track the resumed run needs; it is not stored in
        # conversation.json because it derives directly from the measured lines.
        convo.subtitles = _film_subtitle_track(
            lines, str(data.get("language") or "en"),
            int(self.config.data["subtitles"].get("max_line_chars", 46)),
            int(self.config.data["subtitles"].get("max_lines", 2)))
        return convo

    # -- measured scene windows ---------------------------------------
    @staticmethod
    def _scene_windows(scenes: list[FilmScene],
                       convo: ConversationResult) -> list[tuple[float, float]]:
        per_scene: list[list[RenderedLine]] = [[] for _ in scenes]
        pointer = 0
        for si, scene in enumerate(scenes):
            for line in scene.lines:
                j = pointer
                while j < len(convo.lines) and convo.lines[j].text != line.text:
                    j += 1
                if j < len(convo.lines):
                    per_scene[si].append(convo.lines[j])
                    pointer = j + 1
        first = [g[0].start if g else None for g in per_scene]
        last = [g[-1].end if g else None for g in per_scene]
        cuts = [0.0]
        for i in range(1, len(scenes)):
            prev_end, cur_start = last[i - 1], first[i]
            if prev_end is not None and cur_start is not None:
                cut = (prev_end + cur_start) / 2.0
            elif prev_end is not None:
                cut = prev_end + 0.25
            elif cur_start is not None:
                cut = max(0.0, cur_start - 0.25)
            else:
                cut = cuts[-1] + 0.6
            cuts.append(max(cuts[-1] + 0.25, cut))
        end = max(cuts[-1] + 0.25, convo.total_seconds) if len(cuts) > 1 else convo.total_seconds
        cuts[-1] = max(0.0, min(cuts[-1], end))
        windows = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        windows.append((cuts[-1], end))
        return windows

    # -- rendering -----------------------------------------------------
    def run(self, prompt: str | None = None, options: FilmOptions | None = None, *,
            project_ref: str | None = None, resume: bool = True,
            progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> FilmResult:
        """Plan (when needed) and render a full film, resuming completed scenes."""
        report = progress or (lambda *_: None)

        def step(stage: str, fraction: float, message: str | None = None) -> None:
            if cancel and cancel():
                raise RuntimeError("cancelled")
            report(stage, max(0.0, min(1.0, fraction)), message)

        notes: list[str] = []
        warnings: list[str] = []
        limitations: list[str] = []
        try:
            step("hardware", 0.01, "Detecting hardware and capabilities")
            hw = detect_hardware(self.projects_root)

            if project_ref:
                project = resolve_project(self.projects_root, project_ref)
                plan = self.load_plan(project)
                opts = self.load_options(project)
                if prompt is None:
                    prompt = project.manifest.get("prompt") or plan.prompt
                step("plan", 0.03, f"Loaded film plan: {plan.scene_count} scenes")
            else:
                if not prompt:
                    raise ValueError("a prompt is required to create a film")
                opts = options or FilmOptions()
                step("plan", 0.03, "Writing the film plan (scene, action and camera design)")
                plan = self.build_plan(prompt, opts)
                if not plan.scenes:
                    raise RuntimeError("the film plan has no scenes")
                project = self.create_project(prompt, plan, opts)

            notes.extend(plan.notes)
            warnings.extend(plan.warnings)
            limitations.extend(plan.limitations)
            resolution, crf, preset = opts.render_settings()
            width, height = _resolution_dims(resolution, plan.aspect_ratio)
            fps = int(opts.fps)

            fingerprint = self._fingerprint(project, plan, opts)
            state = self._load_state(project)
            if not resume or state.get("fingerprint") != fingerprint:
                state = {"fingerprint": fingerprint, "scenes": {}}
            if state.get("stage") == "complete" and resume:
                out = project.path("renders", "final.mp4")
                if out.exists():
                    notes.append("The film is already complete; nothing needed re-rendering.")
                    return FilmResult(True, project=project, output=out,
                                      thumbnail=project.path("thumbnails", "thumbnail.jpg"),
                                      plan=plan, notes=notes, warnings=warnings,
                                      limitations=limitations,
                                      duration=float(plan.estimated_seconds),
                                      scene_count=plan.scene_count)

            # ---- Characters ------------------------------------------
            step("characters", 0.06, "Creating persistent character designs")
            lib_root = (Path(opts.characters_library) if opts.characters_library
                        else project.path("assets", "characters"))
            library = CharacterLibrary(lib_root)
            characters = ensure_project_characters(library, plan.characters,
                                                   project.manifest["slug"])
            if not characters:
                raise RuntimeError("no character could be created for this film")
            char_by_id = {c.id: c for c in characters}
            on_screen_chars = [c for c in characters if c.id != NARRATOR_ID] or characters

            # ---- Voices ----------------------------------------------
            convo = self._load_conversation(project) if resume else None
            if convo is not None:
                notes.append("Reusing the measured voice track from the previous run.")
            else:
                step("voices", 0.10, "Synthesizing all dialogue with local voices")
                flat_lines = [ln for scene in plan.scenes for ln in scene.lines]
                convo = render_conversation(
                    flat_lines, characters, project.stage_dir("narration"),
                    language=opts.language,
                    config={"wpm": int(plan.wpm), "tts_engine": opts.tts_engine,
                            "tts_voice": opts.tts_voice},
                    max_chars=int(self.config.data["subtitles"].get("max_line_chars", 46)),
                    max_lines=int(self.config.data["subtitles"].get("max_lines", 2)),
                )
                notes.extend(convo.notes)
                if not convo.ok or convo.voice_track is None:
                    raise RuntimeError(f"dialogue synthesis failed: {convo.error}")
                convo.subtitles = _film_subtitle_track(
                    convo.lines, opts.language,
                    int(self.config.data["subtitles"].get("max_line_chars", 46)),
                    int(self.config.data["subtitles"].get("max_lines", 2)))
                project.write_json("narration", "conversation.json", {
                    "lines": [l.to_dict() for l in convo.lines],
                    "assignments": [a.to_dict() for a in convo.assignments],
                    "total_seconds": convo.total_seconds,
                    "notes": convo.notes,
                })
                project.mark_stage(STATUS_NARRATED)
            for a in convo.assignments:
                if not a.distinct:
                    limitations.append(
                        f"Only one distinct installed voice was available for {a.character_name}; "
                        "rate variation was used instead of a different voice.")
            if any(a.engine == "silence" for a in convo.assignments):
                limitations.append("No local TTS voice was installed; the film has silence with "
                                   "accurately timed subtitles.")

            # ---- Subtitles -------------------------------------------
            step("subtitles", 0.16, "Writing subtitles from measured audio")
            if convo.subtitles is None:
                raise RuntimeError("conversation produced no subtitle track")
            sub_files = write_subtitles(convo.subtitles, project.stage_dir("subtitles"),
                                        stem="subtitles")
            srt_path = sub_files.get("srt")

            # ---- Measured scene windows ------------------------------
            windows = self._scene_windows(plan.scenes, convo)
            for scene, (start, end) in zip(plan.scenes, windows):
                scene.start = round(start, 3)
                scene.end = round(end, 3)
                scene.duration_actual = round(max(0.4, end - start), 3)
            total_duration = convo.total_seconds
            project.write_json("script", "film_plan.json", plan.to_dict())

            # ---- Environments ----------------------------------------
            step("backgrounds", 0.20, "Rendering scene environments")
            backgrounds: list[Path] = []
            for scene in plan.scenes:
                backgrounds.append(self._scene_background(scene, plan, project, opts,
                                                           width, height))
                scene.background = str(backgrounds[-1])

            # ---- Scenes ----------------------------------------------
            placements = self._placements(on_screen_chars)
            segments: list[Path] = []
            rendered_frames = 0
            render_started = time.time()
            n_scenes = len(plan.scenes)
            for i, scene in enumerate(plan.scenes):
                seg = project.path("renders", f"scene_{i + 1:02d}.mp4")
                segments.append(seg)
                cached = state.get("scenes", {}).get(str(i + 1))
                if (resume and cached and cached.get("status") == "done"
                        and seg.exists() and seg.stat().st_size > 1024):
                    try:
                        actual = probe(seg).duration or 0.0
                    except FFmpegError:
                        actual = 0.0
                    if abs(actual - scene.duration_actual) <= 0.8:
                        scene.status = "done"
                        scene.segment = str(seg)
                        scene.frames = int(cached.get("frames") or 0)
                        rendered_frames += scene.frames
                        step("scenes", 0.22 + 0.62 * (i + 1) / n_scenes,
                             f"Scene {i + 1}/{n_scenes}: reused existing segment")
                        continue
                scene.status = "rendering"
                anim = self._build_scene_animation(scene, plan, opts, convo,
                                                   placements, backgrounds[i],
                                                   on_screen_chars, width, height)
                tmp = project.path("renders", f".scene_{i + 1:02d}.tmp.mp4")

                def _prog(done: int, total: int, _i: int = i) -> None:
                    frac = 0.22 + 0.62 * ((_i + done / max(1, total)) / max(1, n_scenes))
                    step("scenes", frac,
                         f"Scene {_i + 1}/{n_scenes}: frame {done}/{total}")

                result = render_scene_animation(
                    anim, tmp, crf=crf, preset=preset, progress=_prog,
                    cancel=(lambda: bool(cancel and cancel())))
                if not result.ok:
                    scene.status = "failed"
                    scene.error = result.error
                    state.setdefault("scenes", {})[str(i + 1)] = {
                        "status": "failed", "error": result.error}
                    self._save_state(project, state)
                    raise RuntimeError(f"scene {i + 1}/{n_scenes} failed: {result.error}")
                tmp.replace(seg)
                scene.status = "done"
                scene.segment = str(seg)
                scene.frames = int(result.frames or 0)
                rendered_frames += scene.frames
                state.setdefault("scenes", {})[str(i + 1)] = {
                    "status": "done", "frames": scene.frames,
                    "seconds": scene.duration_actual, "segment": str(seg)}
                self._save_state(project, state)

            # ---- Assemble --------------------------------------------
            step("assemble", 0.86, "Assembling scenes")
            if len(segments) == 1:
                base = segments[0]
            else:
                base = project.path("renders", "animation.mp4")
                concat_segments(segments, base,
                                list_file=project.path("renders", "segments.txt"))

            step("audio", 0.89, "Mixing the audio track")
            if opts.music == "procedural":
                notes.append("Background bed is procedural ambience, not AI-composed music.")
            mix = mix_final_audio(
                convo.voice_track, project.path("audio", "mix.wav"),
                total_seconds=total_duration, music=opts.music,
                music_volume=opts.music_volume, music_path=self._music_path(opts.music),
                sample_rate=int(self.config.data["render"].get("sample_rate", DEFAULT_SAMPLE_RATE)),
            )
            if not mix.ok or mix.final_mix_path is None:
                raise RuntimeError(f"audio mix failed: {mix.error}")
            notes.extend(mix.notes or [])

            step("export", 0.92, "Encoding the final film")
            out_path = project.path("renders", "final.mp4")
            burn = bool(opts.burn_subtitles and srt_path is not None)
            if burn and not has_filter("subtitles"):
                burn = False
                warnings.append("This FFmpeg build lacks libass; subtitles are provided as "
                                "SRT/VTT but were not burned into the picture.")
            self._finalize(base, mix.final_mix_path, srt_path if burn else None,
                           out_path, crf=crf, preset=preset)
            duration = mix.total_seconds or wav_duration_seconds(mix.final_mix_path)

            step("thumbnail", 0.95, "Creating thumbnail")
            thumb = project.path("thumbnails", "thumbnail.jpg")
            try:
                self._thumbnail(out_path, thumb)
            except FFmpegError:
                thumb = None  # type: ignore[assignment]
                warnings.append("Thumbnail extraction failed; the film is still complete.")

            # ---- Manifest + sidecars for thumbnails -----------------
            self._write_scene_manifest(project, plan)
            project.write_json("script", "script.json", plan.script)
            project.manifest["scenes"] = {"count": len(segments), "method": ANIM_2D_METHOD}
            project.manifest["narration"] = {"lines": len(convo.lines),
                                             "total_seconds": convo.total_seconds}
            project.manifest["render"] = {"width": width, "height": height, "fps": fps,
                                          "segments": len(segments)}
            project.manifest["export"] = {
                "path": str(out_path), "resolution": resolution, "fps": fps,
                "duration_seconds": duration,
            }
            project.mark_stage(STATUS_RENDERED)

            # ---- Validate + QA --------------------------------------
            step("validate", 0.97, "Validating the export and running QA checks")
            methods = {
                "animation": ANIM_2D_METHOD,
                "render_style": "2d",
                "rig": "skeletal_2d",
                "backgrounds": "procedural_environment",
                "lipsync": "viseme_timeline_2d",
                "tts": sorted({l.engine for l in convo.lines}),
                "voices": len({l.voice_id for l in convo.lines if l.voice_id}),
                "music": mix.music_used,
                "subtitles": "libass_burn_in" if burn else "sidecar_files",
            }
            validation = validate_output(
                out_path, expected_width=width, expected_height=height,
                expected_fps=fps, requested_seconds=duration,
                require_audio=True, subtitle_files=list(sub_files.values()),
                expected_scenes=len(segments), rendered_segments=len(segments),
                methods_used=methods, limitations=limitations,
            )
            report_json, report_txt = write_report(validation, project.stage_dir("reports"))

            qa = None
            try:
                from .qa import run_qa, write_qa_report
                qa = run_qa(project, plan, out_path, convo=convo,
                            validation=validation, languages=None)
                write_qa_report(qa, project.stage_dir("reports"))
            except Exception as exc:  # noqa: BLE001 — QA must not destroy a good render
                warnings.append(f"Automated QA could not complete: {type(exc).__name__}: {exc}")

            # ---- Thumbnails -----------------------------------------
            thumbnail_options: list[dict[str, Any]] = []
            try:
                from .thumbnails import generate_thumbnails
                thumb_report = generate_thumbnails(project)
                thumbnail_options = list(thumb_report.get("options") or [])
                notes.append("Generated thumbnail options from real scene images.")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Thumbnail options were not generated: {exc}")

            # ---- Honest report --------------------------------------
            elapsed = time.time() - render_started
            report_data = {
                "generated_at": time.time(),
                "mode": "film",
                "requested_minutes": plan.target_minutes,
                "target_seconds": plan.target_seconds,
                "planned_seconds": plan.estimated_seconds,
                "actual_duration_seconds": duration,
                "scene_count": len(segments),
                "resolution": resolution,
                "fps": fps,
                "quality_preset": opts.quality,
                "wpm": plan.wpm,
                "render_elapsed_seconds": round(elapsed, 1),
                "frames_rendered": rendered_frames,
                "output": str(out_path),
                "validation_ok": validation.ok,
                "qa_ok": getattr(qa, "ok", None),
                "thumbnail_options": thumbnail_options,
                "notes": notes,
                "warnings": warnings,
                "limitations": limitations,
                "hardware": hw.to_dict(),
                "methods_used": methods,
                "model_downloads": "none (nothing was downloaded automatically)",
            }
            fr_path = project.path("reports", "film_report.json")
            fr_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2,
                                          default=str), encoding="utf-8")
            state["stage"] = "complete"
            self._save_state(project, state)
            project.mark_stage(STATUS_EXPORTED)

            step("complete", 1.0, "Film ready")
            return FilmResult(
                ok=bool(validation.ok), project=project, output=out_path, thumbnail=thumb,
                plan=plan, conversation=convo, validation=validation, qa=qa,
                report_json=report_json, report_txt=report_txt,
                notes=notes, warnings=warnings, limitations=limitations,
                duration=duration, scene_count=len(segments),
                error=None if validation.ok else "output validation reported problems",
            )
        except Exception as exc:  # noqa: BLE001 — surface the real problem
            return FilmResult(False, error=f"{type(exc).__name__}: {exc}",
                              notes=notes, warnings=warnings, limitations=limitations)

    # -- preview -------------------------------------------------------
    def preview(self, project_ref: str, options: FilmOptions | None = None, *,
                progress: ProgressFn | None = None,
                cancel: CancelFn | None = None) -> FilmResult:
        """Render one representative scene at low-res preview settings."""
        report = progress or (lambda *_: None)
        notes: list[str] = []
        warnings: list[str] = []
        try:
            project = resolve_project(self.projects_root, project_ref)
            plan = self.load_plan(project)
            opts = self.load_options(project)
            if options is not None:
                for name in FilmOptions.__dataclass_fields__:
                    value = getattr(options, name)
                    if value != getattr(FilmOptions(), name):
                        setattr(opts, name, value)
            # Preview always uses PC-friendly settings, whatever the final choice.
            width, height = _resolution_dims("480p", plan.aspect_ratio)
            fps = int(opts.fps)
            preview_seconds = max(3.0, float(opts.preview_seconds
                                             or FILM_PREVIEW_DEFAULT_SECONDS))

            report("prepare", 0.05, "Loading the measured voice track")
            convo = self._load_conversation(project)
            if convo is None:
                report("voices", 0.10, "Synthesizing the voice track")
                lib_root = (Path(opts.characters_library) if opts.characters_library
                            else project.path("assets", "characters"))
                library = CharacterLibrary(lib_root)
                characters = ensure_project_characters(library, plan.characters,
                                                       project.manifest["slug"])
                flat_lines = [ln for scene in plan.scenes for ln in scene.lines]
                convo = render_conversation(
                    flat_lines, characters, project.stage_dir("narration"),
                    language=opts.language,
                    config={"wpm": int(plan.wpm), "tts_engine": opts.tts_engine,
                            "tts_voice": opts.tts_voice},
                )
                if not convo.ok or convo.voice_track is None:
                    raise RuntimeError(f"dialogue synthesis failed: {convo.error}")
                project.write_json("narration", "conversation.json", {
                    "lines": [l.to_dict() for l in convo.lines],
                    "assignments": [a.to_dict() for a in convo.assignments],
                    "total_seconds": convo.total_seconds,
                    "notes": convo.notes,
                })

            windows = self._scene_windows(plan.scenes, convo)
            scene = plan.scenes[0]
            scene.start, scene.end = windows[0]
            scene.duration_actual = round(max(0.4, scene.end - scene.start), 3)
            preview_duration = min(preview_seconds, scene.duration_actual)

            lib_root = (Path(opts.characters_library) if opts.characters_library
                        else project.path("assets", "characters"))
            library = CharacterLibrary(lib_root)
            characters = ensure_project_characters(library, plan.characters,
                                                   project.manifest["slug"])
            on_screen_chars = [c for c in characters if c.id != NARRATOR_ID] or characters
            placements = self._placements(on_screen_chars)
            bg = self._scene_background(scene, plan, project, opts, width, height)
            scene.background = str(bg)
            anim = self._build_scene_animation(scene, plan, opts, convo, placements, bg,
                                               on_screen_chars, width, height,
                                               duration_override=preview_duration)
            seg = project.path("renders", "preview_scene.mp4")
            started = time.time()

            def _prog(done: int, total: int) -> None:
                report("preview", 0.15 + 0.7 * (done / max(1, total)),
                       f"Preview frame {done}/{total}")

            result = render_scene_animation(
                anim, seg, crf=32, preset="ultrafast", progress=_prog,
                cancel=(lambda: bool(cancel and cancel())))
            if not result.ok:
                raise RuntimeError(f"preview render failed: {result.error}")
            elapsed = max(0.001, time.time() - started)

            report("export", 0.92, "Muxing the preview with audio")
            audio = project.path("audio", "preview_audio.wav")
            run_ffmpeg(["-y", "-i", str(convo.voice_track), "-t", f"{preview_duration:.3f}",
                        "-c:a", "pcm_s16le", str(audio)], timeout=300)
            out = project.path("renders", "preview.mp4")
            self._finalize(seg, audio, None, out, crf=32, preset="ultrafast")

            actual_frames = int(result.frames or max(1, round(preview_duration * fps)))
            frames_per_second = actual_frames / elapsed
            planned_seconds = (float(plan.estimated_seconds)
                               or sum(s.duration_planned for s in plan.scenes)
                               or float(plan.target_seconds))
            total_frames = planned_seconds * fps
            estimate_seconds = total_frames / max(0.01, frames_per_second)
            estimate = {
                "preview_seconds": round(preview_duration, 2),
                "preview_render_seconds": round(elapsed, 2),
                "measured_fps": round(frames_per_second, 2),
                "estimated_full_render_seconds": round(estimate_seconds, 1),
                "estimated_full_render_minutes": round(estimate_seconds / 60.0, 1),
                "basis": ("Extrapolated from this preview scene on this machine. It is an "
                          "estimate, not a guarantee; long renders can be slower."),
            }
            notes.append(
                f"Preview render measured {frames_per_second:.2f} fps at {width}x{height}; "
                f"the full {plan.target_minutes:g}-minute film is estimated at about "
                f"{estimate_seconds / 60.0:.0f} minutes on this hardware (estimate only).")
            project.write_json("reports", "preview_estimate.json", estimate)
            report("complete", 1.0, "Preview ready")
            return FilmResult(True, project=project, output=out, plan=plan,
                              conversation=convo, notes=notes, warnings=warnings,
                              duration=preview_duration, scene_count=1,
                              estimate_seconds=estimate_seconds, perception=estimate)
        except Exception as exc:  # noqa: BLE001
            return FilmResult(False, error=f"{type(exc).__name__}: {exc}",
                              notes=notes, warnings=warnings)

    # -- status --------------------------------------------------------
    def status(self, project_ref: str) -> dict[str, Any]:
        project = resolve_project(self.projects_root, project_ref)
        plan = self.load_plan(project)
        state = self._load_state(project)
        done = 0
        for i, scene in enumerate(plan.scenes):
            seg = project.path("renders", f"scene_{i + 1:02d}.mp4")
            entry = state.get("scenes", {}).get(str(i + 1)) or {}
            if seg.exists() and seg.stat().st_size > 1024 and entry.get("status") == "done":
                scene.status = "done"
                done += 1
            scene.segment = str(seg) if seg.exists() else ""
        return {
            "project": project.manifest.get("slug"),
            "title": plan.title,
            "target_minutes": plan.target_minutes,
            "planned_seconds": plan.estimated_seconds,
            "scene_count": plan.scene_count,
            "scenes_done": done,
            "stage": state.get("stage", "not_started"),
            "final_exists": project.path("renders", "final.mp4").exists(),
            "output": str(project.path("renders", "final.mp4")),
            "scenes": [
                {"index": s.index + 1, "title": s.title, "environment": s.environment,
                 "planned_seconds": s.duration_planned, "actual_seconds": s.duration_actual,
                 "status": s.status,
                 "lines": len(s.lines), "actions": len(s.actions), "props": len(s.props)}
                for s in plan.scenes
            ],
        }

    # -- validate ------------------------------------------------------
    def validate(self, project_ref: str) -> dict[str, Any]:
        project = resolve_project(self.projects_root, project_ref)
        plan = self.load_plan(project)
        out = project.path("renders", "final.mp4")
        if not out.exists():
            raise RuntimeError(f"no final film found at {out}; render it first")
        opts = self.load_options(project)
        resolution, _crf, _preset = opts.render_settings()
        width, height = _resolution_dims(resolution, plan.aspect_ratio)
        validation = validate_output(out, expected_width=width, expected_height=height,
                                     expected_fps=opts.fps, requested_seconds=None,
                                     require_audio=True,
                                     expected_scenes=plan.scene_count,
                                     rendered_segments=plan.scene_count)
        write_report(validation, project.stage_dir("reports"))
        qa = None
        try:
            from .qa import run_qa, write_qa_report
            convo = self._load_conversation(project)
            qa = run_qa(project, plan, out, convo=convo, validation=validation)
            write_qa_report(qa, project.stage_dir("reports"))
        except Exception as exc:  # noqa: BLE001
            qa = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        qa_data = qa.to_dict() if hasattr(qa, "to_dict") else qa
        return {"validation": validation.to_dict(), "qa": qa_data}

    # -- internals -----------------------------------------------------
    def _scene_background(self, scene: FilmScene, plan: FilmPlan, project: Project,
                          opts: FilmOptions, width: int, height: int) -> Path:
        saved = project.path("assets", f"background_scene_{scene.index + 1:02d}.png")
        if saved.exists() and saved.stat().st_size > 1024:
            return saved
        env = scene.environment or plan.environment_type
        spec = EnvironmentSpec(env_type=env, time_of_day=scene.time_of_day or "day",
                               weather="", lighting="warm", elements=[],
                               seed=f"{opts.seed or plan.prompt[:60]}|{scene.scene_id}")
        bg = render_cached(spec, self.cache_root / "backgrounds", width=width, height=height)
        try:
            shutil.copyfile(bg, saved)
            return saved
        except OSError:
            return bg

    def _build_scene_animation(self, scene: FilmScene, plan: FilmPlan, opts: FilmOptions,
                               convo: ConversationResult,
                               placements: list[CharacterPlacement], bg: Path,
                               on_screen_chars: list[Character],
                               width: int, height: int,
                               duration_override: float | None = None) -> SceneAnimation:
        start = scene.start if scene.start >= 0 else 0.0
        duration = float(duration_override if duration_override is not None
                         else scene.duration_actual or scene.duration_planned)
        duration = max(0.6, duration)
        scene_ids = {c.id for c in on_screen_chars}
        spans = [
            DialogueSpan(character_id=line.speaker_id, text=line.text,
                         start=max(0.0, line.start - start),
                         end=max(0.1, min(line.end, start + duration) - start),
                         emotion=line.emotion, action=line.action,
                         visemes=l_visemes(convo, line))
            for line in convo.lines
            if line.start < start + duration and line.end > start
        ]
        spans = [s for s in spans if s.end > s.start]
        scoped = [p for p in placements if p.rig.id in scene_ids] or placements
        local_lines = [(max(0.0, sp.start), min(duration, sp.end), sp.character_id)
                       for sp in spans]
        movement = []
        for cue in scene.actions:
            if cue.get("action") in ("walk", "run", "leave", "approach", "follow"):
                movement.append((float(cue.get("t0", 0.0)) * duration,
                                 float(cue.get("t1", 0.0)) * duration,
                                 str(cue.get("character_id"))))
        camera = build_camera_cues(local_lines, duration, n_chars=len(scoped),
                                   movement=movement)
        actions = [
            ActorCue(
                character_id=str(cue.get("character_id")),
                action=str(cue.get("action", "idle")),
                start=float(cue.get("t0", 0.0)) * duration,
                end=float(cue.get("t1", 0.0)) * duration,
                x_from=_as_float(cue.get("x_from")),
                x_to=_as_float(cue.get("x_to")),
                target_character=cue.get("target_character"),
                prop=cue.get("prop"),
                emotion=str(cue.get("emotion", "neutral")),
                note=str(cue.get("note", "")),
            )
            for cue in scene.actions
            if cue.get("character_id") in scene_ids and cue.get("action") in _ALLOWED_ACTIONS
        ]
        props = [
            SceneProp(
                id=str(p.get("id")), kind=str(p.get("kind", "box")),
                x=float(p.get("x", 0.5)), y=float(p.get("y", 0.93)),
                x_to=_as_float(p.get("x_to")), y_to=_as_float(p.get("y_to")),
                start=float(p.get("t0", 0.0)) * duration,
                end=float(p.get("t1", 0.0)) * duration,
                scale=float(p.get("scale", 1.0)),
                action=str(p.get("action", "static")),
                carrier=p.get("carrier"), color=str(p.get("color", "")),
            )
            for p in scene.props
        ]
        return SceneAnimation(
            background_path=bg, placements=scoped, dialogue=spans,
            duration=duration, fps=int(opts.fps), out_w=width, out_h=height,
            camera_move="push_in", title=None, show_table=False,
            seed=opts.seed or plan.prompt[:60],
            actions=actions, props=props, camera_shots=camera, ground_frac=0.94,
        )

    def _write_scene_manifest(self, project: Project, plan: FilmPlan) -> None:
        scenes = []
        for scene in plan.scenes:
            scenes.append({
                "scene_id": scene.scene_id,
                "title": scene.title,
                "narration_text": " ".join(ln.text for ln in scene.lines),
                "duration_seconds": scene.duration_actual or scene.duration_planned,
                "camera_move": "push_in",
                "style": "2d_animation",
                "animation_method": ANIM_2D_METHOD,
                "status": scene.status,
                "assets": {"image": scene.background},
                "subtitle_start": scene.start,
                "subtitle_end": scene.end,
                "notes": scene.continuity,
            })
        project.write_json("scenes", "scenes.json", {
            "scenes": scenes,
            "scene_count": len(scenes),
            "total_estimated_seconds": plan.estimated_seconds,
            "target_seconds": plan.target_seconds,
            "seed": 0,
            "notes": ["Scenes were planned by film mode and timed from measured audio."],
        })

    def _fingerprint(self, project: Project, plan: FilmPlan,
                     opts: FilmOptions) -> str:
        return _fingerprint({
            "plan": plan.to_dict(),
            "options": opts.to_dict(),
            "script_hash": (plan.script or {}).get("sections") and
                           hashlib.sha256(json.dumps(plan.script, sort_keys=True,
                                                     default=str).encode()).hexdigest()[:16],
        })

    def _load_state(self, project: Project) -> dict[str, Any]:
        path = self._state_path(project)
        if not path.exists():
            return {"fingerprint": None, "scenes": {}, "stage": "not_started"}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"fingerprint": None, "scenes": {}, "stage": "not_started"}

    def _save_state(self, project: Project, state: dict[str, Any]) -> None:
        state["updated_at"] = time.time()
        self._state_path(project).write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
