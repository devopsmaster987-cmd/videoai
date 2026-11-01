"""Scene planner: script sections -> validated scene manifest (JSON).

Each scene becomes an independently renderable unit with its own visual
description, camera move, transition, subtitle block, and asset paths
(section 7).  Scene durations are estimated from narration word counts until
real TTS timings are available; the planner then re-syncs them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any

from .constants import (
    CAMERA_MOVES,
    MAX_SCENE_SECONDS,
    MIN_SCENE_SECONDS,
    TARGET_SCENE_SECONDS,
)
from .script_engine import Script, ScriptSection, _words

MANIFEST_NAME = "scenes.json"


@dataclass
class Scene:
    scene_id: str
    title: str
    narration_text: str
    duration_seconds: float
    visual_description: str
    animation_method: str = "animated_illustration"   # or procedural_motion_graphics
    camera_move: str = "push_in"
    transition_in: str = "fade"
    style: str = "illustration"                        # illustration | motion_graphics
    subtitle_text: str = ""
    status: str = "planned"                            # planned|assets_ready|narrated|rendered|failed
    retries: int = 0
    error: str | None = None
    assets: dict[str, str] = field(default_factory=dict)
    narration_file: str | None = None
    subtitle_start: float = 0.0
    subtitle_end: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SceneManifest:
    scenes: list[Scene]
    total_estimated_seconds: float
    target_seconds: float
    seed: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenes": [s.to_dict() for s in self.scenes],
            "total_estimated_seconds": self.total_estimated_seconds,
            "target_seconds": self.target_seconds,
            "seed": self.seed,
            "notes": self.notes,
            "scene_count": len(self.scenes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneManifest":
        return cls(
            scenes=[Scene.from_dict(s) for s in d.get("scenes", [])],
            total_estimated_seconds=d.get("total_estimated_seconds", 0),
            target_seconds=d.get("target_seconds", 0),
            seed=d.get("seed", 0),
            notes=d.get("notes", []),
        )

    def validate(self) -> list[str]:
        """Return a list of problems; empty list means valid."""
        problems: list[str] = []
        if not self.scenes:
            problems.append("scene manifest contains no scenes")
        seen: set[str] = set()
        for s in self.scenes:
            if not s.scene_id:
                problems.append("scene with empty scene_id")
            if s.scene_id in seen:
                problems.append(f"duplicate scene_id: {s.scene_id}")
            seen.add(s.scene_id)
            if s.duration_seconds <= 0:
                problems.append(f"{s.scene_id}: non-positive duration")
            if not s.narration_text.strip() and not s.title.strip():
                problems.append(f"{s.scene_id}: empty narration and title")
        return problems

    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.scenes)


# ---------------------------------------------------------------------------
def _split_section_into_scenes(section: ScriptSection, target_seconds: float, wpm: int) -> list[ScriptSection]:
    """Split one script section into scene-sized chunks of narration.

    A section longer than ~1.6x the target scene length is split at sentence
    boundaries so no scene runs absurdly long.
    """
    words_per_scene = max(15, int(wpm / 60.0 * target_seconds))
    if section.word_count <= words_per_scene * 1.6:
        return [section]

    sentences = [p for p in _split_loose(section.narration)]
    if len(sentences) <= 1:
        return [section]

    chunks: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for sent in sentences:
        sw = _words(sent)
        if current and current_words + sw > words_per_scene * 1.25:
            chunks.append(current)
            current = [sent]
            current_words = sw
        else:
            current.append(sent)
            current_words += sw
    if current:
        chunks.append(current)

    out: list[ScriptSection] = []
    for i, chunk in enumerate(chunks):
        text = " ".join(chunk)
        out.append(
            ScriptSection(
                id=f"{section.id}_p{i + 1}",
                beat=section.beat,
                title=section.title if i == 0 else f"{section.title} (cont.)",
                narration=text,
                word_count=_words(text),
                visual_hint=section.visual_hint,
            )
        )
    return out


def _split_loose(text: str) -> list[str]:
    import re
    return [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if p.strip()]


def _pick_camera_move(rng: random.Random, previous: str | None) -> str:
    """Pick a camera move, avoiding immediate repeats for visual variety."""
    choices = list(CAMERA_MOVES)
    if previous in choices and len(choices) > 1:
        choices.remove(previous)
    return rng.choice(choices)


def _style_for(section: ScriptSection, mode: str, rng: random.Random) -> str:
    """Decide per-scene visual style given the project animation mode."""
    if mode == "illustration":
        return "illustration"
    if mode == "motion_graphics":
        return "motion_graphics"
    if mode == "hybrid":
        # Text-card-ish beats -> motion graphics; others -> illustration.
        graphic_beats = ("origin", "development", "mechanism", "step", "item", "evidence", "summary", "framing", "prereqs", "limits", "implications", "decline", "legacy")
        if any(section.beat.startswith(g) for g in graphic_beats):
            return rng.choice(("motion_graphics", "motion_graphics", "illustration"))
        return "illustration"
    # ai_video mode falls back to hybrid behaviour when no backend is present;
    # the pipeline records the actual method used per scene.
    return "illustration"


def _visual_description(section: ScriptSection, style: str) -> str:
    if style == "motion_graphics":
        return f"Procedural motion graphic: {section.visual_hint or section.title}"
    return f"Illustrated scene: {section.visual_hint or section.title}"


class ScenePlanner:
    def __init__(self, wpm: int = 130, seed: int | None = None):
        self.wpm = wpm
        self.seed = seed

    def plan(
        self,
        script: Script,
        target_seconds: float,
        *,
        animation_mode: str = "hybrid",
        transition: str = "fade",
    ) -> SceneManifest:
        rng = random.Random(self.seed if self.seed is not None else hash(script.topic) & 0xFFFFFFFF)

        # Flatten script sections into scene-sized narration chunks.
        chunks: list[ScriptSection] = []
        for sec in script.sections:
            chunks.extend(_split_section_into_scenes(sec, TARGET_SCENE_SECONDS, script.target_wpm or self.wpm))

        # Estimate raw durations from word counts, then scale to target.
        raw = [max(MIN_SCENE_SECONDS, c.word_count / (script.target_wpm or self.wpm) * 60.0) for c in chunks]
        raw_total = sum(raw) or 1.0
        scale = target_seconds / raw_total if target_seconds > 0 else 1.0

        scenes: list[Scene] = []
        prev_move: str | None = None
        t = 0.0
        for i, (chunk, r) in enumerate(zip(chunks, raw), start=1):
            dur = max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, r * scale))
            move = _pick_camera_move(rng, prev_move)
            prev_move = move
            style = _style_for(chunk, animation_mode, rng)
            scene = Scene(
                scene_id=f"scene_{i:03d}",
                title=chunk.title,
                narration_text=chunk.narration,
                duration_seconds=round(dur, 3),
                visual_description=_visual_description(chunk, style),
                animation_method="animated_illustration" if style == "illustration" else "procedural_motion_graphics",
                camera_move=move,
                transition_in="none" if i == 1 else transition,
                style=style,
                subtitle_text=chunk.narration,
                subtitle_start=0.0,
                subtitle_end=round(dur, 3),
            )
            scenes.append(scene)
            t += dur

        manifest = SceneManifest(
            scenes=scenes,
            total_estimated_seconds=round(t, 3),
            target_seconds=target_seconds,
            seed=self.seed if self.seed is not None else 0,
            notes=[],
        )

        # Report honest divergence instead of silently stretching.
        if target_seconds > 0:
            delta = abs(t - target_seconds) / target_seconds
            if delta > 0.20:
                manifest.notes.append(
                    f"Planned scenes total {t/60:.1f} min vs target {target_seconds/60:.1f} min; "
                    "narration content drives final length (no silent stretching is performed)."
                )
        return manifest

    @staticmethod
    def resync_durations(manifest: SceneManifest, narration_durations: dict[str, float],
                         *, gap: float = 0.35, target_seconds: float | None = None) -> SceneManifest:
        """Set scene durations from real narration audio lengths.

        Scene duration = narration length + small breathing gap, clamped to
        sane bounds.  Subtitle windows track the narration exactly.
        """
        total = 0.0
        for s in manifest.scenes:
            nd = narration_durations.get(s.scene_id)
            if nd and nd > 0:
                # Never clamp below the real narration length, otherwise the
                # final mux (-shortest) would truncate speech.
                dur = max(MIN_SCENE_SECONDS, nd + gap)
            else:
                dur = max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, s.duration_seconds))
            s.duration_seconds = round(dur, 3)
            s.subtitle_start = 0.25 if dur > 1.5 else 0.0
            s.subtitle_end = round(max(s.subtitle_start + 0.5, dur - 0.35), 3)
            total += dur
        manifest.total_estimated_seconds = round(total, 3)
        if target_seconds is not None:
            manifest.target_seconds = target_seconds
        return manifest
