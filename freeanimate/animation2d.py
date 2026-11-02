"""Genuine rigged 2D character animation.

This is not a Ken Burns pan over a still image.  Characters are built from a
small skeletal rig (torso, neck, head, two arms, two legs) and posed *per frame*
with:

* idle breathing and body sway,
* head tilt / turn toward a conversation partner,
* blinking,
* arm gestures while speaking,
* eyebrow and eye expressions,
* viseme-driven mouth shapes synchronized to each line's measured audio span.

Frames are composited over a scene-specific procedural environment and piped
straight into FFmpeg (no temporary frame dumps), then encoded to H.264.  A
camera move (push, pan, drift) is applied by cropping a moving window of the
larger work canvas.

Honesty: this is 2D vector-style animation, not 3D, not AI-generated footage.
The method string reported is ``rigged_2d_character_animation``.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter

from .characters import Character, _hex_rgb, _shade
from .ffmpeg_utils import FFmpegError, find_ffmpeg, has_filter
from .image_utils import normalize_image
from .lipsync import (VisemeEvent, build_viseme_timeline, emotion_mouth,
                      mouth_openness, mouth_width_scale, viseme_at)

METHOD_NAME = "rigged_2d_character_animation"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class CharacterRig:
    id: str
    name: str
    species: str = "human"
    skin: tuple[int, int, int] = (240, 200, 165)
    hair: tuple[int, int, int] = (60, 45, 35)
    clothing: tuple[int, int, int] = (70, 110, 180)
    accent: tuple[int, int, int] = (230, 120, 90)
    eye: tuple[int, int, int] = (50, 60, 80)
    gender: str | None = None
    age_group: str = "adult"
    build: str = "average"
    hair_style: str = "short"
    accessories: tuple[str, ...] = ()
    height_scale: float = 1.0
    clothing_style: str = "shirt"

    @classmethod
    def from_character(cls, character: Character) -> "CharacterRig":
        pal = character.color_palette or ["#f2c9a0", "#5b3a29", "#3b82f6", "#f5f5f5"]
        def g(i: int, fallback: str) -> tuple[int, int, int]:
            try:
                return _hex_rgb(pal[i])
            except IndexError:
                return _hex_rgb(fallback)

        skin = g(0, "#f2c9a0")
        hair = g(1, "#5b3a29")
        clothing = g(2, "#3b82f6")
        accent = _shade(hair, 1.6)
        cloth_text = (character.clothing or "").lower()
        style = "dress" if any(w in cloth_text for w in ("dress", "skirt", "gown")) else (
            "coat" if any(w in cloth_text for w in ("coat", "jacket", "blazer")) else "shirt")
        return cls(
            id=character.id,
            name=character.name,
            species=character.species or "human",
            skin=skin,
            hair=hair,
            clothing=clothing,
            accent=accent,
            eye=_shade(hair, 0.6),
            gender=character.gender,
            age_group=character.age_group or "adult",
            build=character.build or "average",
            hair_style=character.hair_style or "short",
            accessories=tuple(character.accessories or ()),
            height_scale=float(character.height_scale or 1.0),
            clothing_style=style,
        )


@dataclass
class CharacterPlacement:
    rig: CharacterRig
    x_frac: float = 0.38             # horizontal centre in work-canvas fraction
    ground_frac: float = 0.94        # feet line in work-canvas fraction
    height_frac: float = 0.56        # character height in work-canvas fraction
    facing: int = 1                  # 1 faces right, -1 faces left
    emphasis: float = 1.0


@dataclass
class DialogueSpan:
    character_id: str
    text: str
    start: float
    end: float
    emotion: str = "neutral"
    action: str | None = None
    visemes: list[VisemeEvent] = field(default_factory=list)

    @classmethod
    def create(cls, character_id: str, text: str, start: float, duration: float,
               emotion: str = "neutral", action: str | None = None) -> "DialogueSpan":
        return cls(
            character_id=character_id,
            text=text,
            start=start,
            end=start + duration,
            emotion=emotion,
            action=action,
            visemes=build_viseme_timeline(text, duration),
        )


@dataclass
class ActorCue:
    """One scripted full-body action for a character.

    ``x_from``/``x_to`` are work-canvas fractions and produce real movement
    paths (walking/running) instead of sliding a static sprite.  ``action``
    selects the skeletal pose; ``target_character`` makes the actor face and
    approach another actor, and ``prop`` links object interaction.
    """
    character_id: str
    action: str = "idle"
    start: float = 0.0
    end: float = 0.0
    x_from: float | None = None
    x_to: float | None = None
    facing_from: int | None = None
    facing_to: int | None = None
    target_character: str | None = None
    prop: str | None = None
    emotion: str = "neutral"
    note: str = ""


@dataclass
class SceneProp:
    """An interactive prop that participates in the animation."""
    id: str
    kind: str
    x: float = 0.5
    y: float = 0.93
    x_to: float | None = None
    y_to: float | None = None
    start: float = 0.0
    end: float = 0.0
    scale: float = 1.0
    action: str = "static"          # static | roll | bounce | fly | carried
    carrier: str | None = None
    color: str = ""
    note: str = ""


@dataclass
class CameraCue:
    """A cinematic shot inside a scene (cut or smooth move)."""
    kind: str = "wide"              # establishing|wide|medium|close_up|reaction|
                                    # tracking|over_shoulder|low_angle|high_angle|orbit
    movement: str = "push_in"
    start: float = 0.0
    end: float = 0.0
    focus: str | None = None        # character id, prop id, "ensemble", or None
    cut: bool = True


@dataclass
class SceneAnimation:
    background_path: Path
    placements: list[CharacterPlacement]
    dialogue: list[DialogueSpan]
    duration: float
    fps: int = 24
    out_w: int = 1280
    out_h: int = 720
    camera_move: str = "push_in"
    work_scale: float = 1.25
    title: str | None = None
    show_table: bool = False
    table_top_frac: float = 0.74
    seed: str = ""
    actions: list[ActorCue] = field(default_factory=list)
    props: list[SceneProp] = field(default_factory=list)
    camera_shots: list[CameraCue] = field(default_factory=list)
    ground_frac: float = 0.94

    @property
    def characters(self) -> dict[str, CharacterPlacement]:
        """Character id -> placement (ids come from the rig, which is unique)."""
        out: dict[str, CharacterPlacement] = {}
        for p in self.placements:
            out.setdefault(p.rig.id, p)
        return out


@dataclass
class AnimationResult:
    ok: bool
    path: Path | None = None
    frames: int = 0
    error: str | None = None
    method: str = METHOD_NAME


# ---------------------------------------------------------------------------
# Pose model
# ---------------------------------------------------------------------------
@dataclass
class Pose:
    bob: float = 0.0
    sway: float = 0.0
    lean: float = 0.0
    head_tilt: float = 0.0
    head_turn: float = 0.0
    arm_l: float = 8.0
    arm_r: float = 8.0
    elbow_l: float = 10.0
    elbow_r: float = 10.0
    hand_l: float = 0.0
    hand_r: float = 0.0
    eye_open: float = 1.0
    brow: float = 0.0
    mouth: str = "rest"
    walking: float = 0.0
    facing: int = 1
    blush: bool = False
    surprised: bool = False
    running: float = 0.0
    jump: float = 0.0             # 0..1 vertical lift in body heights
    crouch: float = 0.0
    sit: float = 0.0              # 0..1 sitting amount
    dance: float = 0.0
    laugh: float = 0.0
    reach: float = 0.0
    point: float = 0.0
    clap: float = 0.0
    wave: float = 0.0
    throw: float = 0.0
    catch: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    shadow: float = 1.0


# ---------------------------------------------------------------------------
# Frame pose computation
# ---------------------------------------------------------------------------
def _blink_amount(t: float, seed: str, index: int) -> float:
    """Deterministic blink: ~1 every 3.4s, 0.13s long, phase varies per char."""
    phase = (hash(f"{seed}:{index}") % 1000) / 1000.0 * 3.4
    period = 3.4
    local = (t + phase) % period
    if local < 0.13:
        # Smooth open->closed->open.
        x = local / 0.13
        return 1.0 - math.sin(math.pi * x)
    return 1.0


def _speaking_gesture(t: float, start: float, end: float, seed: str) -> tuple[float, float]:
    """Return (arm_l, arm_r) angles for a speaking character."""
    if t < start - 0.1 or t > end + 0.2:
        return 8.0, 8.0
    local = t - start
    # Gentle alternating hand motion while talking; a bigger raise near the start.
    wave = math.sin(local * 3.4 + (hash(seed) % 7))
    raise_amt = max(0.0, 1.0 - local / 1.2)
    arm_l = 14.0 + 10.0 * wave + 18.0 * raise_amt
    arm_r = 12.0 + 8.0 * math.sin(local * 2.7 + 1.7)
    return arm_l, arm_r


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _cue_phase(cue: ActorCue, t: float) -> tuple[bool, float]:
    span = max(0.001, cue.end - cue.start)
    if cue.start <= t <= cue.end:
        return True, min(1.0, max(0.0, (t - cue.start) / span))
    return False, 0.0


def active_cue(actions: list[ActorCue], character_id: str, t: float) -> ActorCue | None:
    """Return the most recent matching cue active at time *t*.

    A cue that has not ended keeps the character in that action.  When cues
    overlap (e.g. a long ``approach`` plus a short ``wave``) the latest cue
    wins, which keeps the planner's explicit hand-offs deterministic.
    """
    matches = [c for c in actions if c.character_id == character_id and c.start <= t]
    if not matches:
        return None
    cue = max(matches, key=lambda c: (c.start, c.end))
    return cue if t <= cue.end + 0.001 else cue


def actor_state_at(actions: list[ActorCue], character_id: str, t: float,
                   base_x: float, base_facing: int,
                   characters: dict[str, CharacterPlacement] | None = None
                   ) -> tuple[float, int, str, float, float]:
    """Return (x_frac, facing, action, phase, move_amount) at time t."""
    cue = active_cue(actions, character_id, t)
    if cue is None:
        return base_x, base_facing, "idle", 0.0, 0.0
    phase = min(1.0, max(0.0, (t - cue.start) / max(0.001, cue.end - cue.start)))
    x = base_x
    if cue.x_from is not None:
        x = cue.x_from
    if cue.x_to is not None:
        x = _lerp(cue.x_from if cue.x_from is not None else base_x,
                  cue.x_to, _smoothstep(phase))
    facing = cue.facing_from if cue.facing_from is not None else base_facing
    if cue.facing_to is not None:
        facing = cue.facing_to
    if cue.target_character and characters and cue.target_character in characters:
        target = characters[cue.target_character]
        if cue.x_to is None:
            x = target.x_frac + (-0.12 if target.x_frac > x else 0.12)
            x = max(0.08, min(0.92, x))
        direction = 1 if target.x_frac >= x else -1
        facing = direction
    move = 0.0
    if cue.action in ("walk", "run", "approach", "leave", "follow", "chase", "play"):
        if cue.x_from is not None and cue.x_to is not None and abs(cue.x_to - cue.x_from) > 0.005:
            move = abs(cue.x_to - cue.x_from) / max(0.4, cue.end - cue.start)
        else:
            move = 0.08 if cue.action != "run" else 0.18
    return x, facing, cue.action, phase, move


def compute_pose(placement: CharacterPlacement, spans: list[DialogueSpan], t: float,
                 seed: str, index: int, *, action: str = "idle", action_phase: float = 0.0,
                 move_amount: float = 0.0, target_x: float | None = None) -> Pose:
    rig = placement.rig
    bob = math.sin(t * 1.7 + index * 1.3) * 0.006
    sway = math.sin(t * 0.9 + index * 2.1) * 0.004
    blink = _blink_amount(t, seed, index)
    active = next((s for s in spans if s.character_id == rig.id and s.start <= t <= s.end), None)
    recently = next((s for s in spans if s.character_id == rig.id and t > s.end and t - s.end < 1.4), None)

    pose = Pose(facing=placement.facing, bob=bob, sway=sway, eye_open=blink)
    pose.head_turn = placement.facing * 0.55
    if target_x is not None:
        pose.gaze_x = max(-1.0, min(1.0, (target_x - placement.x_frac) * 3.0))
        pose.head_turn = pose.gaze_x * 0.65

    if active is not None:
        local = t - active.start
        pose.mouth = viseme_at(active.visemes, local, default="rest")
        pose.arm_l, pose.arm_r = _speaking_gesture(t, active.start, active.end, rig.id)
        pose.head_tilt = math.sin(local * 2.2) * 2.2 * placement.facing
        pose.head_turn = placement.facing * 0.2
        e = active.emotion.lower()
        pose.surprised = e in ("surprised", "shocked")
        pose.brow = 0.7 if pose.surprised else (0.25 if e in ("happy", "amused", "curious") else 0.0)
        pose.blush = e in ("happy", "amused")
        if e in ("laughing", "laugh") and (local % 1.1) < 0.18:
            pose.mouth = "laugh"
    elif recently is not None:
        # Smile briefly after finishing a line.
        pose.mouth = emotion_mouth(recently.emotion)
        pose.head_tilt = math.sin(t * 1.6) * 1.2
        pose.brow = 0.15
    else:
        # Listening: occasional nod and a soft expression.
        pose.mouth = "smile" if (index % 2 == 0) else "rest"
        pose.head_tilt = math.sin(t * 1.1 + index) * 2.6 * placement.facing
        nod = math.sin(t * 2.0 + index * 0.7)
        pose.head_turn = placement.facing * (0.45 + 0.08 * nod)
        pose.brow = 0.1
        pose.eye_open = min(pose.eye_open, 0.95)

    # Head follows the current speaker when someone else is talking.
    speaker = next((s for s in spans if s.start <= t <= s.end), None)
    if speaker is not None and speaker.character_id != rig.id:
        pose.head_turn = placement.facing * 0.7
        pose.eye_open = min(pose.eye_open, 1.0)

    # ------------------------------------------------------------------
    # Full-body action overlay.  Actions are applied on top of the dialogue
    # pose so a character can walk *and* speak, or wave while listening.
    # ------------------------------------------------------------------
    p = max(0.0, min(1.0, action_phase))
    act = (action or "idle").lower()
    pose.shadow = 1.0
    if act in ("walk", "approach", "follow", "leave"):
        pose.walking = p * 2.0 + (move_amount * 3.0)
        pose.arm_l = 10.0 + 14.0 * math.sin(p * math.pi * 4)
        pose.arm_r = 10.0 - 14.0 * math.sin(p * math.pi * 4)
        pose.elbow_l = 12.0
        pose.elbow_r = 12.0
        pose.head_tilt = math.sin(p * math.pi * 6) * 1.5
        pose.mouth = "smile"
    elif act in ("run", "chase", "play"):
        pose.walking = p * 3.0 + 0.5
        pose.running = 1.0
        pose.lean = 7.0 * placement.facing
        pose.arm_l = 22.0 + 26.0 * math.sin(p * math.pi * 6)
        pose.arm_r = 22.0 - 26.0 * math.sin(p * math.pi * 6)
        pose.elbow_l = 45.0
        pose.elbow_r = 45.0
        pose.mouth = "open"
        pose.brow = 0.4
        pose.eye_open = 1.0
    elif act == "turn":
        # A turn is expressed by leaning through the rotation.
        pose.lean = math.sin(p * math.pi) * 9.0 * placement.facing
        pose.head_tilt = math.sin(p * math.pi) * 4.0
        pose.mouth = "rest"
    elif act == "jump":
        pose.jump = math.sin(p * math.pi)
        pose.crouch = max(0.0, 1.0 - abs(p - 0.5) * 4.0) * 0.4
        pose.arm_l = 40.0 + 40.0 * pose.jump
        pose.arm_r = 40.0 + 40.0 * pose.jump
        pose.elbow_l = 15.0
        pose.elbow_r = 15.0
        pose.mouth = "open" if p < 0.5 else "smile"
        pose.brow = 0.5
    elif act == "wave":
        pose.wave = 1.0
        pose.arm_r = 95.0
        pose.elbow_r = 25.0 + 40.0 * abs(math.sin(p * math.pi * 3))
        pose.hand_r = math.sin(p * math.pi * 6) * 35.0
        pose.mouth = "smile"
        pose.head_tilt = -4.0 * placement.facing
        pose.brow = 0.3
    elif act == "greet":
        pose.wave = 1.0
        pose.arm_r = 80.0
        pose.elbow_r = 30.0 + 30.0 * abs(math.sin(p * math.pi * 2))
        pose.hand_r = math.sin(p * math.pi * 4) * 25.0
        pose.mouth = "smile"
        pose.walking = p * 0.6 if p < 0.4 else 0.0
    elif act == "point":
        pose.point = 1.0
        pose.arm_r = 88.0
        pose.elbow_r = 2.0
        pose.hand_r = 0.0
        pose.mouth = "rest"
        pose.brow = 0.35
        pose.head_turn = placement.facing * 0.8
    elif act == "laugh":
        pose.laugh = 1.0
        pose.bob = math.sin(p * math.pi * 8) * 0.012
        pose.lean = math.sin(p * math.pi * 4) * 5.0
        pose.arm_l = 30.0
        pose.arm_r = 30.0
        pose.elbow_l = 55.0
        pose.elbow_r = 55.0
        pose.mouth = "laugh"
        pose.brow = 0.55
        pose.eye_open = 0.75
        pose.blush = True
    elif act == "clap":
        pose.clap = 1.0
        in_out = abs(math.sin(p * math.pi * 6))
        pose.arm_l = 65.0 - 25.0 * in_out
        pose.arm_r = 65.0 - 25.0 * in_out
        pose.elbow_l = 55.0 + 25.0 * in_out
        pose.elbow_r = 55.0 + 25.0 * in_out
        pose.hand_l = -in_out * 20.0
        pose.hand_r = in_out * 20.0
        pose.mouth = "smile"
        pose.brow = 0.4
    elif act == "dance":
        pose.dance = 1.0
        beat = math.sin(p * math.pi * 4)
        pose.sway = beat * 0.02
        pose.lean = beat * 6.0
        pose.bob = abs(math.sin(p * math.pi * 4)) * 0.014
        pose.arm_l = 30.0 + 45.0 * max(0.0, beat)
        pose.arm_r = 30.0 - 45.0 * min(0.0, beat)
        pose.elbow_l = 35.0
        pose.elbow_r = 35.0
        pose.walking = abs(beat) * 1.2
        pose.mouth = "smile"
        pose.brow = 0.5
        pose.blush = True
    elif act in ("sit", "seated"):
        pose.sit = 1.0
        pose.arm_l = 18.0
        pose.arm_r = 18.0
        pose.elbow_l = 40.0
        pose.elbow_r = 40.0
        pose.mouth = "smile" if active else "rest"
    elif act in ("stand", "stand_up"):
        pose.sit = max(0.0, 1.0 - p * 1.4)
        pose.crouch = max(0.0, 1.0 - p * 1.6) * 0.5
        pose.arm_l = 12.0
        pose.arm_r = 12.0
        pose.mouth = "rest"
    elif act in ("reach", "pick_up", "put_down"):
        pose.reach = 1.0
        reach_pose = 1.0 if act == "reach" else math.sin(p * math.pi)
        pose.arm_r = 55.0 + 35.0 * reach_pose
        pose.elbow_r = 45.0 - 30.0 * reach_pose
        pose.lean = 8.0 * reach_pose * placement.facing
        pose.crouch = reach_pose * 0.35
        pose.mouth = "rest"
        pose.brow = 0.2
    elif act in ("throw", "catch"):
        pose.throw = 1.0 if act == "throw" else 0.0
        pose.catch = 1.0 if act == "catch" else 0.0
        windup = math.sin(min(1.0, p * 1.6) * math.pi)
        if act == "throw":
            pose.arm_r = 40.0 + 80.0 * (1.0 - windup)
            pose.elbow_r = 20.0 + 40.0 * windup
            pose.lean = 6.0 * windup * placement.facing
        else:
            pose.arm_r = 75.0
            pose.arm_l = 55.0
            pose.elbow_r = 30.0
            pose.elbow_l = 30.0
        pose.mouth = "open" if p < 0.6 else "smile"
        pose.brow = 0.45
    elif act in ("react", "surprised"):
        pose.surprised = True
        pose.brow = 0.9
        pose.eye_open = 1.15
        pose.arm_l = 25.0
        pose.arm_r = 25.0
        pose.elbow_l = 30.0
        pose.elbow_r = 30.0
        pose.mouth = "open"
        pose.lean = -3.0 * placement.facing
    elif act in ("listen", "watch"):
        pose.mouth = "smile" if (index % 2 == 0) else "rest"
        pose.brow = 0.15
        pose.head_tilt = math.sin(t * 1.2) * 3.0 * placement.facing
    elif act in ("speak", "talk"):
        pose.arm_l, pose.arm_r = _speaking_gesture(t, 0.0, 9999.0, rig.id)
        pose.mouth = ("aa", "ee", "oo", "mm")[int(p * 8) % 4]

    if pose.jump > 0:
        pose.shadow = max(0.35, 1.0 - pose.jump * 0.5)
    if pose.sit > 0.6:
        pose.shadow = 0.8

    # Lip-sync wins while the character is actually delivering a line: the
    # action overlay may have set an expression mouth, but the viseme from the
    # measured audio timeline drives the mouth whenever a span is active.
    if active is not None and act not in ("laugh",):
        pose.mouth = viseme_at(active.visemes, t - active.start, default="rest")

    return pose


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _rounded_line(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float],
                  width: int, fill: tuple[int, int, int]) -> None:
    draw.line([p0, p1], fill=fill, width=width)
    r = width // 2
    for (x, y) in (p0, p1):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _limb(draw: ImageDraw.ImageDraw, origin: tuple[float, float], angle_deg: float,
          length: float, width: int, fill: tuple[int, int, int]) -> tuple[float, float]:
    a = math.radians(angle_deg)
    end = (origin[0] + math.sin(a) * length, origin[1] + math.cos(a) * length)
    _rounded_line(draw, origin, end, width, fill)
    return end


def _draw_mouth(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, viseme: str,
                skin: tuple[int, int, int]) -> None:
    lip = _shade(skin, 0.55)
    inner = (90, 35, 45)
    open_amt = mouth_openness(viseme)
    width_scale = mouth_width_scale(viseme)
    w = r * 0.34 * width_scale
    h = r * 0.22 * open_amt
    if viseme in ("smile",):
        draw.arc([cx - w, cy - h, cx + w, cy + h * 1.5], 15, 165, fill=lip,
                 width=max(2, int(r * 0.1)))
        return
    if viseme in ("rest", "mm"):
        draw.line([(cx - w * 0.7, cy), (cx + w * 0.7, cy)], fill=lip,
                  width=max(2, int(r * 0.1)))
        return
    if h < r * 0.07:
        h = r * 0.07
    draw.ellipse([cx - w, cy - h, cx + w, cy + h], fill=inner)
    if viseme == "laugh":
        draw.ellipse([cx - w * 0.55, cy + h * 0.15, cx + w * 0.55, cy + h * 0.85],
                     fill=(220, 90, 100))
    # Upper lip highlight.
    draw.arc([cx - w, cy - h, cx + w, cy + h], 190, 350, fill=lip,
             width=max(2, int(r * 0.09)))


def draw_character(rig: CharacterRig, pose: Pose, height: float, *, supersample: int = 2) -> Image.Image:
    """Render one posed character on a transparent RGBA layer (feet near bottom).

    Body proportions follow the character's ``age_group`` and ``build`` (a
    child is head-heavy with short legs, a stocky adult is wider at the torso)
    and the drawing includes hands, shoes, hair styles, accessories, clothing
    detail and a soft ground shadow so characters read as designed figures
    rather than identical silhouettes.
    """
    H = max(40, int(height))
    cw = int(H * 1.02)
    ch = int(H * 1.22)
    ss = max(1, int(supersample))
    img = Image.new("RGBA", (cw * ss, ch * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    W = cw * ss
    HH = ch * ss
    Hh = H * ss
    cx = W / 2 + pose.sway * Hh
    feet_y = HH - 0.06 * Hh + pose.bob * Hh

    # --- proportions from age / build -------------------------------------
    age = (rig.age_group or "adult").lower()
    build = (rig.build or "average").lower()
    if age == "child":
        head_r = 0.125 * Hh
        hip_off, shoulder_off, head_off = 0.40, 0.655, 0.815
    elif age == "teen":
        head_r = 0.113 * Hh
        hip_off, shoulder_off, head_off = 0.415, 0.685, 0.845
    elif age == "elderly":
        head_r = 0.103 * Hh
        hip_off, shoulder_off, head_off = 0.41, 0.675, 0.855
    else:
        head_r = 0.105 * Hh
        hip_off, shoulder_off, head_off = 0.42, 0.70, 0.86
    if build in ("tall", "slim"):
        hip_off, shoulder_off = hip_off + 0.02, shoulder_off + 0.02
    if build == "stocky":
        hip_off, shoulder_off = hip_off - 0.01, shoulder_off - 0.01
    if pose.sit > 0.01:
        hip_off *= (1.0 - 0.16 * pose.sit)
        shoulder_off *= (1.0 - 0.10 * pose.sit)

    hip_y = feet_y - hip_off * Hh
    shoulder_y = feet_y - shoulder_off * Hh
    head_cy = feet_y - head_off * Hh

    skin = rig.skin
    hair = rig.hair
    cloth = rig.clothing
    accent = rig.accent
    outline = _shade(skin, 0.55)
    lw = max(2, int(Hh * 0.012))

    # --- ground shadow -----------------------------------------------------
    if pose.shadow > 0.05:
        sh = int(255 * pose.shadow * 0.30)
        rx = Hh * (0.15 + 0.03 * pose.crouch)
        d.ellipse([cx - rx, feet_y - Hh * 0.018, cx + rx, feet_y + Hh * 0.022],
                  fill=(0, 0, 0, sh))

    # --- legs --------------------------------------------------------------
    leg_w = max(4, int(Hh * 0.055))
    shoe_col = _shade(hair, 0.42)

    def _shoe(fx: float, fy: float, angle: float, size: float) -> None:
        """A rounded shoe pointing along *angle* (degrees from vertical)."""
        rad = math.radians(angle)
        tx = fx + math.sin(rad) * size * 0.55
        ty = fy + math.cos(rad) * size * 0.35
        d.line([(fx, fy), (tx, ty)], fill=shoe_col, width=leg_w)
        d.ellipse([tx - size * 0.42, ty - size * 0.22, tx + size * 0.52, ty + size * 0.28],
                  fill=shoe_col)

    leg_len = 0.34 * Hh * (1.0 - 0.30 * pose.sit)
    if pose.jump > 0.05:
        # Tucked legs while airborne.
        tuck = pose.jump
        for sign in (-1, 1):
            kx = cx + sign * Hh * 0.05
            _limb(d, (kx, hip_y), sign * 10 + 55 * tuck, leg_len * 0.55, leg_w, _shade(cloth, 0.75))
            _limb(d, (kx + math.sin(math.radians(sign * 10 + 55 * tuck)) * leg_len * 0.55,
                      hip_y + math.cos(math.radians(sign * 10 + 55 * tuck)) * leg_len * 0.55),
                  sign * 10 - 35 * tuck, leg_len * 0.5, leg_w, _shade(cloth, 0.78))
    elif pose.sit > 0.5:
        # Seated: thighs forward, shins down.
        for sign in (-1, 1):
            kx = cx + sign * Hh * 0.05
            kx2 = kx + sign * Hh * 0.16
            d.line([(kx, hip_y), (kx2, hip_y + Hh * 0.02)], fill=_shade(cloth, 0.75), width=leg_w)
            d.line([(kx2, hip_y + Hh * 0.02), (kx2, feet_y - Hh * 0.02)],
                   fill=_shade(cloth, 0.78), width=leg_w)
            _shoe(kx2, feet_y - Hh * 0.01, sign * 4, Hh * 0.075)
    elif pose.walking > 0:
        swing = math.sin(pose.walking * math.pi * 2)
        amp = 26.0 if pose.running > 0 else 16.0
        for sign, sw in ((-1, swing), (1, -swing)):
            ang = 6 + amp * sw
            _limb(d, (cx + sign * Hh * 0.05, hip_y), ang, leg_len, leg_w, _shade(cloth, 0.75))
            fx = cx + sign * Hh * 0.05 + math.sin(math.radians(ang)) * leg_len
            fy = hip_y + math.cos(math.radians(ang)) * leg_len
            _shoe(fx, fy, ang, Hh * 0.08)
    else:
        for sign in (-1, 1):
            ang = 5 * sign
            _limb(d, (cx + sign * Hh * 0.055, hip_y), ang, leg_len, leg_w, _shade(cloth, 0.78))
            fx = cx + sign * Hh * 0.055 + math.sin(math.radians(ang)) * leg_len
            fy = hip_y + math.cos(math.radians(ang)) * leg_len
            _shoe(fx, fy, ang, Hh * 0.075)

    # --- torso -------------------------------------------------------------
    torso_top = shoulder_y
    torso_bottom = hip_y
    tw_top = Hh * (0.145 if build == "stocky" else 0.115 if build == "slim" else 0.13)
    tw_bot = Hh * (0.115 if build == "stocky" else 0.09 if build == "slim" else 0.10)
    if rig.clothing_style == "dress":
        tw_bot *= 1.55
    torso = [
        (cx - tw_top, torso_top), (cx + tw_top, torso_top),
        (cx + tw_bot, torso_bottom), (cx - tw_bot, torso_bottom),
    ]
    d.polygon(torso, fill=cloth)
    # Collar / shirt detail.
    d.polygon([(cx - tw_top * 0.5, torso_top), (cx + tw_top * 0.5, torso_top),
               (cx, torso_top + Hh * 0.07)], fill=_shade(cloth, 1.25))
    if rig.clothing_style == "coat":
        # Coat lapels + buttons.
        d.polygon([(cx - tw_top * 0.55, torso_top), (cx - tw_top * 0.05, torso_top),
                   (cx - tw_top * 0.3, torso_bottom)], fill=_shade(cloth, 1.18))
        d.polygon([(cx + tw_top * 0.55, torso_top), (cx + tw_top * 0.05, torso_top),
                   (cx + tw_top * 0.3, torso_bottom)], fill=_shade(cloth, 1.18))
        for k in range(3):
            by = torso_top + (torso_bottom - torso_top) * (0.35 + 0.2 * k)
            d.ellipse([cx - Hh * 0.012, by - Hh * 0.012, cx + Hh * 0.012, by + Hh * 0.012],
                      fill=accent)
    elif rig.clothing_style == "dress":
        # Skirt hem shading.
        d.polygon([(cx - tw_bot, torso_bottom - Hh * 0.03),
                   (cx + tw_bot, torso_bottom - Hh * 0.03),
                   (cx + tw_bot * 1.05, torso_bottom + Hh * 0.02),
                   (cx - tw_bot * 1.05, torso_bottom + Hh * 0.02)],
                  fill=_shade(cloth, 0.85))
    else:
        # Belt.
        d.rectangle([cx - tw_bot, torso_bottom - Hh * 0.02, cx + tw_bot, torso_bottom],
                    fill=_shade(cloth, 0.7))
        # Pocket + seam detail.
        d.rectangle([cx + tw_top * 0.25, torso_top + Hh * 0.10,
                     cx + tw_top * 0.7, torso_top + Hh * 0.17],
                    outline=_shade(cloth, 0.7), width=max(1, lw // 2))

    # --- arms --------------------------------------------------------------
    arm_w = max(3, int(Hh * 0.045))
    shoulder_l = (cx - tw_top * 0.9, torso_top + Hh * 0.02)
    shoulder_r = (cx + tw_top * 0.9, torso_top + Hh * 0.02)
    hand_r = Hh * 0.032
    pose_hand_l = math.radians(pose.hand_l)
    pose_hand_r = math.radians(pose.hand_r)

    elbow_l = _limb(d, shoulder_l, -pose.arm_l, 0.16 * Hh, arm_w, cloth)
    fore_ang_l = -pose.arm_l - pose.elbow_l
    hand_l = _limb(d, elbow_l, fore_ang_l, 0.15 * Hh, arm_w, skin)
    elbow_r = _limb(d, shoulder_r, pose.arm_r, 0.16 * Hh, arm_w, cloth)
    fore_ang_r = pose.arm_r + pose.elbow_r
    hand_rr = _limb(d, elbow_r, fore_ang_r, 0.15 * Hh, arm_w, skin)

    def _hand(hx: float, hy: float, ang: float, spread: float) -> None:
        """Palm + fingers at the end of an arm."""
        d.ellipse([hx - hand_r, hy - hand_r, hx + hand_r, hy + hand_r], fill=skin)
        for k in (-1, 0, 1):
            fa = math.radians(ang + k * spread)
            fx = hx + math.sin(fa) * hand_r * 1.5
            fy = hy + math.cos(fa) * hand_r * 1.5
            d.line([(hx, hy), (fx, fy)], fill=_shade(skin, 0.95),
                   width=max(2, int(hand_r * 0.45)))

    if pose.point > 0.5 or pose.reach > 0.5:
        spread = 8.0
    elif pose.clap > 0.5:
        spread = 20.0
    else:
        spread = 14.0
    _hand(hand_l[0], hand_l[1], fore_ang_l + pose.hand_l, spread)
    _hand(hand_rr[0], hand_rr[1], fore_ang_r + pose.hand_r, spread)

    # --- neck --------------------------------------------------------------
    d.rectangle([cx - Hh * 0.028, head_cy + head_r * 0.55, cx + Hh * 0.028, shoulder_y + Hh * 0.01],
                fill=_shade(skin, 0.92))

    # --- head --------------------------------------------------------------
    hc = (cx + pose.head_turn * head_r * 0.25, head_cy)
    if rig.species == "robot":
        d.rounded_rectangle([hc[0] - head_r, hc[1] - head_r, hc[0] + head_r, hc[1] + head_r],
                            radius=head_r * 0.28, fill=_shade(skin, 0.9), outline=hair, width=lw)
        d.rectangle([hc[0] - head_r * 1.35, hc[1] - head_r * 1.3, hc[0] + head_r * 1.35, hc[1] - head_r * 1.05],
                    fill=_shade(skin, 0.7))
        d.line([(hc[0], hc[1] - head_r), (hc[0], hc[1] - head_r * 1.6)], fill=accent, width=lw)
        d.ellipse([hc[0] - head_r * 0.18, hc[1] - head_r * 1.75,
                   hc[0] + head_r * 0.18, hc[1] - head_r * 1.4], fill=accent)
    else:
        d.ellipse([hc[0] - head_r, hc[1] - head_r, hc[0] + head_r, hc[1] + head_r], fill=skin)
        # Ears.
        for sign in (-1, 1):
            d.ellipse([hc[0] + sign * head_r * 0.98 - head_r * 0.14, hc[1] - head_r * 0.1,
                       hc[0] + sign * head_r * 0.98 + head_r * 0.14, hc[1] + head_r * 0.24],
                      fill=_shade(skin, 0.95))
        if rig.species == "animal":
            ear_r = head_r * 0.42
            for sign in (-1, 1):
                ex = hc[0] + sign * head_r * 0.72
                d.ellipse([ex - ear_r, hc[1] - head_r - ear_r * 0.5 + ex * 0, ex + ear_r, hc[1] - head_r + ear_r],
                          fill=hair)
            d.ellipse([hc[0] - head_r * 0.34, hc[1] + head_r * 0.12, hc[0] + head_r * 0.34, hc[1] + head_r * 0.7],
                      fill=_shade(skin, 1.06))
            d.ellipse([hc[0] - head_r * 0.12, hc[1] + head_r * 0.24, hc[0] + head_r * 0.12, hc[1] + head_r * 0.4],
                      fill=_shade(hair, 0.5))
        elif rig.species == "creature":
            for sign in (-1, 1):
                bx = hc[0] + sign * head_r * 0.55
                d.polygon([(bx, hc[1] - head_r * 0.9),
                           (bx + sign * head_r * 0.3, hc[1] - head_r * 1.5),
                           (bx + sign * head_r * 0.1, hc[1] - head_r * 0.85)], fill=hair)

        # --- hair styles ---------------------------------------------------
        style = (rig.hair_style or "short").lower()
        wears_hat = style == "hat" or any("hat" in a for a in (rig.accessories or ()))
        if wears_hat:
            # A hat covers the scalp; draw a thin hair edge under it.
            if style != "bald":
                d.chord([hc[0] - head_r, hc[1] - head_r * 1.05, hc[0] + head_r,
                         hc[1] + head_r * 0.4], 180, 360, fill=hair)
            d.chord([hc[0] - head_r * 1.05, hc[1] - head_r * 1.2, hc[0] + head_r * 1.05,
                     hc[1] + head_r * 0.2], 180, 360, fill=accent)
            d.rectangle([hc[0] - head_r * 1.4, hc[1] - head_r * 0.35,
                         hc[0] + head_r * 1.4, hc[1] - head_r * 0.1], fill=_shade(accent, 0.85))
        elif style == "bald":
            pass
        elif style == "curly":
            for k in range(9):
                a = math.pi + math.pi * k / 8
                bx = hc[0] + math.cos(a) * head_r * 0.95
                by = hc[1] + math.sin(a) * head_r * 1.0
                rr = head_r * 0.30
                d.ellipse([bx - rr, by - rr, bx + rr, by + rr], fill=hair)
        elif style == "long":
            d.chord([hc[0] - head_r, hc[1] - head_r * 1.12, hc[0] + head_r, hc[1] + head_r * 0.5],
                    180, 360, fill=hair)
            for sign in (-1, 1):
                d.rounded_rectangle([hc[0] + sign * head_r * 0.92 - head_r * 0.20,
                                     hc[1] - head_r * 0.3,
                                     hc[0] + sign * head_r * 0.92 + head_r * 0.20,
                                     hc[1] + head_r * 1.35], radius=head_r * 0.18, fill=hair)
        elif style == "ponytail":
            d.chord([hc[0] - head_r, hc[1] - head_r * 1.12, hc[0] + head_r, hc[1] + head_r * 0.5],
                    180, 360, fill=hair)
            # Tail sweeping behind the head.
            tail_x = hc[0] - head_r * 1.05
            d.ellipse([tail_x - head_r * 0.28, hc[1] - head_r * 0.45,
                       tail_x + head_r * 0.28, hc[1] + head_r * 0.95], fill=hair)
            d.ellipse([tail_x - head_r * 0.24, hc[1] + head_r * 0.7,
                       tail_x + head_r * 0.24, hc[1] + head_r * 1.25], fill=_shade(hair, 0.85))
        elif style == "bun":
            d.chord([hc[0] - head_r, hc[1] - head_r * 1.12, hc[0] + head_r, hc[1] + head_r * 0.5],
                    180, 360, fill=hair)
            d.ellipse([hc[0] - head_r * 0.36, hc[1] - head_r * 1.5,
                       hc[0] + head_r * 0.36, hc[1] - head_r * 0.82], fill=hair)
        else:  # short
            d.chord([hc[0] - head_r, hc[1] - head_r * 1.12, hc[0] + head_r, hc[1] + head_r * 0.5],
                    180, 360, fill=hair)
            for sign in (-1, 1):
                d.rounded_rectangle([hc[0] + sign * head_r * 0.82 - head_r * 0.15,
                                     hc[1] - head_r * 0.2,
                                     hc[0] + sign * head_r * 0.82 + head_r * 0.15,
                                     hc[1] + head_r * 0.7], radius=head_r * 0.15, fill=hair)

    # --- face --------------------------------------------------------------
    eye_y = hc[1] - head_r * 0.02
    eye_dx = head_r * 0.36
    eye_r = head_r * 0.15 * (1.15 if pose.surprised else 1.0)
    open_amt = max(0.0, min(1.0, pose.eye_open))
    gaze_dx = pose.gaze_x * eye_r * 0.45
    gaze_dy = pose.gaze_y * eye_r * 0.35
    for sign in (-1, 1):
        ex = hc[0] + sign * eye_dx
        if open_amt <= 0.12:
            d.line([(ex - eye_r, eye_y), (ex + eye_r, eye_y)],
                   fill=_shade(hair, 0.45), width=max(2, int(head_r * 0.08)))
        else:
            eh = eye_r * open_amt
            d.ellipse([ex - eye_r, eye_y - eh, ex + eye_r, eye_y + eh], fill=(255, 255, 255))
            d.ellipse([ex - eye_r * 0.55 + gaze_dx, eye_y - eh * 0.55 + gaze_dy,
                       ex + eye_r * 0.55 + gaze_dx, eye_y + eh * 0.55 + gaze_dy], fill=rig.eye)
            d.ellipse([ex - eye_r * 0.22 + gaze_dx, eye_y - eh * 0.22 + gaze_dy,
                       ex + eye_r * 0.22 + gaze_dx, eye_y + eh * 0.22 + gaze_dy], fill=(15, 15, 20))
            # Catchlight.
            d.ellipse([ex - eye_r * 0.1 + gaze_dx, eye_y - eh * 0.45 + gaze_dy,
                       ex + eye_r * 0.05 + gaze_dx, eye_y - eh * 0.28 + gaze_dy], fill=(255, 255, 255))
        # Lids / eyelashes for expressive eyes.
        if open_amt < 0.99 and open_amt > 0.12:
            d.arc([ex - eye_r, eye_y - eh, ex + eye_r, eye_y + eh], 180, 360,
                  fill=_shade(hair, 0.5), width=max(1, int(head_r * 0.05)))
        # Brows.
        brow_lift = head_r * 0.32 * pose.brow
        brow_y = eye_y - eye_r * 2.4 - brow_lift
        tilt = head_r * 0.08 * (-pose.brow if pose.brow > 0 else 0)
        d.line([(ex - eye_r * 1.2, brow_y + tilt), (ex + eye_r * 1.2, brow_y - tilt)],
               fill=_shade(hair, 0.55), width=max(2, int(head_r * 0.09)))
    # Nose.
    d.line([(hc[0], hc[1] + head_r * 0.15), (hc[0] - head_r * 0.07, hc[1] + head_r * 0.34)],
           fill=_shade(skin, 0.8), width=max(2, int(head_r * 0.07)))
    # Mouth.
    _draw_mouth(d, hc[0], hc[1] + head_r * 0.52, head_r, pose.mouth, skin)
    # Blush.
    if pose.blush:
        for sign in (-1, 1):
            bx = hc[0] + sign * head_r * 0.62
            d.ellipse([bx - head_r * 0.16, hc[1] + head_r * 0.2,
                       bx + head_r * 0.16, hc[1] + head_r * 0.42], fill=(240, 140, 150, 120))

    # --- accessories -------------------------------------------------------
    acc = [a.lower() for a in (rig.accessories or ())]
    if "glasses" in acc:
        gy = eye_y
        for sign in (-1, 1):
            gx = hc[0] + sign * eye_dx
            d.rounded_rectangle([gx - eye_r * 1.5, gy - eye_r * 1.5,
                                 gx + eye_r * 1.5, gy + eye_r * 1.5],
                                radius=eye_r * 0.5, outline=_shade(hair, 0.4),
                                width=max(2, int(head_r * 0.07)))
        d.line([(hc[0] - eye_dx + eye_r * 1.5, gy), (hc[0] + eye_dx - eye_r * 1.5, gy)],
               fill=_shade(hair, 0.4), width=max(2, int(head_r * 0.06)))
    if "beard" in acc:
        d.chord([hc[0] - head_r * 0.72, hc[1] + head_r * 0.1,
                 hc[0] + head_r * 0.72, hc[1] + head_r * 1.05], 0, 180, fill=hair)
    if "scarf" in acc:
        d.rounded_rectangle([cx - tw_top * 0.9, shoulder_y - Hh * 0.02,
                             cx + tw_top * 0.9, shoulder_y + Hh * 0.06],
                            radius=Hh * 0.02, fill=accent)

    # Apply body lean by rotating the layer around its centre.
    if abs(pose.lean) > 0.5:
        img = img.rotate(pose.lean, resample=Image.BICUBIC, expand=False)
    if ss > 1:
        img = img.resize((cw, ch), Image.LANCZOS)
    # Flip for facing direction.
    if pose.facing < 0:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
SHOT_FRAMING: dict[str, tuple[float, float]] = {
    # kind -> (zoom base, vertical centre bias 0..1 of work height).
    # Characters stand with their feet near 0.94 of the canvas, so a centre of
    # 0.5 leaves two thirds of the frame as empty sky; these biases frame the
    # actors instead of the stage.
    "establishing": (1.00, 0.52),
    "wide": (1.00, 0.55),
    "medium": (1.34, 0.60),
    "close_up": (2.05, 0.50),
    "reaction": (1.75, 0.52),
    "over_shoulder": (1.60, 0.55),
    "low_angle": (1.22, 0.58),
    "high_angle": (1.22, 0.68),
    "tracking": (1.18, 0.60),
    "orbit": (1.12, 0.56),
}


def camera_at(move: str, t_frac: float, work_w: int, work_h: int,
              out_w: int, out_h: int, *,
              shot: str | None = None,
              focus_x: float | None = None,
              focus_y: float | None = None,
              focus_weight: float = 0.85) -> tuple[int, int, int, int]:
    """Return the crop box (x0, y0, x1, y1) on the work canvas for time fraction.

    ``shot`` biases the framing (close-ups frame tighter and higher, high-angle
    shots bias downward) and ``focus_x``/``focus_y`` (work-canvas 0..1) pull the
    frame centre onto a character or prop, so a shot can genuinely hold on an
    actor instead of always centre-framing the stage.
    """
    t = min(1.0, max(0.0, t_frac))
    zoom = 1.0
    cx = work_w / 2
    cy = work_h / 2
    pad_x = (work_w - work_w / 1.14) / 2
    pad_y = (work_h - work_h / 1.14) / 2
    if move == "push_in":
        zoom = 1.0 + 0.14 * t
    elif move == "pull_out":
        zoom = 1.14 - 0.14 * t
    elif move == "slow_zoom_center":
        zoom = 1.06 + 0.06 * math.sin(math.pi * t)
    elif move == "pan_right":
        zoom = 1.08
        cx = work_w / 2 + (pad_x * 0.9) * (t - 0.5) * 2
    elif move == "pan_left":
        zoom = 1.08
        cx = work_w / 2 - (pad_x * 0.9) * (t - 0.5) * 2
    elif move == "pan_down":
        zoom = 1.08
        cy = work_h / 2 + (pad_y * 0.9) * (t - 0.5) * 2
    elif move == "pan_up":
        zoom = 1.08
        cy = work_h / 2 - (pad_y * 0.9) * (t - 0.5) * 2
    elif move == "drift_diagonal":
        zoom = 1.04 + 0.10 * t
        cx = work_w / 2 + (pad_x * 0.7) * (t - 0.5) * 2
        cy = work_h / 2 + (pad_y * 0.7) * (t - 0.5) * 2
    elif move == "orbit":
        zoom = 1.12
        cx = work_w / 2 + math.sin(t * math.pi * 2) * pad_x * 0.8
        cy = work_h / 2 - math.sin(t * math.pi) * pad_y * 0.3
    else:
        zoom = 1.0 + 0.08 * t

    # Shot framing: multiplier on zoom plus a vertical bias for the framing.
    if shot:
        s_zoom, s_bias = SHOT_FRAMING.get(shot, SHOT_FRAMING["wide"])
        zoom *= s_zoom
        cy = work_h * (0.5 + (s_bias - 0.5) * 0.9)

    # Focus: pull the centre toward the target (or push away if it is centred).
    # Close shots aim at the face, which sits above the torso focus point.
    if focus_x is not None and focus_y is not None:
        w = max(0.0, min(1.0, focus_weight))
        fy = focus_y - (0.10 if shot in ("close_up", "reaction") else 0.0)
        cx += (focus_x * work_w - cx) * w
        cy += (fy * work_h - cy) * w

    crop_w = work_w / zoom
    crop_h = work_h / zoom
    # Keep the crop's aspect ratio equal to the output.
    target_ratio = out_w / out_h
    if crop_w / crop_h > target_ratio:
        crop_w = crop_h * target_ratio
    else:
        crop_h = crop_w / target_ratio
    x0 = cx - crop_w / 2
    y0 = cy - crop_h / 2
    if shot and shot not in ("close_up", "reaction"):
        # Keep the ground line (where the feet are) inside the frame instead of
        # letting the bottom edge slice the actors at the shins.
        y0 = min(y0, work_h - crop_h)
    x0 = max(0, min(work_w - crop_w, x0))
    y0 = max(0, min(work_h - crop_h, y0))
    return int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)


def _draw_table_foreground(frame: Image.Image, work_w: int, work_h: int,
                           crop: tuple[int, int, int, int], top_frac: float) -> None:
    """Draw a wooden table surface (world-space) over the characters' legs."""
    x0, y0, x1, y1 = crop
    sx = frame.width / max(1, (x1 - x0))
    sy = frame.height / max(1, (y1 - y0))
    wx0 = work_w * 0.14
    wx1 = work_w * 0.86
    wy = work_h * top_frac
    px0 = (wx0 - x0) * sx
    px1 = (wx1 - x0) * sx
    py = (wy - y0) * sy
    if py >= frame.height:
        return
    d = ImageDraw.Draw(frame)
    wood = (150, 100, 60)
    # Table top (a thick rounded slab).
    d.rectangle([px0, py, px1, min(frame.height, py + max(10, frame.height * 0.06))], fill=wood)
    d.rectangle([px0, py, px1, py + max(3, frame.height * 0.012)], fill=_shade(wood, 1.2))
    d.rectangle([px0, min(frame.height, py + max(10, frame.height * 0.06)) - 3, px1,
                 min(frame.height, py + max(10, frame.height * 0.06))], fill=_shade(wood, 0.7))
    # Front panel down to bottom.
    d.rectangle([px0, min(frame.height, py + max(10, frame.height * 0.06)), px1, frame.height],
                fill=_shade(wood, 0.8))
    # Grain lines.
    for k in range(1, 4):
        gy = py + (frame.height - py) * k / 4
        d.line([(px0, gy), (px1, gy)], fill=_shade(wood, 0.7), width=1)


def draw_prop(prop: SceneProp, size_px: int, *, t: float | None = None) -> Image.Image | None:
    """Render a prop as a transparent RGBA sprite of height ``size_px``.

    Returns ``None`` for an empty/unknown kind so callers can skip quietly.
    """
    kind = (prop.kind or "").strip().lower()
    if not kind:
        return None
    S = max(8, int(size_px))
    ss = 2
    img = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = S * ss
    cx, cy = w / 2, w / 2
    r = w * 0.36 * max(0.2, prop.scale)
    base = _hex_color(prop.color) or _PROP_COLORS.get(_canon(kind), (210, 140, 60))

    if _canon(kind) in ("ball", "sphere", "football", "soccer ball", "beachball"):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base,
                  outline=_shade(base, 0.55), width=max(2, int(r * 0.10)))
        # Panel seams.
        d.arc([cx - r * 0.7, cy - r, cx + r * 0.7, cy + r], 0, 360,
              fill=_shade(base, 0.7), width=max(1, int(r * 0.08)))
        d.arc([cx - r, cy - r * 0.7, cx + r, cy + r * 0.7], 90, 270,
              fill=_shade(base, 0.7), width=max(1, int(r * 0.08)))
        d.ellipse([cx - r * 0.55, cy - r * 0.75, cx - r * 0.2, cy - r * 0.4],
                  fill=(255, 255, 255, 90))
    elif _canon(kind) in ("box", "package", "book", "gift", "crate", "bottle",
                          "cup", "phone", "phone screen"):
        if _canon(kind) in ("bottle", "cup"):
            bw = r * 0.9
            if _canon(kind) == "cup":
                d.polygon([(cx - bw, cy - r), (cx + bw, cy - r),
                           (cx + bw * 0.7, cy + r), (cx - bw * 0.7, cy + r)], fill=base)
            else:
                d.rounded_rectangle([cx - bw * 0.5, cy - r, cx + bw * 0.5, cy + r],
                                    radius=bw * 0.3, fill=base)
                d.rectangle([cx - bw * 0.35, cy - r * 1.3, cx + bw * 0.35, cy - r * 0.9],
                            fill=_shade(base, 0.8))
            d.line([(cx - bw * 0.6, cy - r * 0.5), (cx + bw * 0.6, cy - r * 0.5)],
                   fill=_shade(base, 1.2), width=max(1, int(r * 0.12)))
        else:
            d.rounded_rectangle([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8],
                                radius=r * 0.12, fill=base,
                                outline=_shade(base, 0.5), width=max(2, int(r * 0.08)))
            # Ribbon / tape.
            d.rectangle([cx - r * 0.12, cy - r * 0.8, cx + r * 0.12, cy + r * 0.8],
                        fill=_shade(base, 1.3))
            d.rectangle([cx - r, cy - r * 0.12, cx + r, cy + r * 0.12],
                        fill=_shade(base, 1.3))
    elif _canon(kind) in ("chair", "stool"):
        d.rounded_rectangle([cx - r * 0.7, cy - r * 0.2, cx + r * 0.7, cy + r * 0.1],
                            radius=r * 0.15, fill=base)
        for sx in (-0.6, 0.6):
            d.line([(cx + sx * r, cy + r * 0.05), (cx + sx * r, cy + r)], fill=_shade(base, 0.7),
                   width=max(3, int(r * 0.16)))
    elif _canon(kind) in ("table", "desk"):
        d.rectangle([cx - r * 1.3, cy - r * 0.18, cx + r * 1.3, cy + r * 0.05],
                    fill=base, outline=_shade(base, 0.6), width=max(1, int(r * 0.06)))
        for sx in (-1.05, 1.05):
            d.line([(cx + sx * r, cy), (cx + sx * r, cy + r * 0.95)],
                   fill=_shade(base, 0.7), width=max(3, int(r * 0.14)))
    elif _canon(kind) in ("tree", "plant", "flower"):
        d.rectangle([cx - r * 0.18, cy + r * 0.1, cx + r * 0.18, cy + r],
                    fill=(110, 75, 45))
        for i, (ox, oy, rr) in enumerate(((0, -0.45, 0.55), (-0.4, -0.1, 0.4), (0.4, -0.1, 0.4))):
            d.ellipse([cx + ox * r - rr * r, cy + oy * r - rr * r,
                       cx + ox * r + rr * r, cy + oy * r + rr * r],
                      fill=_shade(base, 1.0 + 0.1 * i))
    elif _canon(kind) in ("vehicle", "car", "bike", "bus"):
        d.rounded_rectangle([cx - r * 1.4, cy - r * 0.35, cx + r * 1.4, cy + r * 0.35],
                            radius=r * 0.25, fill=base, outline=_shade(base, 0.5),
                            width=max(2, int(r * 0.07)))
        d.polygon([(cx - r * 0.7, cy - r * 0.35), (cx - r * 0.45, cy - r * 0.85),
                   (cx + r * 0.45, cy - r * 0.85), (cx + r * 0.7, cy - r * 0.35)],
                  fill=_shade(base, 1.15))
        for sx in (-0.75, 0.75):
            d.ellipse([cx + sx * r - r * 0.3, cy + r * 0.05,
                       cx + sx * r + r * 0.3, cy + r * 0.65], fill=(40, 40, 45))
    elif _canon(kind) in ("sun", "moon", "lantern", "light"):
        d.ellipse([cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6], fill=base)
        for i in range(8):
            a = math.radians(i * 45)
            d.line([(cx + math.cos(a) * r * 0.75, cy + math.sin(a) * r * 0.75),
                    (cx + math.cos(a) * r * 1.05, cy + math.sin(a) * r * 1.05)],
                   fill=base, width=max(2, int(r * 0.12)))
    elif _canon(kind) in ("door", "window"):
        d.rounded_rectangle([cx - r * 0.7, cy - r, cx + r * 0.7, cy + r], radius=r * 0.1,
                            fill=base, outline=_shade(base, 0.5), width=max(2, int(r * 0.08)))
        d.ellipse([cx + r * 0.35, cy - r * 0.1, cx + r * 0.55, cy + r * 0.1],
                  fill=_shade(base, 1.4))
    else:
        # Unknown kind: an identifiable labelled block rather than nothing.
        d.rounded_rectangle([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6],
                            radius=r * 0.2, fill=base, outline=_shade(base, 0.5),
                            width=max(2, int(r * 0.08)))
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=max(8, int(r * 0.42)))
            label = kind[:9]
            tw = d.textlength(label, font=font)
            d.text(((w - tw) / 2, cy - r * 0.22), label, font=font, fill=(255, 255, 255, 235))
        except Exception:  # noqa: BLE001
            pass

    if ss > 1:
        img = img.resize((S, S), Image.LANCZOS)
    return img


_CANON_ALIASES = {
    "soccer": "ball", "football": "ball", "basketball": "ball", "beach ball": "ball",
    "sphere": "ball", "ball": "ball",
    "package": "box", "parcel": "box", "gift box": "gift", "gift": "gift",
    "suitcase": "box", "crate": "crate", "book": "book", "box": "box",
    "bottle": "bottle", "cup": "cup", "mug": "cup", "glass": "cup",
    "smartphone": "phone", "mobile": "phone", "phone screen": "phone",
    "chair": "chair", "stool": "stool", "seat": "chair",
    "table": "table", "desk": "table",
    "tree": "tree", "plant": "plant", "flower": "flower", "bush": "plant",
    "car": "vehicle", "bike": "vehicle", "bicycle": "vehicle", "bus": "vehicle",
    "vehicle": "vehicle", "truck": "vehicle",
    "sun": "sun", "moon": "moon", "lantern": "lantern", "lamp": "light",
    "light": "light", "torch": "lantern",
    "door": "door", "window": "window",
}

_PROP_COLORS = {
    "ball": (230, 90, 70),
    "box": (200, 160, 110),
    "gift": (190, 80, 120),
    "book": (70, 120, 180),
    "bottle": (90, 160, 200),
    "cup": (235, 235, 235),
    "phone": (40, 40, 46),
    "chair": (140, 95, 60),
    "table": (150, 105, 65),
    "tree": (70, 150, 80),
    "plant": (70, 150, 80),
    "flower": (235, 120, 170),
    "vehicle": (60, 110, 190),
    "sun": (250, 210, 80),
    "moon": (225, 225, 210),
    "lantern": (250, 180, 70),
    "light": (250, 230, 140),
    "door": (150, 100, 70),
    "window": (160, 200, 230),
    "crate": (170, 130, 85),
    "stool": (140, 95, 60),
}


def _canon(kind: str) -> str:
    k = (kind or "").strip().lower()
    return _CANON_ALIASES.get(k, k)


def _hex_color(value: str) -> tuple[int, int, int] | None:
    v = (value or "").strip().lstrip("#")
    if len(v) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in v):
        if len(v) == 3:
            v = "".join(c * 2 for c in v)
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return None


def _resolve_focus(focus: str | None, anim: SceneAnimation, state: dict[str, tuple],
                   ) -> tuple[float, float] | None:
    """Map a CameraCue focus id to a work-canvas point (0..1), or None."""
    if not focus:
        return None
    if focus in state:
        x_frac, _facing, _act, _ph, _mv = state[focus]
        place = anim.characters.get(focus)
        gy = place.ground_frac if place else 0.93
        return max(0.0, min(1.0, x_frac)), max(0.0, min(1.0, gy - 0.34))
    for prop in anim.props:
        if prop.id == focus:
            px, py = _prop_position(prop, anim, 1.0, 1.0, 0.0)
            return max(0.0, min(1.0, px)), max(0.0, min(1.0, py))
    if focus in anim.characters:
        place = anim.characters[focus]
        return max(0.0, min(1.0, place.x_frac)), max(0.0, min(1.0, place.ground_frac - 0.34))
    return None


def _active_camera(anim: SceneAnimation, t: float, state: dict[str, tuple]) -> tuple[str, tuple[float, float] | None, str]:
    """Pick the camera move, focus point and shot kind active at time *t*."""
    move = anim.camera_move
    shot = "wide"
    cue = None
    for c in anim.camera_shots:
        if c.start <= t <= max(c.end, c.start + 0.001):
            cue = c
    if cue is None and anim.camera_shots:
        earlier = [c for c in anim.camera_shots if c.start <= t]
        cue = max(earlier, key=lambda c: c.start) if earlier else None
    if cue is not None:
        move = cue.movement or move
        shot = (cue.kind or "wide").lower()
    focus = _resolve_focus(cue.focus, anim, state) if (cue and cue.focus) else None
    return move, focus, shot


def _prop_position(prop: SceneProp, anim: SceneAnimation, work_w: int, work_h: int,
                   t: float) -> tuple[float, float]:
    """World-space centre of a prop at time *t*, honouring movement cues."""
    active = prop.start <= t <= max(prop.end, prop.start + 0.001)
    x = prop.x
    y = prop.y
    if active and prop.x_to is not None:
        span = max(0.001, prop.end - prop.start)
        ph = _smoothstep((t - prop.start) / span)
        x = _lerp(prop.x, prop.x_to, ph)
        y = _lerp(prop.y, prop.y_to if prop.y_to is not None else prop.y, ph)
    if prop.action == "bounce" and active:
        span = max(0.001, prop.end - prop.start)
        ph = (t - prop.start) / span
        y -= abs(math.sin(ph * math.pi * 3)) * 0.10
    if prop.action == "fly" and active and prop.x_to is not None:
        x = prop.x_to
    return x * work_w, y * work_h


def _compose_frame(anim: SceneAnimation, background: Image.Image, work_w: int, work_h: int,
                   t: float) -> Image.Image:
    frac = t / max(0.001, anim.duration)

    # Resolve per-actor state (cue-driven position/facing/action) first so the
    # camera can focus on the actor who is speaking or moving.
    base_state: dict[str, tuple[float, int, str, float, float]] = {}
    for cid, place in anim.characters.items():
        base_state[cid] = actor_state_at(anim.actions, cid, t, place.x_frac, place.facing,
                                         anim.characters)

    camera_move, focus, shot = _active_camera(anim, t, base_state)
    if focus is None and base_state:
        # Frame the character who is talking (or the first actor) instead of
        # always centring the stage.
        speaker = None
        for sp in anim.dialogue:
            if sp.start <= t <= sp.end:
                speaker = sp.character_id
                break
        if speaker is None:
            speaker = next(iter(base_state))
        focus = _resolve_focus(speaker, anim, base_state)

    crop = camera_at(camera_move, frac, work_w, work_h, anim.out_w, anim.out_h,
                     shot=shot, focus_x=focus[0] if focus else None,
                     focus_y=focus[1] if focus else None)
    frame = background.crop(crop).resize((anim.out_w, anim.out_h), Image.LANCZOS).convert("RGB")
    x0, y0, x1, y1 = crop
    sx = anim.out_w / max(1, (x1 - x0))
    sy = anim.out_h / max(1, (y1 - y0))

    # Props behind characters when they sit further back on the ground line.
    def _draw_props(behind: bool) -> None:
        for prop in anim.props:
            pwx, pwy = _prop_position(prop, anim, work_w, work_h, t)
            is_behind = prop.y < 0.90
            if is_behind != behind:
                continue
            size = int(max(30, prop.scale * work_h * 0.14) * sy)
            sprite = draw_prop(prop, size)
            if sprite is None:
                continue
            # Carrier attachment: ride on the carrier's hands.
            if prop.action == "carried" and prop.carrier and prop.carrier in base_state:
                place = anim.characters.get(prop.carrier)
                if place is not None:
                    pwx = base_state[prop.carrier][0] * work_w + place.facing * work_w * 0.05
                    pwy = place.ground_frac * work_h - place.height_frac * work_h * 0.62
            px = int((pwx - x0) * sx - sprite.width / 2)
            py = int((pwy - y0) * sy - sprite.height / 2)
            frame.paste(sprite, (px, py), sprite)

    _draw_props(behind=True)

    # Draw characters back-to-front (right-most first so overlapping reads well).
    order = sorted(range(len(anim.placements)),
                   key=lambda i: anim.placements[i].x_frac, reverse=True)
    for idx in order:
        p = anim.placements[idx]
        state = base_state.get(p.rig.id)
        if state is None:
            x_frac, facing, action, action_phase, move_amount = (
                p.x_frac, p.facing, "idle", 0.0, 0.0)
        else:
            x_frac, facing, action, action_phase, move_amount = state
        pose = compute_pose(p, anim.dialogue, t, anim.seed, idx,
                            action=action, action_phase=action_phase,
                            move_amount=move_amount, target_x=focus[0] if focus else None)
        pose.facing = facing
        H_screen = max(24, int(p.height_frac * work_h * sy * p.emphasis * p.rig.height_scale))
        char_img = draw_character(p.rig, pose, H_screen)
        wx = x_frac * work_w
        wy = p.ground_frac * work_h
        px = int((wx - x0) * sx - char_img.width / 2)
        # Anchor feet a little above the ground line.
        py = int((wy - y0) * sy - char_img.height + char_img.height * 0.06)
        frame.paste(char_img, (px, py), char_img)

    _draw_props(behind=False)

    if anim.show_table:
        _draw_table_foreground(frame, work_w, work_h, crop, anim.table_top_frac)

    # Title caption (short, top-centre) with a soft shadow.
    if anim.title:
        _draw_title(frame, anim.title)
    return frame


def _draw_title(frame: Image.Image, title: str) -> None:
    from PIL import ImageFont

    try:
        from .fonts import find_font_for_script

        fr = find_font_for_script("latin")
        size = max(16, frame.height // 26)
        font = ImageFont.truetype(str(fr.path), size) if fr.ok and fr.path else ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=max(16, frame.height // 26))
        except Exception:  # noqa: BLE001
            return
    d = ImageDraw.Draw(frame, "RGBA")
    text = title.strip()[:70]
    try:
        tw = d.textlength(text, font=font)
    except Exception:  # noqa: BLE001
        tw = len(text) * (size // 2)
    x = (frame.width - tw) / 2
    y = frame.height * 0.05
    pad = size * 0.35
    d.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + size + pad], radius=size * 0.3,
                        fill=(0, 0, 0, 130))
    d.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 180))
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255))


# ---------------------------------------------------------------------------
# Public rendering API
# ---------------------------------------------------------------------------
def prepare_background(path: Path, work_w: int, work_h: int) -> Image.Image:
    """Load any supported image and normalize it for deterministic compositing.

    Handles RGB, RGBA, grayscale, palette PNGs, transparency and JPEG inputs;
    alpha is flattened onto black *before* resizing so transparent pixels do not
    leak unexpected colors.
    """
    return normalize_image(path, (work_w, work_h), mode="RGB", fit="stretch")


def render_preview_frame(anim: SceneAnimation, out_path: Path, *, t: float | None = None) -> Path:
    """Render a single representative frame (used for editor thumbnails)."""
    work_w = int(anim.out_w * anim.work_scale)
    work_h = int(anim.out_h * anim.work_scale)
    work_w += work_w % 2
    work_h += work_h % 2
    bg = prepare_background(anim.background_path, work_w, work_h)
    tt = anim.duration * 0.45 if t is None else t
    frame = _compose_frame(anim, bg, work_w, work_h, tt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(out_path, "PNG")
    return out_path


ProgressFn = Callable[[int, int], None]
CancelFn = Callable[[], bool]


def render_scene_animation(
    anim: SceneAnimation,
    out_path: Path,
    *,
    subtitle_path: Path | None = None,
    crf: int = 21,
    preset: str = "veryfast",
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
    timeout: int = 1800,
) -> AnimationResult:
    """Render a full animated scene to an H.264 MP4 by piping frames to FFmpeg."""
    ffmpeg = find_ffmpeg("ffmpeg")
    if not ffmpeg:
        return AnimationResult(False, error="ffmpeg not found")
    if not anim.background_path.exists():
        return AnimationResult(False, error=f"background missing: {anim.background_path}")

    work_w = int(anim.out_w * anim.work_scale)
    work_h = int(anim.out_h * anim.work_scale)
    work_w += work_w % 2
    work_h += work_h % 2
    bg = prepare_background(anim.background_path, work_w, work_h)

    total_frames = max(1, int(round(anim.duration * anim.fps)))
    if anim.fps <= 0:
        return AnimationResult(False, error="fps must be positive")

    args = [
        ffmpeg, "-hide_banner", "-y", "-nostdin", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{anim.out_w}x{anim.out_h}", "-r", str(anim.fps), "-i", "-",
    ]
    vf: list[str] = []
    if subtitle_path is not None and subtitle_path.exists():
        if has_filter("subtitles"):
            sub = str(subtitle_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
            vf.append(f"subtitles='{sub}'")
    args += [
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if vf:
        args += ["-vf", ",".join(vf)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.append(str(out_path))

    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    try:
        for i in range(total_frames):
            if cancel is not None and cancel():
                proc.kill()
                return AnimationResult(False, error="cancelled", frames=i)
            t = i / anim.fps
            frame = _compose_frame(anim, bg, work_w, work_h, t)
            try:
                assert proc.stdin is not None
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                err = b""
                if proc.stderr is not None:
                    err = proc.stderr.read()
                return AnimationResult(False, error=f"ffmpeg pipe closed: {exc} {err[:300]!r}", frames=i)
            if progress is not None and (i % 5 == 0 or i == total_frames - 1):
                progress(i + 1, total_frames)
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        rc = proc.wait(timeout=timeout)
        if rc != 0:
            return AnimationResult(False, error=f"ffmpeg exited {rc}: {stderr[:400]!r}",
                                   frames=total_frames)
    except subprocess.TimeoutExpired:
        proc.kill()
        return AnimationResult(False, error="ffmpeg encoding timed out")
    except Exception as exc:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return AnimationResult(False, error=f"{type(exc).__name__}: {exc}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        return AnimationResult(False, error="segment file missing after render", frames=total_frames)
    return AnimationResult(True, out_path, frames=total_frames)
