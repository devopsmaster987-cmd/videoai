"""Viseme-based lip-sync for rigged 2D characters.

Real audio-driven phoneme alignment (forced alignment) requires a speech model
we do not ship.  Instead this module derives a *viseme timeline* from the text
using a documented grapheme/letter-cluster mapping and distributes the visemes
across the measured duration of the synthesized audio.  That is a genuine
viseme animation (not random noise), but it is an approximation, and the UI/report
say so — we never claim frame-accurate phoneme synchronization.

The output drives the mouth-shape drawing in :mod:`freeanimate.animation2d`.
"""

from __future__ import annotations

from dataclasses import dataclass

VISEMES = ("rest", "aa", "ee", "oo", "mm", "fv", "th", "smile", "laugh")

# Mouth openness per viseme (0..1) and width scale (1.0 = neutral).
MOUTH_GEOMETRY: dict[str, tuple[float, float]] = {
    "rest": (0.10, 0.55),
    "aa": (1.00, 0.75),
    "ee": (0.55, 1.15),
    "oo": (0.70, 0.45),
    "mm": (0.22, 0.70),
    "fv": (0.35, 0.85),
    "th": (0.42, 0.80),
    "smile": (0.35, 1.25),
    "laugh": (0.95, 1.05),
}

# Approximate relative duration weight per viseme class.
_DURATION_WEIGHT = {
    "rest": 0.6, "aa": 1.25, "ee": 1.05, "oo": 1.15,
    "mm": 0.75, "fv": 0.7, "th": 0.75, "smile": 0.8, "laugh": 1.0,
}

# Multi-letter clusters first (checked left to right).
_CLUSTERS: list[tuple[str, str]] = [
    ("tion", "th"), ("sion", "th"), ("ough", "oo"), ("augh", "aa"),
    ("th", "th"), ("sh", "ee"), ("ch", "ee"), ("ph", "fv"), ("wh", "oo"),
    ("oo", "oo"), ("ee", "ee"), ("ea", "ee"), ("ai", "ee"), ("ay", "ee"),
    ("ou", "oo"), ("ow", "oo"), ("oa", "oo"), ("ue", "oo"), ("ui", "oo"),
    ("ie", "ee"), ("ei", "ee"), ("au", "aa"), ("aw", "aa"),
]

_LETTER: dict[str, str] = {
    "a": "aa", "e": "ee", "i": "ee", "o": "oo", "u": "oo", "y": "ee",
    "m": "mm", "b": "mm", "p": "mm",
    "f": "fv", "v": "fv",
    "s": "ee", "z": "ee", "c": "ee", "x": "ee", "j": "ee",
    "t": "aa", "d": "aa", "n": "aa", "l": "aa", "r": "aa", "k": "aa",
    "g": "aa", "h": "aa", "w": "oo", "q": "aa",
}


@dataclass
class VisemeEvent:
    viseme: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"viseme": self.viseme, "start": round(self.start, 3), "end": round(self.end, 3)}


def text_to_viseme_sequence(text: str) -> list[tuple[str, float]]:
    """Return ``[(viseme, weight)]`` in reading order, including pauses."""
    seq: list[tuple[str, float]] = []
    low = (text or "").lower()
    i = 0
    while i < len(low):
        ch = low[i]
        if ch in " .,;:!?-—\n\t\r":
            if not seq or seq[-1][0] != "rest":
                seq.append(("rest", 1.2 if ch in ".!?" else 0.45))
            i += 1
            continue
        matched = False
        for cluster, vis in _CLUSTERS:
            if low.startswith(cluster, i):
                seq.append((vis, _DURATION_WEIGHT[vis]))
                i += len(cluster)
                matched = True
                break
        if matched:
            continue
        vis = _LETTER.get(ch)
        if vis:
            seq.append((vis, _DURATION_WEIGHT[vis]))
        i += 1
    if not seq:
        return [("rest", 1.0)]
    return seq


def build_viseme_timeline(text: str, duration_seconds: float, *,
                          lead: float = 0.05, tail: float = 0.08) -> list[VisemeEvent]:
    """Spread the viseme sequence across ``duration_seconds``.

    Vowel-heavy syllables take proportionally longer, so mouth movement tracks
    the rhythm of the line rather than flipping at a fixed rate.
    """
    duration = max(0.2, float(duration_seconds))
    usable = max(0.1, duration - lead - tail)
    seq = text_to_viseme_sequence(text)
    total_w = sum(w for _, w in seq) or 1.0
    events: list[VisemeEvent] = []
    t = lead
    for vis, w in seq:
        seg = usable * (w / total_w)
        # Very short segments are hard to perceive; enforce a small floor but
        # never overflow the spoken window.
        seg = max(0.035, min(seg, lead + usable - t))
        end = min(lead + usable, t + seg)
        if end <= t:
            break
        events.append(VisemeEvent(vis, round(t, 4), round(end, 4)))
        t = end
    if not events:
        events = [VisemeEvent("rest", lead, lead + usable)]
    return events


def viseme_at(timeline: list[VisemeEvent], t: float, default: str = "rest") -> str:
    """Return the viseme active at time ``t`` (binary search keep-simple)."""
    if not timeline:
        return default
    if t <= timeline[0].start:
        return default if timeline[0].start > t else timeline[0].viseme
    # Linear scan is fine: timelines are short (tens to hundreds of entries).
    lo, hi = 0, len(timeline) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        ev = timeline[mid]
        if t < ev.start:
            hi = mid - 1
        elif t >= ev.end:
            lo = mid + 1
        else:
            return ev.viseme
    return default


def mouth_openness(viseme: str) -> float:
    return MOUTH_GEOMETRY.get(viseme, MOUTH_GEOMETRY["rest"])[0]


def mouth_width_scale(viseme: str) -> float:
    return MOUTH_GEOMETRY.get(viseme, MOUTH_GEOMETRY["rest"])[1]


def emotion_mouth(emotion: str) -> str:
    """Map a dialogue emotion to a resting mouth shape for listeners/talkers."""
    e = (emotion or "").lower()
    if e in ("happy", "amused", "joy", "cheerful"):
        return "smile"
    if e in ("laughing", "laugh"):
        return "laugh"
    if e in ("sad", "angry", "surprised", "curious", "neutral"):
        return "rest"
    return "rest"
