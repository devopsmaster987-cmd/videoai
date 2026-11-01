"""Subtitle generation: SRT + WebVTT with validated timing (section 10).

Timing model: each scene's narration window is subdivided into phrase blocks
whose lengths are proportional to their word counts (word-level timing is not
available from the offline engines, so we use phrase-level timing and say so).

Right-to-left languages get an RTL marker in the VTT header and the burned-in
render relies on libass (HarfBuzz + FriBidi) for correct shaping and display
order — never raw ``drawtext``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_LINE_CHARS_DEFAULT = 46
MAX_LINES_DEFAULT = 2


@dataclass
class SubtitleCue:
    index: int
    start: float
    end: float
    text: str

    def is_valid(self) -> list[str]:
        problems: list[str] = []
        if self.start < 0:
            problems.append(f"cue {self.index}: negative start")
        if self.end <= self.start:
            problems.append(f"cue {self.index}: end <= start")
        if not self.text.strip():
            problems.append(f"cue {self.index}: empty text")
        return problems


@dataclass
class SubtitleTrack:
    cues: list[SubtitleCue] = field(default_factory=list)
    language: str = "en"
    rtl: bool = False
    timing_source: str = "phrase_estimate"   # phrase_estimate | audio_measured

    def validate(self) -> list[str]:
        problems: list[str] = []
        prev_end = -1e-6
        for cue in self.cues:
            problems.extend(cue.is_valid())
            if cue.start < prev_end - 1e-3:
                problems.append(f"cue {cue.index}: overlaps previous cue")
            prev_end = cue.end
        return problems


# ---------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def wrap_lines(text: str, max_chars: int = MAX_LINE_CHARS_DEFAULT,
               max_lines: int = MAX_LINES_DEFAULT) -> str:
    """Wrap a cue into at most ``max_lines`` lines of <= ``max_chars`` chars.

    Uses word wrapping; CJK text without spaces is chunked by character count.
    Falls back gracefully when even a single word exceeds ``max_chars``.
    """
    text = " ".join(text.split())
    if not text:
        return ""
    words = _words(text)
    if not words:
        # No spaces (e.g. CJK) — chunk by characters.
        return "\n".join(text[i:i + max_chars] for i in range(0, len(text), max_chars))[: max_chars * max_lines]

    lines: list[str] = []
    current = ""
    for w in words:
        trial = f"{current} {w}".strip()
        if len(trial) <= max_chars or not current:
            current = trial
        else:
            lines.append(current)
            current = w
            if len(lines) == max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    elif current:
        # Overflow: append to last line (better than dropping content silently).
        lines[-1] = (lines[-1] + " " + current)[: max_chars + 20]
    return "\n".join(lines)


def build_cues_for_scene(
    text: str,
    window_start: float,
    window_end: float,
    *,
    start_index: int = 1,
    max_chars: int = MAX_LINE_CHARS_DEFAULT,
    max_lines: int = MAX_LINES_DEFAULT,
    min_cue: float = 0.8,
) -> list[SubtitleCue]:
    """Distribute ``text`` across [window_start, window_end] by word weight.

    Splits into sentence-ish phrases, then packs phrases into cues.
    """
    text = text.strip()
    if not text:
        return []
    phrases = [p.strip() for p in re.split(r"(?<=[.!?。！？…])\s+", text) if p.strip()]
    if not phrases:
        phrases = [text]

    # Pack phrases into cue-sized groups.
    groups: list[str] = []
    current = ""
    for ph in phrases:
        # Very long single phrases get hard-split.
        while len(ph) > int(max_chars * max_lines * 1.2):
            cut = ph[: int(max_chars * max_lines * 1.1)].rsplit(" ", 1)[0]
            if not cut:
                cut = ph[: max_chars]
            if current:
                groups.append(current)
                current = ""
            groups.append(cut.strip())
            ph = ph[len(cut):].strip()
        if not ph:
            continue
        trial = f"{current} {ph}".strip()
        if len(trial) <= max_chars * max_lines or not current:
            current = trial
        else:
            groups.append(current)
            current = ph
    if current:
        groups.append(current)

    weights = [max(1, len(_words(g))) for g in groups]
    total_w = sum(weights)
    span = max(0.1, window_end - window_start)
    cues: list[SubtitleCue] = []
    t = window_start
    idx = start_index
    for g, w in zip(groups, weights):
        dur = max(min_cue, span * (w / total_w))
        end = min(window_end, t + dur)
        if end <= t:
            end = t + min_cue
        # Don't overshoot the last cue past the window too much.
        cues.append(SubtitleCue(idx, round(t, 3), round(end, 3), wrap_lines(g, max_chars, max_lines)))
        idx += 1
        t = end
    # Clamp final end to window.
    if cues:
        cues[-1].end = round(max(cues[-1].start + 0.5, window_end), 3)
    return cues


def build_track(
    scenes: list[dict],
    language: str = "en",
    rtl: bool = False,
    *,
    max_chars: int = MAX_LINE_CHARS_DEFAULT,
    max_lines: int = MAX_LINES_DEFAULT,
    narration_durations: dict[str, float] | None = None,
) -> SubtitleTrack:
    """Build a full subtitle track from scene dictionaries.

    ``scenes`` items need: scene_id, duration_seconds, subtitle_text (or
    narration_text).  Optional ``narration_durations`` tightens timing to real
    audio lengths (cue spans the narration, not the padded scene).
    """
    track = SubtitleTrack(language=language, rtl=rtl)
    t = 0.0
    idx = 1
    for sc in scenes:
        sid = sc["scene_id"]
        dur = float(sc.get("duration_seconds", 0))
        text = (sc.get("subtitle_text") or sc.get("narration_text") or "").strip()
        nd = (narration_durations or {}).get(sid)
        if text:
            # Cue window: narration occupies the scene minus small padding.
            if nd and nd > 0:
                window_end = t + min(dur, nd + 0.1)
                track.timing_source = "audio_measured"
            else:
                window_end = t + dur
            cues = build_cues_for_scene(text, t + 0.05, max(t + 0.5, window_end - 0.15),
                                        start_index=idx, max_chars=max_chars, max_lines=max_lines)
            track.cues.extend(cues)
            idx += len(cues)
        t += dur
    return track


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def _fmt_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def to_srt(track: SubtitleTrack) -> str:
    parts: list[str] = []
    for c in track.cues:
        parts.append(str(c.index))
        parts.append(f"{_fmt_srt(c.start)} --> {_fmt_srt(c.end)}")
        parts.append(c.text)
        parts.append("")
    return "\n".join(parts)


def to_vtt(track: SubtitleTrack) -> str:
    header = "WEBVTT\n"
    if track.language:
        header += f"Language: {track.language}\n"
    if track.rtl:
        header += "NOTE RTL language: rendering engines should apply right-to-left direction.\n"
    header += "\n"
    parts: list[str] = [header]
    for c in track.cues:
        parts.append(str(c.index))
        parts.append(f"{_fmt_vtt(c.start)} --> {_fmt_vtt(c.end)}")
        if track.rtl:
            parts.append("\u202b" + c.text)  # RLE mark as a hint for plain players
        else:
            parts.append(c.text)
        parts.append("")
    return "\n".join(parts)


def write_subtitles(track: SubtitleTrack, out_dir: Path, stem: str = "subtitles") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / f"{stem}.srt"
    vtt_path = out_dir / f"{stem}.vtt"
    srt_path.write_text(to_srt(track), encoding="utf-8")
    vtt_path.write_text(to_vtt(track), encoding="utf-8")
    return {"srt": srt_path, "vtt": vtt_path}


def parse_srt(text: str) -> list[SubtitleCue]:
    """Parse SRT content back into cues (used by tests and validation)."""
    cues: list[SubtitleCue] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        idx = 0
        if re.fullmatch(r"\d+", lines[0].strip()):
            idx = int(lines[0].strip())
            time_line = lines[1]
            text_lines = lines[2:]
        else:
            time_line = lines[0]
            text_lines = lines[1:]
            idx = len(cues) + 1
        m = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", time_line)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append(SubtitleCue(idx, start, end, "\n".join(text_lines)))
    return cues


# ---------------------------------------------------------------------------
# Per-scene cues + ASS (for correct burn-in and RTL font control)
# ---------------------------------------------------------------------------
def build_scene_cues(
    scene: dict,
    narration_durations: dict[str, float] | None = None,
    *,
    max_chars: int = MAX_LINE_CHARS_DEFAULT,
    max_lines: int = MAX_LINES_DEFAULT,
) -> list[SubtitleCue]:
    """Cues for a *single* scene with timestamps local to that scene (start=0).

    These are used when burning subtitles into individual scene segments: the
    global SRT timestamps cannot be applied per segment because each segment's
    clock restarts at zero.
    """
    sid = scene.get("scene_id", "")
    dur = float(scene.get("duration_seconds", 0) or 0)
    nd = (narration_durations or {}).get(sid)
    text = (scene.get("subtitle_text") or scene.get("narration_text") or "").strip()
    if not text or dur <= 0:
        return []
    end = (min(dur - 0.15, nd + 0.1) if nd else dur - 0.15)
    end = max(0.6, end)
    return build_cues_for_scene(text, 0.05, end, start_index=1,
                                max_chars=max_chars, max_lines=max_lines)


def _fmt_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def to_ass(
    cues: list[SubtitleCue],
    *,
    font_family: str = "Arial",
    font_size: int = 42,
    margin_v: int = 112,
    outline: int = 2,
    shadow: int = 1,
    rtl: bool = False,
    play_w: int = 1920,
    play_h: int = 1080,
    title: str | None = None,
    title_seconds: float = 0.0,
    title_font_family: str | None = None,
) -> str:
    """Serialize cues as an ASS script so libass applies real styling.

    ASS carries an explicit font family, size, outline, and margins — needed
    for reliable multilingual rendering (the SRT format cannot express these).
    libass performs HarfBuzz shaping and FriBidi reordering for RTL text.

    When ``title`` is given, a top-centered title line is added for the whole
    scene (used for scripts Pillow cannot shape, e.g. Arabic/Devanagari).
    """
    style_lines = [
        f"Style: Default,{font_family},{font_size},&H00FFFFFF,&H000000FF,"
        f"&H00101010,&H64000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,"
        f"60,60,{margin_v},1"
    ]
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {play_w}\n"
        f"PlayResY: {play_h}\n"
        "YCbCr Matrix: None\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
    )
    lines: list[str] = []
    if title and title.strip() and title_seconds > 0:
        tsize = max(20, int(font_size * 1.35))
        # BorderStyle=3 draws a soft opaque box behind the title text.
        style_lines.append(
            f"Style: Title,{title_font_family or font_family},{tsize},&H00FFFFFF,&H000000FF,"
            f"&H00202020,&H88000000,0,0,0,0,100,100,0,0,3,10,0,8,60,60,56,1"
        )
        ttext = title.strip().replace("\n", r"\N")
        if rtl:
            ttext = "\u202b" + ttext
        lines.append(
            f"Dialogue: 1,0:00:00.00,{_fmt_ass_time(title_seconds)},"
            f"Title,,0,0,0,,{ttext}"
        )
    header += "\n".join(style_lines) + (
        "\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    for c in cues:
        text = c.text.replace("\n", r"\N")
        if rtl:
            # Let libass handle bidi; only mark the paragraph direction.
            text = "\u202b" + text
        lines.append(
            f"Dialogue: 0,{_fmt_ass_time(c.start)},{_fmt_ass_time(c.end)},"
            f"Default,,0,0,0,,{text}"
        )
    return header + "\n".join(lines) + ("\n" if lines else "")
