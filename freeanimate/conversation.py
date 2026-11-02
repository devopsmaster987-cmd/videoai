"""Multi-character conversation pipeline.

Assigns each dialogue line to its speaker, gives speakers distinct local voices
where the installed engines allow it, synthesizes each line, measures real audio
lengths, lays the lines onto a turn-taking timeline, and produces both the voice
track and the per-line animation spans (with visemes) used by the 2D renderer.

Honesty rules:

* If only one voice is installed, the limitation is reported; we vary speaking
  rate slightly to differentiate characters but never claim distinct voices.
* Nothing is sent to a network service.
* Text with no assigned speaker is reported, not silently mis-assigned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .audio_mix import concat_wavs, pad_wav_to, wav_duration_seconds
from .characters import Character
from .ffmpeg_utils import FFmpegError, run_ffmpeg
from .animation2d import DialogueSpan
from .lipsync import build_viseme_timeline
from .prompt_understanding import CharacterSpec, DialogueLine
from .subtitles import SubtitleCue, SubtitleTrack, wrap_lines
from .tts import (TTSResult, available_voices, engine_by_name, pick_engine,
                  synthesize_with_fallback, wav_duration, write_silence_wav)

DEFAULT_RATE = 44100


@dataclass
class VoiceAssignment:
    character_id: str
    character_name: str
    voice_id: str | None
    engine: str
    speed: float = 1.0
    distinct: bool = True

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RenderedLine:
    speaker_id: str
    speaker_name: str
    text: str
    emotion: str
    action: str | None
    start: float
    end: float
    wav: Path | None
    engine: str
    voice_id: str | None
    source: str = "prompt"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "wav"}
        d["wav"] = str(self.wav) if self.wav else None
        return d


@dataclass
class ConversationResult:
    ok: bool
    lines: list[RenderedLine] = field(default_factory=list)
    spans: list[DialogueSpan] = field(default_factory=list)
    voice_track: Path | None = None
    total_seconds: float = 0.0
    assignments: list[VoiceAssignment] = field(default_factory=list)
    subtitles: SubtitleTrack | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
def _normalize_wav(src: Path, dst: Path, rate: int = DEFAULT_RATE) -> Path:
    """Convert any TTS output to mono 16-bit PCM at a common sample rate."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-i", str(src), "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", "-y", str(dst),
    ], timeout=300)
    return dst


def norm_lang(code: str | None) -> str:
    return (code or "en").replace("_", "-").split("-")[0].lower()


def assign_voices(
    characters: list[Character | CharacterSpec],
    *,
    language: str = "en",
    config: dict[str, Any] | None = None,
) -> tuple[list[VoiceAssignment], list[str]]:
    """Assign the best available local voices; report honest limitations."""
    cfg = dict(config or {})
    notes: list[str] = []
    voices = available_voices(language, config=cfg)
    assignments: list[VoiceAssignment] = []
    distinct_ids = {v.id for v in voices}
    if not voices:
        for c in characters:
            assignments.append(VoiceAssignment(
                c.id, c.name, None, "silence", 1.0, distinct=False))
        notes.append(
            f"No local voice for language '{language}' was found. Narration will be silent "
            "and correctly timed. Install Piper/espeak-ng or a Windows voice for speech."
        )
        return assignments, notes

    if len(distinct_ids) < len(characters):
        notes.append(
            f"Only {len(distinct_ids)} distinct voice(s) are installed for '{language}', but "
            f"{len(characters)} characters need voices. Voices are reused with small rate changes; "
            "install more voices for clearer separation."
        )

    for i, c in enumerate(characters):
        v = voices[i % len(voices)]
        reuse_round = i // len(voices)
        # Vary rate only when a voice must be reused.
        speed = 1.0 + (0.06 if reuse_round % 2 == 1 else -0.04 if reuse_round >= 2 else 0.0)
        speed = max(0.8, min(1.2, speed))
        assignments.append(VoiceAssignment(
            character_id=c.id,
            character_name=c.name,
            voice_id=v.id,
            engine=v.engine,
            speed=speed,
            distinct=len(distinct_ids) >= len(characters),
        ))
    if all(a.engine == "sapi" for a in assignments) and len(characters) > 1:
        notes.append("Using Windows SAPI voices. Additional SAPI voices can be added in "
                     "Settings > Time & Language > Speech.")
    return assignments, notes


def _speaking_gap(prev: RenderedLine | None, line: DialogueLine) -> float:
    """Natural turn-taking gap: longer when the speaker changes or emotion shifts."""
    base = 0.45
    if prev is None:
        return 0.3
    if prev.speaker_id != line.speaker:
        base = 0.55
    emo = (line.emotion or "").lower()
    if emo in ("surprised", "sad"):
        base += 0.2
    if emo in ("laughing", "laugh"):
        base += 0.15
    return base


def _synth_line(text: str, out_wav: Path, assignment: VoiceAssignment,
                language: str, cfg: dict[str, Any]) -> TTSResult:
    """Synthesize one line with the assigned engine/voice, then fall back."""
    if assignment.engine and assignment.engine != "silence":
        try:
            eng = engine_by_name(assignment.engine, {**cfg, "wpm": cfg.get("wpm", 130)})
            if eng.probe().get("available"):
                res = eng.synthesize(text, out_wav, voice_id=assignment.voice_id,
                                     speed=assignment.speed)
                if res.ok:
                    return res
        except Exception:  # noqa: BLE001 — fall through to the robust helper
            pass
    return synthesize_with_fallback(
        text, out_wav, preferred="auto", language=language,
        voice_id=assignment.voice_id, speed=assignment.speed, config=cfg,
    )


def render_conversation(
    lines: list[DialogueLine],
    characters: list[Character | CharacterSpec],
    out_dir: Path,
    *,
    language: str = "en",
    config: dict[str, Any] | None = None,
    lead_in: float = 0.3,
    tail_out: float = 0.7,
    max_chars: int = 46,
    max_lines: int = 2,
) -> ConversationResult:
    """Synthesize and time a full conversation."""
    cfg = dict(config or {})
    out_dir = Path(out_dir)
    wav_dir = out_dir / "lines"
    wav_dir.mkdir(parents=True, exist_ok=True)
    by_id = {c.id: c for c in characters}
    by_name = {c.name.lower(): c for c in characters}
    assignments, notes = assign_voices(characters, language=language, config=cfg)
    assign_by_id = {a.character_id: a for a in assignments}

    if not lines:
        return ConversationResult(False, error="no dialogue lines to render",
                                  assignments=assignments, notes=notes)

    rendered: list[RenderedLine] = []
    cursor = lead_in
    prev: RenderedLine | None = None
    sub_cues: list[SubtitleCue] = []
    cue_index = 1
    errors: list[str] = []

    for i, line in enumerate(lines, start=1):
        speaker = by_id.get(line.speaker) or by_name.get((line.speaker or "").lower())
        if speaker is None:
            # Try matching by name in the line's speaker string.
            for c in characters:
                if c.name.lower() in (line.speaker or "").lower():
                    speaker = c
                    break
        if speaker is None:
            errors.append(f"line {i}: unknown speaker {line.speaker!r}; skipped")
            continue
        assignment = assign_by_id.get(speaker.id) or VoiceAssignment(
            speaker.id, speaker.name, None, "auto", 1.0)
        raw_wav = wav_dir / f"line_{i:03d}_raw.wav"
        res = _synth_line(line.text, raw_wav, assignment, language, cfg)
        if not res.ok or res.audio_path is None:
            errors.append(f"line {i}: synthesis failed: {res.error}")
            continue
        normalized = wav_dir / f"line_{i:03d}.wav"
        try:
            _normalize_wav(res.audio_path, normalized)
        except FFmpegError as exc:
            errors.append(f"line {i}: audio normalization failed: {exc}")
            continue
        ndur = wav_duration(normalized)
        gap = _speaking_gap(prev, line)
        start = round(cursor, 3)
        end = round(start + ndur, 3)
        rl = RenderedLine(
            speaker_id=speaker.id,
            speaker_name=speaker.name,
            text=line.text,
            emotion=line.emotion or "neutral",
            action=line.action,
            start=start,
            end=end,
            wav=normalized,
            engine=res.engine,
            voice_id=res.voice_id or assignment.voice_id,
            source=line.source,
        )
        rendered.append(rl)
        sub_cues.append(SubtitleCue(
            cue_index, start, min(end, round(start + ndur + 0.4, 3)),
            wrap_lines(f"{speaker.name}: {line.text}" if len(characters) > 1 else line.text,
                       max_chars=max_chars, max_lines=max_lines),
        ))
        cue_index += 1
        cursor = end + gap
        prev = rl

    if not rendered:
        return ConversationResult(False, assignments=assignments, notes=notes,
                                  error="; ".join(errors) or "no lines could be synthesized")

    total = round(cursor + tail_out, 3)
    # Build the aligned voice track: lead silence + each line padded to its slot.
    segments: list[Path] = []
    lead = write_silence_wav(wav_dir / "_lead.wav", lead_in, rate=DEFAULT_RATE)
    segments.append(lead)
    for i, rl in enumerate(rendered):
        if rl.wav is None:
            continue
        next_start = rendered[i + 1].start if i + 1 < len(rendered) else total - lead_in
        slot = max(rl.duration + 0.05, next_start - rl.start)
        padded = wav_dir / f"line_{i + 1:03d}_pad.wav"
        pad_wav_to(rl.wav, padded, slot)
        segments.append(padded)
    tail = write_silence_wav(wav_dir / "_tail.wav", max(0.1, total - lead_in - sum(
        wav_duration_seconds(p) for p in segments[1:])), rate=DEFAULT_RATE)
    segments.append(tail)
    voice_track = out_dir / "conversation_voice.wav"
    try:
        concat_wavs(segments, voice_track, rate=DEFAULT_RATE)
    except FFmpegError as exc:
        return ConversationResult(False, assignments=assignments, notes=notes,
                                  error=f"voice track assembly failed: {exc}")

    spans = [
        DialogueSpan(
            character_id=rl.speaker_id,
            text=rl.text,
            start=rl.start,
            end=rl.end,
            emotion=rl.emotion,
            action=rl.action,
            visemes=build_viseme_timeline(rl.text, rl.duration),
        )
        for rl in rendered
    ]
    track = SubtitleTrack(cues=sub_cues, language=language,
                          timing_source="audio_measured")
    problems = track.validate()
    if problems:
        notes.append("subtitle timing warnings: " + "; ".join(problems[:3]))
    if errors:
        notes.extend(errors)

    actual_total = wav_duration_seconds(voice_track)
    return ConversationResult(
        ok=True,
        lines=rendered,
        spans=spans,
        voice_track=voice_track,
        total_seconds=actual_total,
        assignments=assignments,
        subtitles=track,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Dialogue parsing from free text (editor convenience)
# ---------------------------------------------------------------------------
_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _\-]{0,30}?)\s*:\s*(.+)$")


def parse_dialogue_text(text: str, characters: list[Character | CharacterSpec]) -> list[DialogueLine]:
    """Parse ``Name: line`` or ``Name: [emotion] line`` text into lines.

    Names are matched to characters case-insensitively (or by "Character N").
    Unknown names are reported by raising a ValueError with the offending line.
    """
    by_name = {c.name.lower(): c.id for c in characters}
    for i, c in enumerate(characters, start=1):
        by_name.setdefault(f"character {i}", c.id)
        by_name.setdefault(f"char {i}", c.id)
    out: list[DialogueLine] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = _LINE_RE.match(raw)
        if not m:
            if out:
                out[-1].text += " " + raw
            continue
        who, rest = m.group(1).strip(), m.group(2).strip()
        cid = by_name.get(who.lower())
        if cid is None:
            raise ValueError(f"unknown speaker {who!r} (known: {', '.join(by_name)})")
        emotion = "neutral"
        em = re.match(r"\[([a-zA-Z]+)\]\s*(.+)", rest)
        if em:
            emotion = em.group(1).lower()
            rest = em.group(2)
        out.append(DialogueLine(speaker=cid, text=rest, emotion=emotion, source="prompt"))
    return out
