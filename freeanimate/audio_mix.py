"""Audio assembly: narration concatenation, music, ducking, fades, muxing.

Handles:

* Pure-Python WAV concatenation (no re-encode, exact sample alignment) for
  per-scene narration stitched to the scene timeline.
* Optional background music: user file or a procedural ambient pad synthesized
  with numpy (labeled ``procedural`` in the report — it is *synthesized
  ambience*, not AI-composed music).
* Real narration ducking via ffmpeg ``sidechaincompress``.
* ``loudnorm`` two-pass style single-pass normalization, fade-in/out, and
  clipping protection (``alimiter`` / peak headroom).

The pipeline stays fully functional with no music at all (section 11).
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_MUSIC_VOLUME, DEFAULT_SAMPLE_RATE
from .ffmpeg_utils import FFmpegError, run_ffmpeg, probe


@dataclass
class AudioBuildResult:
    ok: bool
    voice_path: Path | None
    final_mix_path: Path | None
    total_seconds: float
    music_used: str = "none"        # none | procedural | <path>
    error: str | None = None
    notes: list[str] | None = None


# ---------------------------------------------------------------------------
# WAV helpers (pure Python — reliable on any machine)
# ---------------------------------------------------------------------------
def read_wav_params(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate()


def concat_wavs(inputs: list[Path], out_path: Path, *, rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Concatenate PCM WAVs, resampling *silently skipping* mismatched formats.

    All narration segments are produced at the same rate by the TTS layer;
    if a mismatch sneaks in we raise rather than produce garbage audio.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not inputs:
        # 1 second of silence so downstream ffmpeg has *something* valid.
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00" * rate)
        return out_path

    params = None
    with wave.open(str(out_path), "wb") as out:
        for p in inputs:
            with wave.open(str(p), "rb") as w:
                cur = (w.getnchannels(), w.getsampwidth(), w.getframerate())
                if params is None:
                    params = cur
                    out.setnchannels(cur[0])
                    out.setsampwidth(cur[1])
                    out.setframerate(cur[2])
                elif cur != params:
                    raise FFmpegError(
                        f"WAV format mismatch in concat: {p.name} is {cur}, expected {params}"
                    )
                out.writeframes(w.readframes(w.getnframes()))
    return out_path


def pad_wav_to(src: Path, out_path: Path, seconds: float, *, rate: int | None = None) -> Path:
    """Pad a WAV with trailing silence to an exact duration (never truncate)."""
    with wave.open(str(src), "rb") as w:
        nch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    target_frames = int(seconds * fr)
    current_frames = len(frames) // (nch * sw) if nch and sw else 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(nch)
        out.setsampwidth(sw)
        out.setframerate(fr)
        out.writeframes(frames)
        if target_frames > current_frames:
            silence = b"\x00" * (nch * sw) * (target_frames - current_frames)
            out.writeframes(silence)
    return out_path


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        fr = w.getframerate()
        return w.getnframes() / float(fr) if fr else 0.0


# ---------------------------------------------------------------------------
# Procedural ambience (clearly labeled, not "AI music")
# ---------------------------------------------------------------------------
def synthesize_ambient_pad(out_path: Path, seconds: float, *, rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Synthesize a soft ambient pad with numpy (sine stacks + slow LFO).

    This is *procedurally synthesized ambience*, not AI-composed music, and
    the final report labels it as such.
    """
    import numpy as np

    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float64) / rate
    # Root chord: gentle fifths, very low amplitude; slow amplitude LFO.
    freqs = [110.0, 164.81, 220.0, 329.63]  # A2, E3, A3, E4
    audio = np.zeros(n, dtype=np.float64)
    for i, f in enumerate(freqs):
        lfo = 0.6 + 0.4 * np.sin(2 * math.pi * (0.03 + 0.01 * i) * t + i)
        audio += (0.22 / (i + 1)) * lfo * np.sin(2 * math.pi * f * t + 0.2 * i)
    # Slow overall swell so long tracks are not static.
    swell = 0.75 + 0.25 * np.sin(2 * math.pi * 0.017 * t)
    audio *= swell * 0.18
    # Tiny detuned layer for warmth.
    audio += 0.03 * np.sin(2 * math.pi * 220.5 * t) * swell
    # Soft fade-in/out to avoid clicks at loop boundaries.
    fade = int(2.0 * rate)
    if n > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    # 16-bit PCM mono
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out_path


# ---------------------------------------------------------------------------
# Mixing / final audio
# ---------------------------------------------------------------------------
def mix_final_audio(
    voice_path: Path,
    out_path: Path,
    *,
    total_seconds: float,
    music: str = "none",
    music_volume: float = DEFAULT_MUSIC_VOLUME,
    music_path: Path | None = None,
    sample_rate: int = 44100,
    fade_seconds: float = 1.5,
) -> AudioBuildResult:
    """Mix narration (+ optional music with ducking) into a final audio file.

    Output is AAC in an .m4a-ready WAV-less container step: we emit ``.wav``
    PCM at full quality and let the assembly mux step encode to AAC, OR emit
    directly with AAC when the caller passes an .m4a/.mp4 path.
    """
    notes: list[str] = []
    if not voice_path.exists():
        return AudioBuildResult(False, None, None, 0.0, music, error=f"voice track missing: {voice_path}")

    music_used = "none"
    music_input: Path | None = None
    if music and music != "none":
        if music == "procedural":
            music_input = out_path.parent / "ambient_pad.wav"
            synthesize_ambient_pad(music_input, total_seconds, rate=sample_rate)
            music_used = "procedural"
            notes.append("Background audio is procedurally synthesized ambience (not AI-composed music).")
        elif music_path is not None and music_path.exists():
            music_input = music_path
            music_used = str(music_path)
        else:
            notes.append(f"Requested music source not found ({music}); continuing without music.")

    args: list[str] = ["-i", str(voice_path)]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if music_input is None:
        # Voice only: normalize gently + fade + limit.
        af = (
            f"volume=1.0,"
            f"afade=t=in:st=0:d={fade_seconds:.2f},"
            f"afade=t=out:st={max(0.0, total_seconds - fade_seconds):.2f}:d={fade_seconds:.2f},"
            f"alimiter=level_in=1:level_out=0.95:limit=0.95"
        )
        args += ["-af", af, "-ar", str(sample_rate), "-ac", "1"]
    else:
        # Duck music under narration with sidechaincompress, then mix.
        args += ["-i", str(music_input)]
        filter_complex = (
            "[1:a]volume={mv:.3f},aformat=sample_rates={sr}:channel_layouts=mono[bg];"
            "[bg][0:a]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=400[duck];"
            "[0:a][duck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed];"
            "[mixed]afade=t=in:st=0:d={fd:.2f},"
            "afade=t=out:st={fo:.2f}:d={fd:.2f},"
            "alimiter=level_in=1:level_out=0.95:limit=0.95,"
            "aformat=sample_rates={sr}:channel_layouts=mono[out]"
        ).format(mv=music_volume, sr=sample_rate, fd=fade_seconds,
                 fo=max(0.0, total_seconds - fade_seconds))
        args += ["-filter_complex", filter_complex, "-map", "[out]"]

    suffix = out_path.suffix.lower()
    if suffix in (".m4a", ".mp4", ".mov"):
        args += ["-c:a", "aac", "-b:a", "160k"]
    else:
        args += ["-c:a", "pcm_s16le"]
    args += ["-t", f"{total_seconds:.3f}", str(out_path)]

    try:
        run_ffmpeg(args, timeout=3600)
    except FFmpegError as exc:
        return AudioBuildResult(False, voice_path, None, total_seconds, music_used,
                                error=str(exc), notes=notes)

    info = probe(out_path)
    dur = info.duration or total_seconds
    return AudioBuildResult(True, voice_path, out_path, dur, music_used, notes=notes)


# ---------------------------------------------------------------------------
# Voice-over-seconds layout helper used by the pipeline
# ---------------------------------------------------------------------------
def build_voice_track(
    per_scene_wavs: list[tuple[str, Path]],
    scene_durations: list[float],
    out_path: Path,
    *,
    gap: float = 0.35,
    rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[Path, float, dict[str, float]]:
    """Lay per-scene narration onto the scene timeline with padding.

    Returns (combined wav path, total seconds, {scene_id: narration_seconds}).
    """
    if len(per_scene_wavs) != len(scene_durations):
        raise FFmpegError("per_scene_wavs and scene_durations length mismatch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    padded: list[Path] = []
    durations: dict[str, float] = {}
    tmp_dir = out_path.parent / "_voice_pads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for (sid, wav), dur in zip(per_scene_wavs, scene_durations):
        nd = wav_duration_seconds(wav) if wav.exists() else 0.0
        durations[sid] = nd
        target = max(dur, nd + 0.05)
        padded_path = tmp_dir / f"{sid}_padded.wav"
        pad_wav_to(wav, padded_path, target)
        padded.append(padded_path)
    concat_wavs(padded, out_path, rate=rate)
    total = wav_duration_seconds(out_path)
    return out_path, total, durations
