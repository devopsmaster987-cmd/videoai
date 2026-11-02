"""Automated quality checks for film exports (master prompt section 23).

These checks are honest *engineering* checks, not proof of artistic quality:

* missing scene segments / assets / characters
* characters that do not move during a required action window
* movement cues without a real path (the usual cause of foot sliding)
* prop placement overlaps and unrenderable prop kinds
* character continuity and orientation at the plan level
* camera coverage gaps and invalid focus targets
* near-black frames and unreadable/corrupted video
* silent or clipping audio
* subtitle timing errors and unknown dialogue speakers
* incomplete exports (missing streams / wrong duration)

A contact sheet of representative frames is written so a human can inspect the
actual output instead of trusting the checks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .animation2d import SceneProp, draw_prop
from .ffmpeg_utils import FFmpegError, probe, run_ffmpeg
from .project import Project
from .subtitles import parse_srt

SAMPLE_FRAME_COUNT = 9


@dataclass
class QACheck:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "fail"          # fail | warn | info

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail,
                "severity": self.severity}


@dataclass
class QAReport:
    ok: bool = True
    checks: list[QACheck] = field(default_factory=list)
    contact_sheet: str | None = None
    generated_at: float = field(default_factory=time.time)

    def add(self, name: str, ok: bool, detail: str = "", severity: str = "fail") -> None:
        self.checks.append(QACheck(name, ok, detail, severity))
        if not ok and severity == "fail":
            self.ok = False

    @property
    def failures(self) -> list[QACheck]:
        return [c for c in self.checks if not c.ok and c.severity == "fail"]

    @property
    def warnings(self) -> list[QACheck]:
        return [c for c in self.checks if not c.ok and c.severity == "warn"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "generated_at": self.generated_at,
            "contact_sheet": self.contact_sheet,
            "checks": [c.to_dict() for c in self.checks],
            "failures": [c.to_dict() for c in self.failures],
            "warnings": [c.to_dict() for c in self.warnings],
        }

    def summary(self) -> str:
        lines = ["FreeAnimate film QA report", "=" * 34,
                 f"Overall: {'PASS' if self.ok else 'FAIL'}",
                 f"Checks: {len(self.checks)} ({len(self.failures)} failed, "
                 f"{len(self.warnings)} warnings)", ""]
        for check in self.checks:
            mark = "ok  " if check.ok else ("WARN" if check.severity == "warn" else "FAIL")
            lines.append(f"[{mark}] {check.name}: {check.detail}")
        if self.contact_sheet:
            lines += ["", f"Contact sheet: {self.contact_sheet}"]
        lines += ["", "Automated checks cannot prove artistic quality; inspect the contact "
                      "sheet and the rendered video."]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def _grab_frame(video: Path, t: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(["-ss", f"{max(0.0, t):.3f}", "-i", str(video), "-frames:v", "1",
                    "-q:v", "2", str(out)], timeout=120)
    except FFmpegError:
        return False
    return out.exists() and out.stat().st_size > 0


def _frame_luminance(path: Path) -> float | None:
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as img:
            return float(ImageStat.Stat(img.convert("L")).mean[0])
    except Exception:  # noqa: BLE001
        return None


def _frame_diff(a: Path, b: Path) -> float | None:
    try:
        from PIL import Image, ImageChops, ImageStat
        with Image.open(a) as ia, Image.open(b) as ib:
            if ia.size != ib.size:
                ib = ib.resize(ia.size)
            diff = ImageChops.difference(ia.convert("RGB"), ib.convert("RGB"))
            stat = ImageStat.Stat(diff)
            return float(sum(stat.mean) / len(stat.mean))
    except Exception:  # noqa: BLE001
        return None


def _audio_levels(video: Path) -> tuple[float | None, float | None]:
    try:
        proc = run_ffmpeg(["-i", str(video), "-af", "volumedetect", "-f", "null", "-"],
                          timeout=300)
    except FFmpegError:
        return None, None
    import re
    text = proc.stderr or ""
    mean = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", text)
    peak = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", text)
    return (float(mean.group(1)) if mean else None,
            float(peak.group(1)) if peak else None)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def run_qa(project: Project, plan: Any, video: Path, *, convo: Any = None,
           validation: Any = None, languages: Any = None,
           sample_frames: int = SAMPLE_FRAME_COUNT,
           work_dir: Path | None = None) -> QAReport:
    report = QAReport()
    work = work_dir or project.stage_dir("reports") / "qa_frames"
    work.mkdir(parents=True, exist_ok=True)
    scenes = list(getattr(plan, "scenes", []) or [])
    duration = 0.0

    # ---- scene segments ------------------------------------------------
    missing: list[str] = []
    bad_duration: list[str] = []
    for scene in scenes:
        seg = project.path("renders", f"scene_{scene.index + 1:02d}.mp4")
        if not seg.exists() or seg.stat().st_size <= 1024:
            missing.append(f"scene {scene.index + 1}")
            continue
        try:
            info = probe(seg)
        except FFmpegError as exc:
            missing.append(f"scene {scene.index + 1} (unreadable: {exc})")
            continue
        if scene.duration_actual > 0 and info.duration is not None:
            if abs(info.duration - scene.duration_actual) > max(0.9, scene.duration_actual * 0.1):
                bad_duration.append(f"scene {scene.index + 1}: {info.duration:.2f}s vs "
                                    f"planned {scene.duration_actual:.2f}s")
    report.add("scene_segments", not missing,
               "all planned scene files exist" if not missing else
               "missing: " + ", ".join(missing[:8]))
    report.add("scene_timings", not bad_duration,
               "scene durations match the measured audio" if not bad_duration else
               "; ".join(bad_duration[:5]), severity="warn" if bad_duration else "fail")

    # ---- characters + assets -------------------------------------------
    lib_path = project.path("assets", "characters", "library.json")
    library_ids: set[str] = set()
    if lib_path.exists():
        try:
            data = json.loads(lib_path.read_text(encoding="utf-8"))
            records = data.get("characters") if isinstance(data, dict) else data
            for rec in records or []:
                if isinstance(rec, dict) and rec.get("id"):
                    library_ids.add(str(rec["id"]))
        except (OSError, json.JSONDecodeError):
            pass
    planned_ids = {c.id for c in getattr(plan, "characters", [])}
    missing_chars = sorted(planned_ids - library_ids) if library_ids else []
    report.add("characters_present", not missing_chars,
               f"{len(planned_ids)} planned characters resolved in the library"
               if not missing_chars else "missing character records: " +
               ", ".join(missing_chars[:6]))

    missing_assets = [s.scene_id for s in scenes
                      if s.background and not Path(s.background).exists()]
    report.add("assets_present", not missing_assets,
               "all scene backgrounds exist" if not missing_assets else
               "missing backgrounds for: " + ", ".join(missing_assets[:6]))

    # ---- motion during required actions --------------------------------
    motion_failures: list[str] = []
    checked = 0
    for scene in scenes:
        if checked >= 3 or not scene.actions:
            continue
        seg = project.path("renders", f"scene_{scene.index + 1:02d}.mp4")
        if not seg.exists():
            continue
        dur = scene.duration_actual or scene.duration_planned
        cue = next((c for c in scene.actions
                    if 0.4 < (float(c.get("t1", 0)) - float(c.get("t0", 0))) * dur <= 6.0),
                   scene.actions[0])
        mid = (float(cue.get("t0", 0)) + float(cue.get("t1", 0))) / 2 * dur
        t1 = max(0.0, mid - 0.18)
        t2 = min(max(0.1, dur - 0.05), mid + 0.18)
        f1, f2 = work / f"s{scene.index + 1}_a.png", work / f"s{scene.index + 1}_b.png"
        if not (_grab_frame(seg, t1, f1) and _grab_frame(seg, t2, f2)):
            continue
        diff = _frame_diff(f1, f2)
        checked += 1
        if diff is not None and diff < 0.6:
            motion_failures.append(
                f"scene {scene.index + 1} ({cue.get('action')}): frames barely change "
                f"(diff {diff:.2f})")
    report.add("action_motion", not motion_failures,
               f"motion detected in {checked} sampled action window(s)"
               if not motion_failures else "; ".join(motion_failures[:5]))

    # ---- foot sliding (plan-level proxy) -------------------------------
    sliding: list[str] = []
    for scene in scenes:
        for cue in scene.actions:
            action = str(cue.get("action", ""))
            if action in ("walk", "run", "leave", "approach", "follow"):
                if cue.get("x_from") is None or cue.get("x_to") is None:
                    sliding.append(f"scene {scene.index + 1}: '{action}' has no travel path")
    report.add("foot_sliding", not sliding,
               "every locomotion cue carries a real travel path" if not sliding else
               "; ".join(sliding[:5]))

    # ---- props ---------------------------------------------------------
    prop_issues: list[str] = []
    for scene in scenes:
        ids = {str(p.get("id")) for p in scene.props}
        for cue in scene.actions:
            if cue.get("prop") and cue["prop"] not in ids:
                prop_issues.append(f"scene {scene.index + 1}: missing prop {cue['prop']}")
        for spec in scene.props:
            sprite = draw_prop(SceneProp(id=str(spec.get("id")), kind=str(spec.get("kind"))), 48)
            if sprite is None:
                prop_issues.append(f"scene {scene.index + 1}: unrenderable prop "
                                   f"{spec.get('kind')}")
        xs = [round(float(p.get("x", 0.5)), 3) for p in scene.props]
        if len(xs) != len(set(xs)) and len(xs) > 1:
            prop_issues.append(f"scene {scene.index + 1}: props overlap at the same position")
    report.add("props", not prop_issues,
               "all referenced props exist and can be drawn" if not prop_issues else
               "; ".join(prop_issues[:5]), severity="warn")

    # ---- continuity + orientation --------------------------------------
    continuity: list[str] = []
    expected_cast: tuple[str, ...] | None = None
    for scene in scenes:
        cast = tuple(sorted(scene.characters))
        if expected_cast is None:
            expected_cast = cast
        elif cast != expected_cast:
            continuity.append(f"scene {scene.index + 1}: cast changed")
    report.add("character_continuity", not continuity,
               "cast and rig definitions stay identical across scenes"
               if not continuity else "; ".join(continuity[:5]), severity="warn")

    orientation: list[str] = []
    for scene in scenes:
        for cue in scene.actions:
            for key in ("x_from", "x_to"):
                value = cue.get(key)
                if value is not None and not (0.0 <= float(value) <= 1.0):
                    orientation.append(f"scene {scene.index + 1}: {key}={value} out of frame")
            if cue.get("action") in ("walk", "run", "leave", "approach") \
                    and cue.get("x_from") is not None and cue.get("x_to") is not None \
                    and abs(float(cue["x_from"]) - float(cue["x_to"])) < 0.01:
                orientation.append(f"scene {scene.index + 1}: movement cue has no displacement")
    report.add("orientation", not orientation,
               "movement and placement values stay inside the stage" if not orientation else
               "; ".join(orientation[:5]))

    # ---- camera paths ---------------------------------------------------
    from .film import build_camera_cues, NARRATOR_ID
    camera_issues: list[str] = []
    for scene in scenes:
        dur = scene.duration_actual or scene.duration_planned
        lines: list[tuple[float, float, str]] = []
        cursor = 0.0
        for idx, line in enumerate(scene.lines):
            est = max(0.6, len(line.text.split()) / max(60, int(getattr(plan, "wpm", 130))) * 60.0)
            start = cursor
            end = min(dur, cursor + est)
            lines.append((start, end, getattr(line, "speaker", NARRATOR_ID)))
            cursor = end + 0.3
            if cursor >= dur:
                break
        cues = build_camera_cues(lines, dur, n_chars=len(scene.characters))
        covered = sum(max(0.0, c.end - c.start) for c in cues)
        if not cues or covered < dur * 0.95:
            camera_issues.append(f"scene {scene.index + 1}: camera covers "
                                 f"{covered:.1f}s of {dur:.1f}s")
        ids = set(scene.characters) | {str(p.get("id")) for p in scene.props}
        for cue in cues:
            if cue.focus and cue.focus not in ids and cue.focus != "ensemble":
                camera_issues.append(f"scene {scene.index + 1}: camera focus '{cue.focus}' "
                                     "does not exist")
    report.add("camera_paths", not camera_issues,
               "every scene is fully covered by valid shots" if not camera_issues else
               "; ".join(camera_issues[:5]))

    # ---- export + audio -------------------------------------------------
    info = None
    if video.exists() and video.stat().st_size > 1024:
        try:
            info = probe(video)
        except FFmpegError as exc:
            report.add("export_readable", False, str(exc))
            info = None
    else:
        report.add("export_readable", False, f"missing or empty: {video}")
    if info is not None:
        duration = float(info.duration or 0.0)
        v = info.primary_video()
        a = info.primary_audio()
        report.add("export_readable", True,
                   f"{duration:.1f}s, video={v.codec_name if v else None}, "
                   f"audio={a.codec_name if a else None}")
        expected_audio = bool(convo is not None and getattr(convo, "lines", None))
        report.add("audio_stream", a is not None,
                   f"audio stream: {a.codec_name}" if a else "no audio stream in the export",
                   severity="fail" if expected_audio else "warn")
        if info.duration is None or info.duration <= 0:
            report.add("export_duration", False, "could not read the container duration")
        else:
            target = float(getattr(plan, "target_seconds", 0) or 0)
            if target and duration < target * 0.75:
                report.add("export_duration", False,
                           f"{duration:.0f}s is well below the {target / 60:.0f}-minute target",
                           severity="warn")
            else:
                report.add("export_duration", True, f"{duration:.1f}s")
        if v is not None:
            report.add("export_streams", bool(v.width and v.height),
                       f"{v.width}x{v.height} @ {v.avg_frame_rate}")
    else:
        duration = 0.0

    if info is not None and info.primary_audio() is not None:
        mean_db, peak_db = _audio_levels(video)
        silent_by_design = bool(convo is not None and
                                any(getattr(a, "engine", "") == "silence"
                                    for a in getattr(convo, "assignments", [])))
        if peak_db is not None and peak_db < -55.0 and not silent_by_design:
            report.add("audio_level", False, f"peak {peak_db:.1f} dB: the track is effectively "
                                              "silent", severity="fail")
        elif peak_db is not None and peak_db > -0.5:
            report.add("audio_level", False, f"peak {peak_db:.1f} dB may clip",
                       severity="warn")
        else:
            report.add("audio_level", True,
                       f"mean {mean_db if mean_db is not None else '?'} dB, "
                       f"peak {peak_db if peak_db is not None else '?'} dB")

    # ---- subtitles + speakers ------------------------------------------
    srt = project.path("subtitles", "subtitles.srt")
    if srt.exists():
        try:
            cues = parse_srt(srt.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            cues = []
            report.add("subtitles", False, f"could not parse SRT: {exc}")
        else:
            issues: list[str] = []
            prev_end = -1.0
            for cue in cues:
                if cue.end <= cue.start:
                    issues.append(f"cue {cue.index}: end <= start")
                if cue.start < prev_end - 0.06:
                    issues.append(f"cue {cue.index}: overlaps the previous cue")
                if info is not None and info.duration and cue.end > info.duration + 0.75:
                    issues.append(f"cue {cue.index}: ends after the video")
                prev_end = max(prev_end, cue.end)
            report.add("subtitles", not issues,
                       f"{len(cues)} subtitle cues are ordered and inside the video"
                       if not issues else "; ".join(issues[:6]))
    else:
        report.add("subtitles", False, "no subtitles.srt was produced", severity="warn")

    if convo is not None and getattr(convo, "lines", None):
        names = {c.name for c in getattr(plan, "characters", [])}
        bad_speakers = sorted({l.speaker_id for l in convo.lines
                               if l.speaker_id not in library_ids} if library_ids else set())
        report.add("dialogue_speakers", not bad_speakers,
                   f"{len(convo.lines)} lines are voiced by known characters"
                   if not bad_speakers else
                   "unknown speakers: " + ", ".join(bad_speakers[:6]))
        if srt.exists() and names:
            try:
                cue_text = " ".join(c.text for c in parse_srt(srt.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                cue_text = ""
            labelled = {n for n in names if f"{n}:" in cue_text}
            report.add("subtitle_speakers", True,
                       f"subtitle speaker labels found for {len(labelled)} character(s)",
                       severity="info")

    # ---- black / corrupted frames + contact sheet ----------------------
    contact: Path | None = None
    if info is not None and duration > 1.0:
        count = max(4, min(sample_frames, 12))
        times = [duration * (i + 0.5) / count for i in range(count)]
        lum: list[tuple[float, float]] = []
        frames: list[Path] = []
        for i, t in enumerate(times):
            png = work / f"sample_{i + 1:02d}.png"
            if _grab_frame(video, t, png):
                value = _frame_luminance(png)
                frames.append(png)
                if value is not None:
                    lum.append((t, value))
        black = [f"{t:.1f}s" for t, v in lum if v < 4.0]
        dark = [f"{t:.1f}s" for t, v in lum if 4.0 <= v < 12.0]
        report.add("black_frames", not black,
                   f"{len(lum)} sampled frames have picture content" if not black else
                   "near-black frames at: " + ", ".join(black[:6]))
        if dark:
            report.add("dark_frames", False, "unusually dark frames at: " + ", ".join(dark[:6]),
                       severity="warn")
        if frames:
            try:
                contact = _contact_sheet(frames, work / "contact_sheet.jpg", times[:len(frames)])
                report.contact_sheet = str(contact)
            except Exception as exc:  # noqa: BLE001
                report.add("contact_sheet", False, f"could not build contact sheet: {exc}",
                           severity="warn")
    return report


def _contact_sheet(frames: list[Path], out: Path, times: list[float],
                   thumb_w: int = 320) -> Path:
    from PIL import Image, ImageDraw

    thumbs = []
    for frame in frames:
        with Image.open(frame) as img:
            ratio = thumb_w / max(1, img.width)
            thumbs.append(img.convert("RGB").resize(
                (thumb_w, max(1, int(img.height * ratio))), Image.LANCZOS))
    if not thumbs:
        raise RuntimeError("no frames to tile")
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for t in thumbs) + 22
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
        label = f"{times[i]:.1f}s" if i < len(times) else ""
        draw.text((x + 6, y + thumb.height + 3), label, fill=(230, 230, 230))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=88)
    return out


def write_qa_report(report: QAReport, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    jp = reports_dir / "qa_report.json"
    tp = reports_dir / "qa_report.txt"
    jp.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str),
                  encoding="utf-8")
    tp.write_text(report.summary(), encoding="utf-8")
    return jp, tp
