"""Pipeline orchestrator — runs the full create→export flow with real progress.

Stages (section 16):

    [1/8] Preparing script
    [2/8] Planning scenes
    [3/8] Generating narration
    [4/8] Creating subtitles
    [5/8] Preparing visual assets
    [6/8] Rendering video
    [7/8] Assembling final video
    [8/8] Validating output

Every stage does genuine work, records its state in the project manifest, and
supports resume: completed stages are skipped when the project is re-opened.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .audio_mix import build_voice_track, mix_final_audio, wav_duration_seconds
from .config import Config
from .constants import (
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_FPS,
    DEFAULT_MUSIC_VOLUME,
    DEFAULT_WPM,
    PIPELINE_STAGES,
    RESOLUTIONS,
)
from .ffmpeg_utils import FFmpegError, probe
from .languages import LanguageRegistry, LanguageSettings
from .project import (
    STATUS_EXPORTED,
    STATUS_NARRATED,
    STATUS_PLANNED,
    STATUS_RENDERED,
    STATUS_SCENES_READY,
    STATUS_SCRIPT_READY,
    STATUS_THUMBNAILS,
    Project,
)
from .renderer import concat_segments, concat_segments_reencode, render_scene_segment
from .scene_planner import MANIFEST_NAME, Scene, SceneManifest, ScenePlanner
from .script_engine import Script, TemplateScriptEngine, parse_user_script
from .subtitles import build_track, write_subtitles
from .tts import (
    pick_engine,
    probe_all_engines,
    synthesize_with_fallback,
    wav_duration,
)
from .validation import ValidationReport, validate_output, write_report
from .visuals import generate_scene_asset


ProgressCallback = Callable[[str, int, int, dict[str, Any] | None], None]


def _noop_progress(stage: str, index: int, total: int, extra: dict | None) -> None:
    pass


@dataclass
class PipelineOptions:
    """Everything the pipeline needs beyond the project manifest."""

    duration_minutes: float = 10
    resolution: str = "1080p"
    fps: int = DEFAULT_FPS
    animation_mode: str = "hybrid"
    transition: str = "fade"
    wpm: int = DEFAULT_WPM
    burn_subtitles: bool = True
    music: str = "none"
    music_volume: float = DEFAULT_MUSIC_VOLUME
    tts_engine: str = "auto"
    tts_voice: str | None = None
    tts_speed: float = 1.0
    low_resource: bool = False
    duration_tolerance: float = DEFAULT_DURATION_TOLERANCE
    script_file: str | None = None
    facts: list[str] = field(default_factory=list)
    x264_preset: str = "veryfast"
    crf: int = 21
    allow_network: bool = False
    thumbnails: bool = True
    auto_select_thumbnail: bool = False
    subtitle_font_size: int = 42
    subtitle_margin_v: int = 112
    subtitle_outline: int = 2
    subtitle_shadow: int = 1
    subtitle_max_chars: int = 46
    subtitle_max_lines: int = 2

    @classmethod
    def from_config(cls, cfg: Config, **overrides: Any) -> "PipelineOptions":
        d = cfg.data
        opts = cls(
            duration_minutes=float(d["defaults"]["duration_minutes"]),
            resolution=d["defaults"]["resolution"],
            fps=int(d["defaults"]["fps"]),
            animation_mode=d["defaults"]["animation_mode"],
            transition=d["defaults"]["transition"],
            wpm=int(d["defaults"]["wpm"]),
            burn_subtitles=bool(d["defaults"]["burn_subtitles"]),
            music=d["defaults"]["music"],
            music_volume=float(d["defaults"]["music_volume"]),
            tts_engine=d["tts"]["engine"],
            tts_voice=d["tts"].get("voice"),
            tts_speed=float(d["tts"]["speed"]),
            low_resource=bool(d["defaults"]["low_resource"]),
            duration_tolerance=float(d["defaults"]["duration_tolerance"]),
            x264_preset=d["render"]["x264_preset"],
            crf=int(d["render"]["crf"]),
            allow_network=bool(d["network"].get("allow_translation_upload", False)),
            subtitle_font_size=int(d["subtitles"]["font_size"]),
            subtitle_margin_v=int(d["subtitles"]["margin_v"]),
            subtitle_outline=int(d["subtitles"]["outline"]),
            subtitle_shadow=int(d["subtitles"]["shadow"]),
            subtitle_max_chars=int(d["subtitles"]["max_line_chars"]),
            subtitle_max_lines=int(d["subtitles"]["max_lines"]),
            thumbnails=bool(d.get("thumbnails", {}).get("enabled", True)),
            auto_select_thumbnail=bool(d.get("thumbnails", {}).get("auto_select", False)),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(opts, k):
                setattr(opts, k, v)
        return opts

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PipelineResult:
    ok: bool
    project: Project
    output_path: Path | None
    report: ValidationReport | None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        project: Project,
        options: PipelineOptions,
        *,
        progress: ProgressCallback | None = None,
        languages: LanguageRegistry | None = None,
    ):
        self.project = project
        self.opts = options
        self.progress = progress or _noop_progress
        self.langs = languages or LanguageRegistry.builtin()
        self.total = len(PIPELINE_STAGES)
        self._start = time.time()

    # ------------------------------------------------------------------ util
    def _stage(self, i: int, name: str) -> None:
        self.progress(name, i, self.total, {"elapsed": time.time() - self._start})

    def _log(self, message: str) -> None:
        log_dir = self.project.stage_dir("logs")
        line = f"[{time.strftime('%H:%M:%S')}] {message}\n"
        with (log_dir / "pipeline.log").open("a", encoding="utf-8") as f:
            f.write(line)

    # ------------------------------------------------------------------ run
    def run(self, *, resume: bool = True) -> PipelineResult:
        p = self.project
        notes: list[str] = []
        try:
            lang_settings = LanguageSettings.from_dict(p.manifest.get("languages", {}))

            # Stage 1: script
            self._stage(1, PIPELINE_STAGES[0])
            if resume and p.stage_done(STATUS_SCRIPT_READY) and p.read_json("script", "script.json"):
                script = Script.from_dict(p.read_json("script", "script.json"))
                self._log("resume: reusing existing script")
            else:
                script = self._prepare_script(lang_settings)
                p.write_json("script", "script.json", script.to_dict())
                p.mark_stage(STATUS_SCRIPT_READY)
            notes.extend(script.warnings)

            # Stage 2: scenes
            self._stage(2, PIPELINE_STAGES[1])
            existing_manifest = p.read_json("scenes", MANIFEST_NAME)
            if resume and p.stage_done(STATUS_SCENES_READY) and existing_manifest:
                manifest = SceneManifest.from_dict(existing_manifest)
                self._log("resume: reusing existing scene manifest")
            else:
                planner = ScenePlanner(wpm=script.target_wpm, seed=None)
                target_seconds = self.opts.duration_minutes * 60.0
                manifest = planner.plan(script, target_seconds,
                                        animation_mode=self.opts.animation_mode,
                                        transition=self.opts.transition)
                problems = manifest.validate()
                if problems:
                    raise FFmpegError("invalid scene manifest: " + "; ".join(problems[:5]))
                p.write_json("scenes", MANIFEST_NAME, manifest.to_dict())
                p.mark_stage(STATUS_SCENES_READY)
            notes.extend(manifest.notes)

            # Stage 3: narration
            self._stage(3, PIPELINE_STAGES[2])
            narration_info = self._generate_narration(manifest, lang_settings)
            notes.extend(narration_info.get("notes", []))
            p.mark_stage(STATUS_NARRATED)

            # Honest duration top-up: when measured narration is well short of
            # the request (e.g. a fast TTS voice), extend the *content* with new
            # composed scenes and narrate them.  Never stretch with silence and
            # never repeat existing scenes.
            for _attempt in range(2):
                if not self._top_up(manifest, script, narration_info, lang_settings):
                    break
                p.write_json("scenes", MANIFEST_NAME, manifest.to_dict())
            notes.extend(narration_info.get("notes", []))

            # Re-sync scene durations to real narration lengths.
            real_durations = narration_info.get("durations", {})
            ScenePlanner.resync_durations(
                manifest, real_durations,
                target_seconds=self.opts.duration_minutes * 60.0,
            )
            p.write_json("scenes", MANIFEST_NAME, manifest.to_dict())

            # Stage 4: subtitles
            self._stage(4, PIPELINE_STAGES[3])
            subtitle_files = self._create_subtitles(manifest, lang_settings, real_durations)

            # Stage 5: visual assets
            self._stage(5, PIPELINE_STAGES[4])
            assets_ok = self._generate_assets(manifest, lang_settings)
            if assets_ok == 0:
                raise FFmpegError("no visual assets could be generated")

            # Stage 6: render scenes
            self._stage(6, PIPELINE_STAGES[5])
            rendered = self._render_scenes(manifest, lang_settings, subtitle_files)
            if rendered:
                p.mark_stage(STATUS_RENDERED)

            # Stage 7: assemble
            self._stage(7, PIPELINE_STAGES[6])
            output_path, total_seconds, methods = self._assemble(manifest, rendered, narration_info)

            # Stage 8: validate
            self._stage(8, PIPELINE_STAGES[7])
            report = self._validate(output_path, manifest, subtitle_files, total_seconds, methods, notes)

            p.manifest["export"] = {
                "path": str(output_path),
                "resolution": self.opts.resolution,
                "fps": self.opts.fps,
                "duration_seconds": report.actual_duration,
                "validated": report.ok,
            }
            p.mark_stage(STATUS_EXPORTED)

            # Stage 9: thumbnails (section 12) — always attempted after a
            # successful export, but isolated so a thumbnail failure never
            # destroys an otherwise successful video project.
            self._stage(9, PIPELINE_STAGES[8])
            if self.opts.thumbnails:
                already = (p.stage_done(STATUS_THUMBNAILS)
                           and p.read_json("thumbnails", "thumbnail_recommendations.json"))
                if resume and already:
                    self._log("resume: reusing existing thumbnails")
                else:
                    self._generate_thumbnails(lang_settings)

            self.progress("Complete", self.total, self.total,
                          {"output": str(output_path), "ok": report.ok})
            return PipelineResult(report.ok, p, output_path, report, notes=notes)

        except Exception as exc:  # noqa: BLE001 — surface any failure honestly
            self._log(f"pipeline failed: {traceback.format_exc()}")
            p.mark_stage(p.status if p.status != STATUS_PLANNED else STATUS_SCRIPT_READY,
                         error=str(exc))
            p.manifest["status"] = "failed"
            p.add_error(str(exc))
            return PipelineResult(False, p, None, None, error=f"{type(exc).__name__}: {exc}", notes=notes)

    # ---------------------------------------------------------------- script
    def _prepare_script(self, lang_settings: LanguageSettings) -> Script:
        script_lang = lang_settings.resolved("script")
        topic_for_display = None

        if self.opts.script_file:
            sf = Path(self.opts.script_file)
            if not sf.exists():
                raise FFmpegError(f"script file not found: {sf}")
            text = sf.read_text(encoding="utf-8")
            from .script_engine import extract_topic
            script = parse_user_script(text, language=script_lang,
                                       topic=extract_topic(self.project.manifest.get("prompt", "")))
            self._log(f"loaded user script from {sf} ({script.actual_words} words)")
            return script

        engine = TemplateScriptEngine(wpm=self.opts.wpm)
        script = engine.generate(
            self.project.manifest.get("prompt", ""),
            target_minutes=self.opts.duration_minutes,
            language=script_lang,
            facts=self.opts.facts,
        )
        # Save per-section files for inspection/editing.
        script_dir = self.project.stage_dir("script")
        for sec in script.sections:
            (script_dir / f"{sec.id}.txt").write_text(sec.narration, encoding="utf-8")
        (script_dir / "full_script.txt").write_text(
            "\n\n".join(f"# {s.title}\n{s.narration}" for s in script.sections),
            encoding="utf-8",
        )
        self._log(f"generated template script: {script.actual_words} words, "
                  f"~{script.estimated_seconds / 60:.1f} min at {script.target_wpm} wpm")
        return script

    # ------------------------------------------------------------ narration
    def _generate_narration(self, manifest: SceneManifest, lang_settings: LanguageSettings) -> dict:
        narration_lang = lang_settings.resolved("narration")
        engine = pick_engine(self.opts.tts_engine, language=narration_lang,
                             config={"wpm": self.opts.wpm, "volume": 1.0})
        info = engine.probe()
        self._log(f"TTS engine: {engine.name} — {info.get('detail', '')}")

        narration_dir = self.project.stage_dir("narration")
        per_scene: list[tuple[str, Path]] = []
        durations: dict[str, float] = {}
        notes: list[str] = []
        engine_notes: list[str] = []
        methods: dict[str, str] = {}

        for i, scene in enumerate(manifest.scenes, start=1):
            out_wav = narration_dir / f"{scene.scene_id}.wav"
            # Resume: reuse existing narration if present and non-empty.
            if out_wav.exists() and out_wav.stat().st_size > 44 and scene.status in ("narrated", "rendered"):
                durations[scene.scene_id] = wav_duration(out_wav)
                per_scene.append((scene.scene_id, out_wav))
                methods[scene.scene_id] = scene.assets.get("narration_engine", engine.name)
                continue
            res = synthesize_with_fallback(
                scene.narration_text or scene.title,
                out_wav,
                preferred=self.opts.tts_engine,
                language=narration_lang,
                voice_id=self.opts.tts_voice,
                speed=self.opts.tts_speed,
                config={"wpm": self.opts.wpm},
            )
            if not res.ok or res.audio_path is None:
                scene.status = "failed"
                scene.error = res.error
                raise FFmpegError(f"narration failed for {scene.scene_id}: {res.error}")
            durations[scene.scene_id] = res.duration_seconds
            per_scene.append((scene.scene_id, res.audio_path))
            methods[scene.scene_id] = res.engine
            scene.narration_file = str(res.audio_path)
            scene.assets["narration_engine"] = res.engine
            scene.status = "narrated"
            for n in res.notes or []:
                if n not in engine_notes:
                    engine_notes.append(n)
            self.progress("Generating narration", i, len(manifest.scenes),
                          {"scene": scene.scene_id, "engine": res.engine})

        voice_path = narration_dir / "voice_track.wav"
        voice_path, total_seconds, _pad = build_voice_track(
            per_scene, [s.duration_seconds for s in manifest.scenes], voice_path
        )
        if engine.name == "silence":
            notes.append("Narration track is silent: no TTS engine with usable voices was available. "
                         "Install Piper/espeak-ng or a Windows voice for spoken narration.")
        notes.extend(engine_notes)

        self.project.manifest["narration"] = {
            "engine": engine.name,
            "language": narration_lang,
            "voice": self.opts.tts_voice,
            "total_seconds": total_seconds,
            "scene_durations": durations,
            "methods": methods,
        }
        self.project.manifest.setdefault("methods_used", {})["narration"] = engine.name
        self.project.manifest.setdefault("providers_used", {})["tts"] = engine.name
        self.project.write_json("narration", "narration.json", self.project.manifest["narration"])
        return {"voice_path": voice_path, "durations": durations, "total_seconds": total_seconds,
                "notes": notes, "engine": engine.name}

    # ------------------------------------------------------------------ top-up
    def _top_up(self, manifest: SceneManifest, script: Script, narration_info: dict,
                lang_settings: LanguageSettings) -> bool:
        """Extend content when measured narration is meaningfully short.

        Returns True when new scenes were added.  Content comes from the
        deep-dive scaffold rotation in the script engine, so the extension is
        further exploration rather than filler or repeated scenes.
        """
        from .constants import CAMERA_MOVES, MAX_SCENE_SECONDS, MIN_SCENE_SECONDS
        from .script_engine import compose_extension_narration

        target = self.opts.duration_minutes * 60.0
        if target <= 0:
            return False
        # Compare against the duration implied by *measured* narration, not the
        # planner's optimistic estimate (which is already scaled to target).
        measured = narration_info.get("durations") or {}

        def projected(scene: Scene) -> float:
            nd = measured.get(scene.scene_id)
            if nd and nd > 0:
                return max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, nd + 0.35))
            return scene.duration_seconds

        total = sum(projected(s) for s in manifest.scenes)
        shortfall = target - total
        if shortfall <= max(15.0, target * 0.08):
            return False

        # Calibrate against the *measured* narration pace.
        words_total = sum(len(s.narration_text.split()) for s in manifest.scenes)
        narr_secs = sum((narration_info.get("durations") or {}).values())
        if narr_secs <= 0:
            return False
        actual_wpm = max(60.0, min(240.0, words_total / narr_secs * 60.0))

        add_seconds = min(shortfall, max(20.0, target * 0.5))
        n_new = max(1, min(8, int(round(add_seconds / 22.0))))
        # Keep each extension scene's narration under ~40s so scenes stay well
        # under the render clamp and nothing is truncated at mux time.
        words_per = min(max(20, int(add_seconds * actual_wpm / 60.0 / n_new)),
                        int(40.0 * actual_wpm / 60.0))

        narration_dir = self.project.stage_dir("narration")
        narration_lang = lang_settings.resolved("narration")
        existing = {s.scene_id for s in manifest.scenes}
        next_id = 900

        new_scenes: list[Scene] = []
        for k in range(n_new):
            while f"scene_{next_id:03d}" in existing:
                next_id += 1
            sid = f"scene_{next_id:03d}"
            existing.add(sid)
            next_id += 1
            text = compose_extension_narration(script.topic, words_per,
                                               seed_index=len(manifest.scenes) + k)
            new_scenes.append(
                Scene(
                    scene_id=sid,
                    title=f"Extended Look {k + 1}",
                    narration_text=text,
                    duration_seconds=max(MIN_SCENE_SECONDS, words_per / actual_wpm * 60.0),
                    visual_description=f"Illustrated deep dive: {script.topic}",
                    animation_method="animated_illustration",
                    camera_move=CAMERA_MOVES[(len(manifest.scenes) + k) % len(CAMERA_MOVES)],
                    transition_in=self.opts.transition,
                    style="illustration",
                    subtitle_text=text,
                    status="planned",
                )
            )

        durations = narration_info.setdefault("durations", {})
        for i, scene in enumerate(new_scenes, start=1):
            wav = narration_dir / f"{scene.scene_id}.wav"
            res = synthesize_with_fallback(
                scene.narration_text, wav,
                preferred=self.opts.tts_engine,
                language=narration_lang,
                voice_id=self.opts.tts_voice,
                speed=self.opts.tts_speed,
                config={"wpm": self.opts.wpm},
            )
            if not res.ok or res.audio_path is None:
                raise FFmpegError(f"top-up narration failed for {scene.scene_id}: {res.error}")
            durations[scene.scene_id] = res.duration_seconds
            scene.narration_file = str(res.audio_path)
            scene.assets["narration_engine"] = res.engine
            scene.status = "narrated"
            self.progress("Generating narration", i, len(new_scenes),
                          {"scene": scene.scene_id, "engine": res.engine})

        # Insert before the closing scene so the ending stays the ending.
        if len(manifest.scenes) >= 2:
            manifest.scenes[-1:-1] = new_scenes
        else:
            manifest.scenes.extend(new_scenes)

        # Rebuild the voice track across all scenes, in order.
        per_scene: list[tuple[str, Path]] = []
        for scene in manifest.scenes:
            wav = (Path(scene.narration_file) if scene.narration_file
                   else narration_dir / f"{scene.scene_id}.wav")
            if wav.exists():
                per_scene.append((scene.scene_id, wav))
                durations.setdefault(scene.scene_id, wav_duration(wav))
        voice_path = narration_dir / "voice_track.wav"
        voice_path, total_seconds, _ = build_voice_track(
            per_scene, [s.duration_seconds for s in manifest.scenes], voice_path)
        narration_info["voice_path"] = voice_path
        narration_info["total_seconds"] = total_seconds
        narration_info.setdefault("notes", []).append(
            f"Extended content with {len(new_scenes)} additional scene(s): measured narration "
            f"pace (~{actual_wpm:.0f} wpm) was faster than planned, so the script was grown "
            f"instead of padding with silence or repeating scenes."
        )
        ScenePlanner.resync_durations(manifest, durations, target_seconds=target)
        self.project.manifest["narration"] = {
            **(self.project.manifest.get("narration") or {}),
            "total_seconds": total_seconds,
            "scene_durations": durations,
        }
        self._log(f"top-up: +{len(new_scenes)} scene(s), measured pace {actual_wpm:.1f} wpm, "
                  f"new total {sum(s.duration_seconds for s in manifest.scenes):.1f}s")
        return True

    # ------------------------------------------------------------ subtitles
    def _titles_via_libass(self, lang_settings: LanguageSettings) -> tuple[bool, str]:
        """Whether on-screen titles need libass shaping, plus the font family.

        Pillow without libraqm cannot shape Arabic/Hebrew/Indic scripts, so for
        those the scene title is burned as an ASS overlay instead of drawn into
        the illustration.
        """
        from .fonts import family_name_for, find_font_for_script, script_needs_shaping

        onscreen = lang_settings.resolved("onscreen")
        try:
            family = self.langs.require(onscreen).script
        except KeyError:
            family = "latin"
        if not script_needs_shaping(family):
            return False, "Arial"
        res = find_font_for_script(family)
        return True, (family_name_for(res.font.filename) if res.ok else "Arial")

    def _create_subtitles(self, manifest: SceneManifest, lang_settings: LanguageSettings,
                          real_durations: dict[str, float]) -> dict[str, Any]:
        from .fonts import family_name_for, find_font_for_script
        from .subtitles import build_scene_cues, to_ass

        sub_lang = lang_settings.resolved("subtitle")
        onscreen_lang = lang_settings.resolved("onscreen")
        rtl = False
        family = "latin"
        try:
            lang = self.langs.require(sub_lang)
            rtl = lang.rtl
            family = lang.script
        except KeyError:
            pass

        # Global track -> .srt / .vtt for players and YouTube uploads.
        scene_dicts = [s.to_dict() for s in manifest.scenes]
        track = build_track(
            scene_dicts, language=sub_lang, rtl=rtl,
            narration_durations=real_durations,
            max_chars=self.opts.subtitle_max_chars,
            max_lines=self.opts.subtitle_max_lines,
        )
        problems = track.validate()
        if problems:
            self._log(f"subtitle warnings: {problems[:3]}")
            self.project.add_warning("subtitle timing issues: " + "; ".join(problems[:3]))
        sub_dir = self.project.stage_dir("subtitles")
        files = write_subtitles(track, sub_dir)

        # Per-scene ASS with LOCAL timestamps, so burning each scene segment
        # shows that scene's lines (not the first scene's lines repeated).
        font_res = find_font_for_script(family)
        font_family = family_name_for(font_res.font.filename) if font_res.ok else "Arial"
        if not font_res.ok:
            self.project.add_warning(f"subtitle font for {family}: {font_res.guidance}")
        per_scene_dir = sub_dir / "scenes"
        per_scene_dir.mkdir(parents=True, exist_ok=True)
        titles_via_libass, title_font_family = self._titles_via_libass(lang_settings)
        per_scene: dict[str, Path] = {}
        for sc in manifest.scenes:
            cues = build_scene_cues(
                sc.to_dict(), real_durations,
                max_chars=self.opts.subtitle_max_chars,
                max_lines=self.opts.subtitle_max_lines,
            )
            show_title = titles_via_libass and bool((sc.title or "").strip())
            if not cues and not show_title:
                continue
            ass_path = per_scene_dir / f"{sc.scene_id}.ass"
            ass_path.write_text(
                to_ass(
                    cues,
                    font_family=font_family,
                    font_size=self.opts.subtitle_font_size,
                    margin_v=self.opts.subtitle_margin_v,
                    outline=self.opts.subtitle_outline,
                    shadow=self.opts.subtitle_shadow,
                    rtl=rtl,
                    title=sc.title if show_title else None,
                    title_seconds=sc.duration_seconds if show_title else 0.0,
                    title_font_family=title_font_family,
                ),
                encoding="utf-8",
            )
            per_scene[sc.scene_id] = ass_path

        self.project.manifest["subtitles"] = {
            "language": sub_lang,
            "rtl": rtl,
            "cues": len(track.cues),
            "timing_source": track.timing_source,
            "font_family": font_family,
            "font_source": font_res.font.filename if font_res.ok else None,
            "fonts_checked_for": onscreen_lang,
            "files": {k: str(v) for k, v in files.items()},
            "per_scene_dir": str(per_scene_dir),
        }
        return {**files, "per_scene": per_scene, "font_family": font_family}

    # ---------------------------------------------------------------- assets
    def _generate_assets(self, manifest: SceneManifest, lang_settings: LanguageSettings) -> int:
        w, h = RESOLUTIONS[self.opts.resolution]
        # Generate assets at 1.35x for zoompan headroom (cap for low-resource).
        if self.opts.low_resource:
            asset_scale = 1.15
        else:
            asset_scale = 1.35
        aw, ah = int(w * asset_scale), int(h * asset_scale)
        # Keep even dimensions for yuv420p friendliness.
        aw, ah = aw - (aw % 2), ah - (ah % 2)
        onscreen_lang = lang_settings.resolved("onscreen")
        titles_via_libass, _ = self._titles_via_libass(lang_settings)
        assets_dir = self.project.stage_dir("assets")
        ok = 0
        for i, scene in enumerate(manifest.scenes, start=1):
            img = assets_dir / f"{scene.scene_id}.png"
            if img.exists() and img.stat().st_size > 0 and scene.status in ("assets_ready", "narrated", "rendered"):
                ok += 1
                continue
            res = generate_scene_asset(scene, assets_dir, width=aw, height=ah,
                                       fps=self.opts.fps, script_language=onscreen_lang,
                                       bake_title=not titles_via_libass)
            if res.ok:
                ok += 1
                scene.status = "assets_ready"
                scene.assets["image"] = str(res.image_path)
                scene.animation_method = res.method
            else:
                scene.status = "failed"
                scene.error = res.error
                self.project.add_error(f"asset {scene.scene_id}: {res.error}")
            self.progress("Preparing visual assets", i, len(manifest.scenes),
                          {"scene": scene.scene_id, "ok": res.ok})
        return ok

    # --------------------------------------------------------------- render
    def _render_scenes(self, manifest: SceneManifest, lang_settings: LanguageSettings,
                       subtitle_files: dict[str, Path]) -> list[Path]:
        renders_dir = self.project.stage_dir("renders")
        seg_dir = renders_dir / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        w, h = RESOLUTIONS[self.opts.resolution]
        # Each segment is burned with ITS OWN subtitle file (local timestamps).
        per_scene_subs: dict[str, Path] = subtitle_files.get("per_scene", {}) \
            if self.opts.burn_subtitles else {}

        segments: list[Path] = []
        fade = 0.35 if self.opts.transition != "none" else 0.0
        preset = "ultrafast" if self.opts.low_resource else self.opts.x264_preset

        for i, scene in enumerate(manifest.scenes, start=1):
            seg = seg_dir / f"{scene.scene_id}_segment.mp4"
            if seg.exists() and seg.stat().st_size > 0 and scene.status == "rendered":
                segments.append(seg)
                continue
            img = Path(scene.assets.get("image", ""))
            if not img.exists():
                # Try regenerating.
                res = generate_scene_asset(scene, self.project.stage_dir("assets"),
                                           width=w, height=h, fps=self.opts.fps,
                                           script_language=lang_settings.resolved("onscreen"),
                                           bake_title=not self._titles_via_libass(lang_settings)[0])
                if not res.ok or res.image_path is None:
                    raise FFmpegError(f"asset missing for {scene.scene_id}: {scene.error}")
                img = res.image_path
            from .visuals import MotionSpec
            motion = MotionSpec(
                camera_move=scene.camera_move,
                duration_seconds=scene.duration_seconds,
                fps=self.opts.fps,
                input_w=int(img_width(img) * 0.75) or w,
                input_h=int(img_height(img) * 0.75) or h,
                out_w=w, out_h=h,
            )
            r = render_scene_segment(
                img, seg, motion,
                subtitle_path=per_scene_subs.get(scene.scene_id),
                crf=self.opts.crf, preset=preset,
                fade_in=fade if i > 1 else 0.0,
                fade_out=fade if i < len(manifest.scenes) else 0.0,
            )
            if not r.ok:
                scene.status = "failed"
                scene.error = r.error
                raise FFmpegError(f"render failed for {scene.scene_id}: {r.error}")
            scene.status = "rendered"
            segments.append(seg)
            self.progress("Rendering video", i, len(manifest.scenes),
                          {"scene": scene.scene_id, "camera": scene.camera_move})
        self.project.write_json("scenes", MANIFEST_NAME, manifest.to_dict())
        return segments

    # -------------------------------------------------------------- assemble
    def _assemble(self, manifest: SceneManifest, segments: list[Path],
                  narration_info: dict) -> tuple[Path, float, dict]:
        renders_dir = self.project.stage_dir("renders")
        w, h = RESOLUTIONS[self.opts.resolution]
        video_only = renders_dir / "video_only.mp4"
        try:
            concat_segments(segments, video_only)
            info = probe(video_only)
            if not info.video_streams or (info.duration or 0) < 1.0:
                raise FFmpegError("concat demuxer produced unreadable video; re-encoding")
        except FFmpegError:
            self._log("concat copy failed, falling back to re-encode")
            video_only = concat_segments_reencode(
                segments, renders_dir / "video_only_reenc.mp4",
                fps=self.opts.fps, width=w, height=h,
                crf=self.opts.crf, preset=self.opts.x264_preset,
            )

        voice_path = Path(narration_info["voice_path"])
        total_seconds = narration_info["total_seconds"] or manifest.total_estimated_seconds
        mix = mix_final_audio(
            voice_path, renders_dir / "final_audio.wav",
            total_seconds=total_seconds,
            music=self.opts.music,
            music_volume=self.opts.music_volume,
        )
        if not mix.ok:
            raise FFmpegError(f"audio mix failed: {mix.error}")

        container = "mp4"
        final_path = renders_dir / f"freeanimate_final.{container}"
        run_args = [
            "-i", str(video_only),
            "-i", str(mix.final_mix_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            "-shortest",
            str(final_path),
        ]
        from .ffmpeg_utils import run_ffmpeg
        run_ffmpeg(run_args, timeout=3600)

        methods = {
            "scenes": len(manifest.scenes),
            "animation_methods": sorted({s.animation_method for s in manifest.scenes}),
            "music": mix.music_used,
            "narration_engine": narration_info.get("engine"),
            "ai_video_used": any(s.animation_method == "ai_video" for s in manifest.scenes),
            "upscaled": False,
            "note": "Output resolution equals asset working resolution scaled to export size; "
                    "illustrations are vector-raster composites, not native camera footage.",
        }
        self.project.manifest["methods_used"] = {**self.project.manifest.get("methods_used", {}), **methods}
        return final_path, total_seconds, methods

    # -------------------------------------------------------------- validate
    def _validate(self, output_path: Path, manifest: SceneManifest,
                  subtitle_files: dict[str, Path], total_seconds: float,
                  methods: dict, notes: list[str]) -> ValidationReport:
        w, h = RESOLUTIONS[self.opts.resolution]
        files = [Path(subtitle_files[k]) for k in ("srt", "vtt") if k in subtitle_files]
        report = validate_output(
            output_path,
            expected_width=w, expected_height=h,
            expected_fps=float(self.opts.fps),
            requested_seconds=self.opts.duration_minutes * 60.0,
            duration_tolerance=self.opts.duration_tolerance,
            require_audio=True,
            subtitle_files=files,
            expected_scenes=len(manifest.scenes),
            rendered_segments=len(manifest.scenes),
            methods_used=methods,
            limitations=notes,
        )
        write_report(report, self.project.stage_dir("reports"))
        return report

    # ------------------------------------------------------------ thumbnails
    def _generate_thumbnails(self, lang_settings: LanguageSettings) -> dict | None:
        """Generate, score, and recommend three YouTube thumbnails (section 12).

        Failures are isolated: they are recorded as warnings and never turn a
        successful export into a failed pipeline run.
        """
        from .thumbnails import generate_thumbnails as _gen
        from .thumbnails import select_thumbnail as _select

        slug = self.project.manifest.get("slug", "")
        try:
            report = _gen(self.project, languages=lang_settings, cfg={})
            self.project.manifest["thumbnails"] = {
                "recommended": report["recommendation"]["index"],
                "reasoning": report["recommendation"]["reasoning"],
                "options": [
                    {k: o.get(k) for k in ("index", "file", "concept",
                                           "concept_label", "total",
                                           "passed_quality")}
                    for o in report["options"]
                ],
                "report": "thumbnails/thumbnail_recommendations.json",
                "generated_at": report["generated_at"],
            }
            if (self.opts.auto_select_thumbnail
                    and report["recommendation"]["index"]):
                sel = _select(self.project, "accept", overwrite_confirmed=True)
                self.project.manifest["thumbnails"]["selected"] = sel["index"]
            self.project.mark_stage(STATUS_THUMBNAILS)
            for w in report.get("warnings", []):
                self.project.add_warning(f"thumbnails: {w}")
            if report["recommendation"]["index"] is None:
                self.project.add_warning(
                    "thumbnails: no recommendation (all options failed quality "
                    f"checks). Retry with: freeanimate thumbnails regenerate {slug}")
            self._log(f"thumbnails: recommended={report['recommendation']['index']}")
            return report
        except Exception as exc:  # noqa: BLE001 — video must survive
            self._log(f"thumbnail stage failed: {exc}")
            self.project.manifest.setdefault("stages", {})[STATUS_THUMBNAILS] = {
                "done": False, "error": str(exc)}
            self.project.add_warning(
                f"thumbnail generation failed: {exc}. The video remains valid; "
                f"retry with: freeanimate thumbnails generate {slug}")
            self.project.save()
            return None


# ---------------------------------------------------------------------------
# Image dimension helpers (avoid hard PIL dependency at import in hot paths)
# ---------------------------------------------------------------------------
_img_dims_cache: dict[str, tuple[int, int]] = {}


def img_width(path: Path) -> int:
    return _img_dims(path)[0]


def img_height(path: Path) -> int:
    return _img_dims(path)[1]


def _img_dims(path: Path) -> tuple[int, int]:
    key = str(path)
    if key in _img_dims_cache:
        return _img_dims_cache[key]
    try:
        from PIL import Image
        with Image.open(path) as im:
            dims = im.size
    except Exception:  # noqa: BLE001
        dims = (1920, 1080)
    _img_dims_cache[key] = dims
    return dims
