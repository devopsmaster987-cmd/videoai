"""End-to-end studio: prompt -> conversation video (real, local, honest).

This is the orchestration layer used by the web UI and the ``create`` CLI
command.  It reuses the existing project/manifest/audio/subtitle/validation
infrastructure and the new 2D rig, prompt planner, character library,
backgrounds, lip-sync and conversation engine.

Requested capabilities that are unavailable (Blender 3D, local diffusion
models) are *reported* and the studio falls back to the genuine procedural paths
instead of pretending.  Every generated artifact is labelled in the report.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .animation2d import (CharacterPlacement, CharacterRig, DialogueSpan, SceneAnimation,
                          METHOD_NAME as ANIM_2D_METHOD, render_scene_animation)
from .audio_mix import mix_final_audio, wav_duration_seconds
from .backgrounds import environment_for_plan, render_cached
from .blender3d import find_blender, render_conversation_3d, METHOD_NAME as BLENDER_METHOD
from .characters import Character, CharacterLibrary, ensure_project_characters
from .config import Config
from .conversation import ConversationResult, RenderedLine, render_conversation
from .ffmpeg_utils import FFmpegError, has_filter, run_ffmpeg
from .hardware import detect_hardware
from .local_models import (check_image_backend, generate_image_diffusers, generate_image_comfyui,
                           GenerationResult)
from .project import (Project, STATUS_EXPORTED, STATUS_NARRATED, STATUS_RENDERED,
                      STATUS_SCRIPT_READY, resolve_project)
from .prompt_understanding import DialogueLine, PromptPlan, understand_prompt
from .renderer import concat_segments
from .subtitles import write_subtitles
from .validation import validate_output, write_report

ProgressFn = Callable[[str, float, str | None], None]
CancelFn = Callable[[], bool]
RESOLUTIONS = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}


@dataclass
class StudioOptions:
    title: str | None = None
    method: str = "auto"                 # auto | 2d | 3d | 3d_cartoon | realistic |
                                         # realistic_3d | cinematic | photoreal |
                                         # still_image | ai_image | ai_video |
                                         # educational | children
    resolution: str = "720p"
    fps: int = 24
    language: str = "en"
    music: str = "none"                  # none | procedural | <path>
    music_volume: float = 0.18
    burn_subtitles: bool = True
    dialogue: list[DialogueLine] | None = None
    duration_seconds: float | None = None
    crf: int = 20
    preset: str = "veryfast"
    tts_engine: str = "auto"
    tts_voice: str | None = None
    tts_speed: float = 1.0
    aspect_ratio: str = "16:9"
    seed: str = ""
    characters_library: str | None = None
    image_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.dialogue is not None:
            d["dialogue"] = [x.model_dump() for x in self.dialogue]
        return d


@dataclass
class StudioResult:
    ok: bool
    project: Project | None = None
    output: Path | None = None
    thumbnail: Path | None = None
    method: str = ""
    requested_method: str = ""
    plan: PromptPlan | None = None
    conversation: ConversationResult | None = None
    validation: Any = None
    report_json: Path | None = None
    report_txt: Path | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    duration: float = 0.0
    scene_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project": str(self.project.root) if self.project else None,
            "project_slug": self.project.manifest.get("slug") if self.project else None,
            "output": str(self.output) if self.output else None,
            "thumbnail": str(self.thumbnail) if self.thumbnail else None,
            "method": self.method,
            "requested_method": self.requested_method,
            "duration": round(self.duration, 2),
            "scene_count": self.scene_count,
            "notes": self.notes,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "error": self.error,
            "validation_ok": getattr(self.validation, "ok", None),
            "report_json": str(self.report_json) if self.report_json else None,
            "report_txt": str(self.report_txt) if self.report_txt else None,
            "voices": ([a.to_dict() for a in self.conversation.assignments]
                       if self.conversation else []),
            "lines": ([l.to_dict() for l in self.conversation.lines]
                      if self.conversation else []),
            "subtitles": ({"srt": str(self.project.path("subtitles", "subtitles.srt"))
                           if self.project and self.project.path("subtitles", "subtitles.srt").exists()
                           else None} if self.project else None),
        }


def _resolution_dims(name: str, aspect: str = "16:9") -> tuple[int, int]:
    if aspect in ("9:16", "vertical"):
        short = RESOLUTIONS.get(name if name in RESOLUTIONS else "720p")
        h, w = short if short[0] > short[1] else (short[1], short[0])
        # For portrait use the long edge of the chosen preset as height.
        long_edge = max(short)
        return (round(long_edge * 9 / 16) // 2 * 2, long_edge // 2 * 2)
    w, h = RESOLUTIONS.get(name, RESOLUTIONS["720p"])
    return w, h


def _subtitle_filter(path: Path) -> str:
    sub = str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return f"subtitles='{sub}'"


def _camera_move(shot: str | None, index: int) -> str:
    text = (shot or "").lower()
    if "pull" in text or "out" in text:
        return "pull_out"
    if "pan" in text and "left" in text:
        return "pan_left"
    if "pan" in text and "down" in text:
        return "pan_down"
    if "pan" in text and "up" in text:
        return "pan_up"
    if "pan" in text or "track" in text:
        return "pan_right"
    if "drift" in text or "diagonal" in text:
        return "drift_diagonal"
    if "push" in text or "zoom" in text or "close" in text:
        return "push_in"
    return ["push_in", "pan_right", "push_in"][index % 3]


class Studio:
    def __init__(self, config: Config, *, projects_root: Path | None = None,
                 cache_root: Path | None = None):
        self.config = config
        paths = config.data.get("paths", {})
        self.projects_root = Path(projects_root or paths.get("projects_dir", "projects"))
        self.cache_root = Path(cache_root or paths.get("cache_dir", "cache"))
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, path: Path | str | None = None, **kwargs: Any) -> "Studio":
        return cls(Config.load(path), **kwargs)

    # ------------------------------------------------------------------
    def plan(self, prompt: str, options: StudioOptions | None = None) -> tuple[PromptPlan, Any]:
        opts = options or StudioOptions()
        hw = detect_hardware(self.projects_root)
        caps = {k: (c.status == "available") for k, c in hw.capabilities.items()}
        plan = understand_prompt(prompt, available=caps,
                                 default_duration=opts.duration_seconds or 30.0)
        self._apply_method_override(plan, opts, hw)
        return plan, hw

    def _apply_method_override(self, plan: PromptPlan, opts: StudioOptions, hw: Any) -> None:
        """Apply an explicit method choice to a parsed plan, honestly.

        ``understand_prompt`` chooses from the prompt text; this layer applies
        the user's Method selector and records any downgrade in the plan itself
        so the browser preview and the final report agree.
        """
        blender_ok = (hw.capabilities.get("blender_3d") is not None and
                      hw.capabilities["blender_3d"].status == "available")
        if opts.method in ("3d", "3d_cartoon", "blender_3d", "blender_3d_cartoon"):
            plan.requested_method = "blender_3d_cartoon"
            plan.supported_method = ("blender_3d_cartoon" if blender_ok
                                     else "conversation_2d")
            plan.method_note = ("Genuine armature-rigged Blender 3D cartoon rendering was "
                                "explicitly selected for this project."
                                if blender_ok else
                                "3D cartoon was explicitly selected, but Blender is "
                                "unavailable; real rigged 2D animation is used and clearly "
                                "labelled as 2D.")
        elif opts.method in ("realistic", "realistic_3d"):
            plan.requested_method = "realistic_human_scene"
            if blender_ok:
                plan.supported_method = "blender_3d_realistic"
                plan.method_note = ("Realistic human 3D was selected. Blender will produce "
                                    "genuine rigged 3D characters with natural proportions, "
                                    "skin/fabric materials and cinematic lighting; this is not "
                                    "photoreal neural video.")
            else:
                from .local_models import check_image_backend
                if check_image_backend().available:
                    plan.supported_method = "ai_still_image_story"
                    plan.method_note = ("Realistic mode uses genuine local AI still images "
                                        "with procedural camera motion, not AI video.")
                else:
                    plan.supported_method = "conversation_2d"
                    plan.method_note = ("No local photorealistic 3D or AI image backend is "
                                        "available; clearly labelled 2D animation is used.")
        elif opts.method in ("cinematic", "photoreal", "still_image"):
            from .local_models import check_image_backend
            plan.requested_method = "cinematic_still_image_story"
            if check_image_backend().available:
                plan.supported_method = "ai_still_image_story"
                plan.method_note = ("A cinematic still-image story will be built from genuine "
                                    "local AI images with procedural camera motion. It is not "
                                    "AI video and is labelled as a still-image story.")
            else:
                plan.supported_method = "animated_story_2d"
                plan.method_note = ("No local AI image backend is available; a clearly labelled "
                                    "procedural still-image/2D story is produced instead. No "
                                    "photoreal imagery is claimed.")
        elif opts.method == "2d":
            plan.requested_method = "animated_story_2d"
            plan.supported_method = "animated_story_2d"
            plan.method_note = "2D animation was explicitly selected for this project."
        elif opts.method == "ai_image":
            plan.requested_method = "ai_image"
            plan.supported_method = "animated_story_2d"
            plan.method_note = ("A local AI image background was explicitly selected; "
                                "animation remains 2D and is labelled accordingly.")
        elif opts.method == "ai_video":
            from .local_models import check_video_backend
            backend = check_video_backend()
            plan.requested_method = "ai_video"
            plan.supported_method = "ai_video" if backend.available else "conversation_2d"
            plan.method_note = ("A genuine local AI video backend was selected."
                                if backend.available else
                                "AI video was selected, but no practical local backend is "
                                "available on this hardware. Real 2D animation is used and "
                                "the report states that no AI video was produced.")
            if not backend.available:
                plan.warnings.append("AI video unavailable: " + backend.detail)
        elif opts.method in ("educational", "children"):
            plan.requested_method = f"{opts.method}_story"
            plan.supported_method = "conversation_2d"
            plan.visual_style = ("children_illustration" if opts.method == "children"
                                 else "educational_diagram")
            plan.transitions = "soft fade" if opts.method == "children" else "cut"
            plan.method_note = (f"{opts.method.title()} mode uses the rigged 2D renderer "
                                "with measured speech and readable pacing; it does not "
                                "claim a 3D or neural render.")
        if (str(plan.supported_method).startswith("blender_3d") and
                not plan.dialogue and plan.characters):
            from .prompt_understanding import _build_scenes, _generate_dialogue
            plan.dialogue = _generate_dialogue(plan.characters, plan.setting or plan.environment_type,
                                               plan.prompt)
            if plan.dialogue:
                plan.scenes = _build_scenes(plan)
                plan.warnings.append("No explicit dialogue was supplied, so short placeholder "
                                     "lines were generated for the 3D character animation "
                                     "(marked source=generated).")

    # ------------------------------------------------------------------
    def run(self, prompt: str, options: StudioOptions | None = None, *,
            progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> StudioResult:
        opts = options or StudioOptions()
        report = progress or (lambda *_: None)

        def step(stage: str, fraction: float, message: str | None = None) -> None:
            if cancel and cancel():
                raise RuntimeError("cancelled")
            report(stage, max(0.0, min(1.0, fraction)), message)

        notes: list[str] = []
        warnings: list[str] = []
        limitations: list[str] = []
        try:
            step("hardware", 0.02, "Detecting hardware and capabilities")
            plan, hw = self.plan(prompt, opts)
            notes.extend(plan.parse_notes)
            warnings.extend(plan.warnings)
            if opts.dialogue:
                plan.dialogue = list(opts.dialogue)
                plan.parse_notes = [n for n in plan.parse_notes
                                    if "No explicit dialogue was supplied" not in n]
            if opts.duration_seconds:
                plan.duration_seconds = opts.duration_seconds
            if opts.aspect_ratio:
                plan.aspect_ratio = opts.aspect_ratio
            width, height = _resolution_dims(opts.resolution, plan.aspect_ratio)

            name = opts.title or plan.title
            if not plan.dialogue:
                notes.append("No dialogue was planned, so the tested narrated-illustration pipeline is used.")
                return self._narrated_fallback(prompt, plan, opts, hw, step, cancel, notes, warnings)

            step("project", 0.05, f"Creating project {name!r}")
            project = Project.create(
                self.projects_root, name=name, prompt=prompt,
                languages={"project": opts.language},
                settings={"studio": opts.to_dict(), "studio_version": 1},
            )
            project.manifest["project_type"] = plan.project_type
            project.manifest["requested_method"] = plan.requested_method
            project.manifest["method_note"] = plan.method_note
            project.save()
            project.write_json("script", "prompt_plan.json", plan.model_dump())
            project.mark_stage(STATUS_SCRIPT_READY)

            # Characters ---------------------------------------------------
            step("characters", 0.09, "Creating persistent character designs")
            lib_root = Path(opts.characters_library) if opts.characters_library else (
                project.path("assets", "characters"))
            library = CharacterLibrary(lib_root)
            characters = ensure_project_characters(library, plan.characters,
                                                   project.manifest["slug"])
            if not characters:
                return self._narrated_fallback(prompt, plan, opts, hw, step, cancel, notes,
                                               warnings, project=project)

            # Dialogue + voices -------------------------------------------
            step("voices", 0.14, "Synthesizing dialogue with local voices")
            convo = render_conversation(
                plan.dialogue, characters, project.stage_dir("narration"),
                language=opts.language,
                config={"wpm": int(self.config.data["defaults"].get("wpm", 130)),
                        "tts_engine": opts.tts_engine, "tts_voice": opts.tts_voice},
                max_chars=int(self.config.data["subtitles"].get("max_line_chars", 46)),
                max_lines=int(self.config.data["subtitles"].get("max_lines", 2)),
            )
            notes.extend(convo.notes)
            if not convo.ok or convo.voice_track is None:
                raise RuntimeError(f"conversation synthesis failed: {convo.error}")
            for a in convo.assignments:
                if not a.distinct:
                    limitations.append(
                        f"Only one distinct installed voice was available for {a.character_name}; "
                        "rate variation was used instead of a different voice.")
            if any(a.engine == "silence" for a in convo.assignments):
                limitations.append("No local TTS voice was installed; the video has silence with "
                                   "accurately timed subtitles.")
            project.write_json("narration", "conversation.json", {
                "lines": [l.to_dict() for l in convo.lines],
                "assignments": [a.to_dict() for a in convo.assignments],
                "total_seconds": convo.total_seconds,
                "notes": convo.notes,
            })
            project.mark_stage(STATUS_NARRATED)

            # Subtitles ----------------------------------------------------
            step("subtitles", 0.28, "Writing subtitles from measured audio")
            assert convo.subtitles is not None
            sub_files = write_subtitles(convo.subtitles, project.stage_dir("subtitles"),
                                        stem="subtitles")
            srt_path = sub_files.get("srt")

            # Background ---------------------------------------------------
            step("background", 0.32, f"Building {plan.environment_type} environment")
            bg_path, bg_method, bg_notes, bg_limits = self._make_background(
                plan, opts, project, width, height, notes, limitations)
            warnings.extend(bg_notes)

            # Render -------------------------------------------------------
            method = self._choose_method(opts, plan, hw)
            if method == "blender_3d":
                step("render3d", 0.4, "Rendering genuine 3D scene with Blender")
                result3d = render_conversation_3d(
                    characters, convo.spans, project.stage_dir("renders"),
                    environment=plan.environment_type, duration=convo.total_seconds,
                    fps=opts.fps, resolution=(width, height),
                    style=("realistic" if plan.supported_method == "blender_3d_realistic"
                           else "cartoon"),
                    progress=lambda i, n: step("render3d", 0.4 + 0.45 * (i / max(1, n)),
                                               f"3D frame {i}/{n}"),
                    cancel=cancel,
                )
                if result3d.ok and result3d.video:
                    base_video = result3d.video
                    method_used = BLENDER_METHOD
                    scene_count = 1
                    notes.extend(result3d.notes)
                else:
                    warnings.append(f"Blender 3D failed ({result3d.error}); using 2D animation.")
                    limitations.append("3D was requested but the Blender render failed; genuine 2D "
                                       "animation was produced instead.")
                    base_video, method_used, scene_count = self._render_2d(
                        convo, characters, bg_path, plan, opts, project, width, height, step, cancel)
                    method = "animated_story_2d"
            else:
                base_video, method_used, scene_count = self._render_2d(
                    convo, characters, bg_path, plan, opts, project, width, height, step, cancel)
                method = "animated_story_2d"

            if method_used != BLENDER_METHOD and plan.requested_method == "blender_3d":
                limitations.append(
                    "3D was requested, but Blender is not installed and no genuine 3D render was "
                    "produced. The exported video is real 2D rigged character animation, not 3D.")

            # Audio mix ----------------------------------------------------
            step("audio", 0.88, "Mixing audio")
            music = opts.music
            if music == "procedural":
                notes.append("Background bed is procedural ambience, not AI-composed music.")
            mix = mix_final_audio(
                convo.voice_track, project.path("audio", "mix.wav"),
                total_seconds=convo.total_seconds, music=music,
                music_volume=opts.music_volume, music_path=self._music_path(music),
                sample_rate=int(self.config.data["render"].get("sample_rate", 44100)),
            )
            if not mix.ok or mix.final_mix_path is None:
                raise RuntimeError(f"audio mix failed: {mix.error}")
            notes.extend(mix.notes or [])

            # Final mux + subtitle burn -----------------------------------
            step("export", 0.92, "Encoding final video")
            out_path = project.path("renders", "final.mp4")
            burn = opts.burn_subtitles and srt_path is not None
            if burn and not has_filter("subtitles"):
                burn = False
                warnings.append("This FFmpeg build lacks libass; subtitles are provided as SRT/VTT "
                                "but were not burned into the picture.")
            self._finalize(base_video, mix.final_mix_path, srt_path if burn else None,
                           out_path, crf=opts.crf, preset=opts.preset)
            duration = mix.total_seconds or wav_duration_seconds(mix.final_mix_path)

            # Thumbnail ----------------------------------------------------
            step("thumbnail", 0.95, "Creating thumbnail")
            thumb = project.path("thumbnails", "thumbnail.jpg")
            try:
                self._thumbnail(out_path, thumb)
            except FFmpegError:
                thumb = None  # type: ignore[assignment]
                warnings.append("Thumbnail extraction failed; the video is still complete.")

            # Validate -----------------------------------------------------
            step("validate", 0.97, "Validating output")
            methods = {
                "animation": method_used,
                "render_style": ("realistic_3d" if method_used == BLENDER_METHOD and
                                 plan.supported_method == "blender_3d_realistic" else
                                 "cartoon_3d" if method_used == BLENDER_METHOD else "2d"),
                "rig": ("armature" if method_used == BLENDER_METHOD else "skeletal_2d"),
                "backgrounds": bg_method,
                "lipsync": "viseme_timeline_2d" if method_used == ANIM_2D_METHOD else "blender_mouth_motion",
                "tts": sorted({l.engine for l in convo.lines}),
                "voices": len({l.voice_id for l in convo.lines if l.voice_id}),
                "music": mix.music_used,
                "subtitles": "libass_burn_in" if burn else "sidecar_files",
            }
            if bg_method.startswith("procedural") and method_used != BLENDER_METHOD:
                limitations.append("Backgrounds are procedurally drawn 2D environments, not AI images.")
            if opts.method == "ai_image" and not bg_method.startswith("ai_"):
                limitations.append("An AI image background was requested but no usable local model was "
                                   "available; procedural environments were rendered instead.")
            if opts.method == "ai_video":
                limitations.append("AI video was requested. No genuine AI video was produced unless "
                                   "the report explicitly names a local video backend.")
            validation = validate_output(
                out_path, expected_width=width, expected_height=height,
                expected_fps=opts.fps, requested_seconds=duration,
                require_audio=True, subtitle_files=[p for p in sub_files.values()],
                expected_scenes=scene_count, rendered_segments=scene_count,
                methods_used=methods, limitations=limitations,
            )
            report_json, report_txt = write_report(validation, project.stage_dir("reports"))

            project.manifest["scenes"] = {"count": scene_count, "method": method_used}
            project.manifest["narration"] = {"lines": len(convo.lines),
                                             "total_seconds": convo.total_seconds}
            project.manifest["render"] = {"width": width, "height": height, "fps": opts.fps,
                                          "segments": scene_count}
            project.manifest["export"] = {
                "path": str(out_path), "resolution": opts.resolution, "fps": opts.fps,
                "duration_seconds": duration, "validated": validation.ok,
            }
            project.manifest["methods_used"] = methods
            project.manifest["capabilities"] = {
                k: c.status for k, c in hw.capabilities.items()}
            for note in notes:
                project.add_warning(note)
            for limitation in limitations:
                project.add_warning(limitation)
            project.mark_stage(STATUS_RENDERED)
            project.mark_stage(STATUS_EXPORTED)

            # Extra honest report -----------------------------------------
            studio_report = {
                "generated_at": time.time(),
                "requested_method": plan.requested_method,
                "supported_method": plan.supported_method,
                "method_used": method_used,
                "method_note": plan.method_note,
                "model_downloads": "none (nothing was downloaded automatically)",
                "hardware": hw.to_dict(),
                "voices": [a.to_dict() for a in convo.assignments],
                "limitations": limitations,
                "notes": notes,
                "warnings": warnings,
                "output": str(out_path),
            }
            sr_path = project.path("reports", "studio_report.json")
            sr_path.write_text(json.dumps(studio_report, ensure_ascii=False, indent=2),
                               encoding="utf-8")

            step("complete", 1.0, "Video ready")
            return StudioResult(
                ok=validation.ok, project=project, output=out_path, thumbnail=thumb,
                method=method_used, requested_method=plan.requested_method, plan=plan,
                conversation=convo, validation=validation,
                report_json=report_json, report_txt=report_txt,
                notes=notes, warnings=warnings, limitations=limitations,
                duration=duration, scene_count=scene_count,
                error=None if validation.ok else "output validation reported problems",
            )
        except Exception as exc:  # noqa: BLE001 — surface the real error
            return StudioResult(False, error=f"{type(exc).__name__}: {exc}",
                                notes=notes, warnings=warnings, limitations=limitations)

    # ------------------------------------------------------------------
    def _music_path(self, music: str) -> Path | None:
        if music in ("none", "procedural", ""):
            return None
        p = Path(music)
        return p if p.exists() else None

    def _make_background(self, plan: PromptPlan, opts: StudioOptions, project: Project,
                         width: int, height: int, notes: list[str],
                         limitations: list[str]) -> tuple[Path, str, list[str], list[str]]:
        warnings: list[str] = []
        custom = project.path("assets", "custom_background.png")
        wants_ai = (opts.method == "ai_image" or
                    plan.supported_method == "ai_still_image_story")
        if custom.exists() and not wants_ai:
            notes.append("Using the custom background saved in this project's assets.")
            return custom, "custom_background", warnings, []
        spec = environment_for_plan(plan, seed=opts.seed or plan.prompt[:80])
        if wants_ai:
            backend = check_image_backend()
            style_word = ("photorealistic, natural skin and fabric" if
                          plan.supported_method == "ai_still_image_story"
                          else f"{plan.visual_style} illustration")
            people = ("" if plan.requested_method == "cinematic_still_image_story"
                      else ", no people")
            prompt = (f"{style_word} of {plan.setting or plan.environment_type}, "
                      f"{plan.time_of_day}, {plan.lighting} lighting, no text{people}")
            gen: GenerationResult | None = None
            if backend.available and backend.backend == "diffusers":
                gen = generate_image_diffusers(
                    prompt, project.path("assets", "background_ai.png"),
                    model_id=opts.image_model, width=min(512, width), height=min(512, height),
                    seed=42)
            elif backend.available and backend.backend == "comfyui":
                warnings.append("ComfyUI is running but no checkpoint name was configured; "
                                "cannot request an AI background.")
            if gen and gen.ok and gen.path:
                notes.append("Background generated locally with an open-weight diffusion model.")
                if str(gen.model_id or "").lower() == "tiny_sd":
                    limitations.append(
                        "Tiny SD is a very small CPU model: prompt adherence and image "
                        "detail are limited compared with larger diffusion models.")
                return gen.path, f"ai_image_local:{gen.model_id}", warnings, limitations
            limitations.append("AI background requested but unavailable: " +
                               ((gen.error if gen else backend.detail) or "no backend"))
            warnings.append("Falling back to a procedural environment.")
        bg = render_cached(spec, self.cache_root / "backgrounds", width=width, height=height)
        saved = project.path("assets", f"background_{spec.env_type}.png")
        try:
            shutil.copyfile(bg, saved)
            bg = saved
        except OSError:
            pass
        return bg, f"procedural_environment:{spec.env_type}", warnings, limitations

    def _choose_method(self, opts: StudioOptions, plan: PromptPlan, hw: Any) -> str:
        if opts.method in ("3d", "3d_cartoon", "blender_3d", "blender_3d_cartoon",
                          "realistic", "realistic_3d"):
            status = find_blender()
            return "blender_3d" if status.available else "animated_story_2d"
        if opts.method in ("2d", "ai_image", "ai_video", "educational", "children"):
            return "animated_story_2d"
        # auto: honour a genuine capability when available
        if str(plan.supported_method).startswith("blender_3d"):
            return "blender_3d"
        return "animated_story_2d"

    def _render_2d(self, convo: ConversationResult, characters: list[Character],
                   bg_path: Path, plan: PromptPlan, opts: StudioOptions, project: Project,
                   width: int, height: int, step: Callable[..., None],
                   cancel: CancelFn | None) -> tuple[Path, str, int]:
        scenes = self._partition(convo, plan, max_scenes=3)
        placements = self._placements(characters)
        segments: list[Path] = []
        for i, (start, end, lines) in enumerate(scenes):
            bg = self._scene_background(plan, opts, project, width, height, i, bg_path)
            anim = self._scene_animation(i, start, end, lines, convo, characters,
                                         placements, bg, plan, opts, width, height)
            seg = project.path("renders", f"scene_{i + 1:02d}.mp4")

            def _prog(done: int, total: int, _i: int = i) -> None:
                frac = 0.36 + 0.5 * ((_i + done / max(1, total)) / max(1, len(scenes)))
                step("render2d", frac, f"Rendering scene {_i + 1}/{len(scenes)}: frame {done}/{total}")

            result = render_scene_animation(anim, seg, crf=opts.crf, preset=opts.preset,
                                            progress=_prog,
                                            cancel=(lambda: bool(cancel and cancel())))
            if not result.ok:
                raise RuntimeError(f"scene {i + 1} render failed: {result.error}")
            segments.append(seg)
        if len(segments) == 1:
            base = segments[0]
        else:
            base = project.path("renders", "animation.mp4")
            concat_segments(segments, base,
                            list_file=project.path("renders", "segments.txt"))
        return base, ANIM_2D_METHOD, len(segments)

    def _scene_background(self, plan: PromptPlan, opts: StudioOptions, project: Project,
                          width: int, height: int, index: int, base_bg: Path) -> Path:
        """Return a deterministic, scene-specific procedural background.

        Scene 1 reuses the project background; later scenes get a different
        deterministic variant of the same environment so cuts are visible while
        staying inside one consistent setting.  AI backgrounds are reused as-is
        (one genuine generated image, not disguised variants).
        """
        ai = project.path("assets", "background_ai.png")
        if ai.exists() and base_bg == ai:
            return ai
        if index == 0:
            first = project.path("assets", f"background_{plan.environment_type}.png")
            return first if first.exists() else base_bg
        saved = project.path("assets", f"background_scene_{index + 1:02d}.png")
        if saved.exists():
            return saved
        spec = environment_for_plan(plan, seed=opts.seed or plan.prompt[:80])
        variant = replace(spec, seed=f"{spec.seed}|scene{index + 1}")
        bg = render_cached(variant, self.cache_root / "backgrounds", width=width, height=height)
        try:
            shutil.copyfile(bg, saved)
            return saved
        except OSError:
            return bg

    def _scene_animation(self, index: int, start: float, end: float, lines: list[Any],
                         convo: ConversationResult, characters: list[Character],
                         placements: list[CharacterPlacement], bg_path: Path,
                         plan: PromptPlan, opts: StudioOptions,
                         width: int, height: int) -> SceneAnimation:
        seg_duration = max(0.8, end - start)
        spans = [
            DialogueSpan(character_id=l.speaker_id, text=l.text,
                         start=max(0.0, l.start - start), end=max(0.1, l.end - start),
                         emotion=l.emotion, action=l.action,
                         visemes=l_visemes(convo, l))
            for l in lines
        ]
        return SceneAnimation(
            background_path=bg_path, placements=placements, dialogue=spans,
            duration=seg_duration, fps=opts.fps, out_w=width, out_h=height,
            camera_move=_camera_move(
                plan.camera_shots[index].movement if index < len(plan.camera_shots) else None,
                index),
            title=None,
            show_table=len(characters) >= 2,
            seed=opts.seed or plan.prompt[:60],
        )

    def regenerate_scene(self, slug: str, index: int,
                         options: StudioOptions | None = None, *,
                         progress: ProgressFn | None = None,
                         cancel: CancelFn | None = None) -> StudioResult:
        """Re-render one scene and re-assemble the final video from all scenes.

        Only the requested segment is encoded again; the other scene files,
        recorded voice track, subtitles and music mix are reused.  This is a
        genuine partial re-render, not a full pipeline re-run.
        """
        report = progress or (lambda *_: None)
        notes: list[str] = []
        warnings: list[str] = []
        try:
            project = resolve_project(self.projects_root, slug)
            plan_data = project.read_json("script", "prompt_plan.json")
            convo_data = project.read_json("narration", "conversation.json")
            if not plan_data or not convo_data:
                raise RuntimeError("project has no studio plan or conversation to re-render")
            plan = PromptPlan.model_validate(plan_data)
            lines: list[RenderedLine] = []
            for d in convo_data.get("lines", []):
                kw = {k: v for k, v in d.items() if k in RenderedLine.__dataclass_fields__}
                kw["wav"] = Path(kw["wav"]) if kw.get("wav") else None
                lines.append(RenderedLine(**kw))
            convo = ConversationResult(
                ok=True, lines=lines,
                total_seconds=float(convo_data.get("total_seconds") or 0.0),
                notes=list(convo_data.get("notes") or []))
            convo.spans = [DialogueSpan(character_id=l.speaker_id, text=l.text,
                                        start=l.start, end=l.end, emotion=l.emotion,
                                        action=l.action,
                                        visemes=l_visemes(convo, l)) for l in lines]
            chars_dir = project.path("assets", "characters")
            characters = (CharacterLibrary(chars_dir).list()
                          if (chars_dir / "library.json").exists() else [])
            if not characters:
                raise RuntimeError("project has no reusable character library")

            settings = dict((project.manifest.get("settings") or {}).get("studio") or {})
            opts = StudioOptions(**{k: v for k, v in settings.items()
                                    if k in StudioOptions.__dataclass_fields__})
            opts.dialogue = None
            if options is not None:
                base = StudioOptions()
                for f in StudioOptions.__dataclass_fields__:
                    if f == "dialogue":
                        continue
                    value = getattr(options, f)
                    if value != getattr(base, f):
                        setattr(opts, f, value)
            width, height = _resolution_dims(opts.resolution, plan.aspect_ratio)
            scenes = self._partition(convo, plan, max_scenes=3)
            if not (1 <= index <= len(scenes)):
                raise ValueError(f"scene {index} does not exist (project has {len(scenes)})")
            start, end, scene_lines = scenes[index - 1]
            placements = self._placements(characters)

            report("prepare", 0.05, f"Preparing scene {index}/{len(scenes)}")
            base_bg = project.path("assets", f"background_{plan.environment_type}.png")
            if not base_bg.exists():
                ai = project.path("assets", "background_ai.png")
                base_bg = ai if ai.exists() else base_bg
            if not base_bg.exists():
                base_bg, _, _, _ = self._make_background(
                    plan, opts, project, width, height, notes, warnings)
            bg = self._scene_background(plan, opts, project, width, height, index - 1, base_bg)
            anim = self._scene_animation(index - 1, start, end, scene_lines, convo,
                                         characters, placements, bg, plan, opts,
                                         width, height)
            seg = project.path("renders", f"scene_{index:02d}.mp4")
            tmp = project.path("renders", f".scene_{index:02d}.tmp.mp4")

            def _prog(done: int, total: int) -> None:
                report("render2d", 0.08 + 0.78 * (done / max(1, total)),
                       f"Re-rendering scene {index}/{len(scenes)}: frame {done}/{total}")

            result = render_scene_animation(
                anim, tmp, crf=opts.crf, preset=opts.preset, progress=_prog,
                cancel=(lambda: bool(cancel and cancel())))
            if not result.ok:
                raise RuntimeError(f"scene {index} render failed: {result.error}")
            tmp.replace(seg)

            segments = sorted(project.path("renders").glob("scene_*.mp4"))
            if not segments:
                raise RuntimeError("no scene segments found after rendering")
            if len(segments) == 1:
                base = segments[0]
            else:
                base = project.path("renders", "animation.mp4")
                concat_segments(segments, base,
                                list_file=project.path("renders", "segments.txt"))

            report("export", 0.9, "Re-assembling the final video with the existing audio")
            audio = project.path("audio", "mix.wav")
            if not audio.exists():
                raise RuntimeError("audio mix is missing; run a full render instead")
            srt = project.path("subtitles", "subtitles.srt")
            out_path = project.path("renders", "final.mp4")
            burn = bool(opts.burn_subtitles and srt.exists() and has_filter("subtitles"))
            self._finalize(base, audio, srt if burn else None, out_path,
                           crf=opts.crf, preset=opts.preset)
            try:
                self._thumbnail(out_path, project.path("thumbnails", "thumbnail.jpg"))
            except FFmpegError:
                warnings.append("Thumbnail refresh failed; the video is still complete.")

            meta = project.manifest.setdefault("scenes", {})
            meta["count"] = len(segments)
            meta["last_regenerated"] = index
            meta["regenerated_at"] = time.time()
            project.add_warning(f"Scene {index} was re-rendered on request; other scene "
                                "segments, voices and subtitles were reused.")
            project.save()
            notes.append(f"Re-rendered scene {index}/{len(segments)} and re-assembled the export.")
            report("complete", 1.0, "Scene updated")
            return StudioResult(
                ok=True, project=project, output=out_path,
                thumbnail=project.path("thumbnails", "thumbnail.jpg"),
                method=project.manifest.get("methods_used", {}).get(
                    "animation", ANIM_2D_METHOD),
                requested_method=project.manifest.get("requested_method", "auto"),
                plan=plan, conversation=convo, duration=wav_duration_seconds(audio),
                scene_count=len(segments), notes=notes, warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001 — surface the real error
            return StudioResult(False, error=f"{type(exc).__name__}: {exc}",
                                notes=notes, warnings=warnings)

    def _placements(self, characters: list[Character]) -> list[CharacterPlacement]:
        n = len(characters)
        out: list[CharacterPlacement] = []
        for i, c in enumerate(characters):
            if n == 1:
                x, facing = 0.5, 1
            else:
                x = 0.32 + 0.36 * i / (n - 1)
                facing = 1 if i == 0 else -1 if i == n - 1 else 1
            out.append(CharacterPlacement(
                rig=CharacterRig.from_character(c), x_frac=x, ground_frac=0.94,
                height_frac=0.58 if n <= 2 else 0.52, facing=facing,
                emphasis=1.08 if i == 0 else 1.0,
            ))
        return out

    def _partition(self, convo: ConversationResult, plan: PromptPlan,
                   max_scenes: int = 3) -> list[tuple[float, float, list[Any]]]:
        lines = convo.lines
        if len(lines) <= 1:
            return [(0.0, convo.total_seconds, lines)]
        n = max(1, min(max_scenes, len(lines)))
        groups: list[list[Any]] = [[] for _ in range(n)]
        for i, line in enumerate(lines):
            groups[min(n - 1, i * n // len(lines))].append(line)
        groups = [g for g in groups if g]
        cuts = [0.0]
        for a, b in zip(groups, groups[1:]):
            cuts.append((a[-1].end + b[0].start) / 2.0)
        cuts.append(convo.total_seconds)
        return [(cuts[i], cuts[i + 1], groups[i]) for i in range(len(groups))]

    def _finalize(self, video: Path, audio: Path, srt: Path | None, out: Path, *,
                  crf: int, preset: str) -> None:
        args = ["-i", str(video), "-i", str(audio)]
        if srt is not None and has_filter("subtitles"):
            args += ["-vf", _subtitle_filter(srt), "-c:v", "libx264", "-preset", preset,
                     "-crf", str(crf), "-pix_fmt", "yuv420p"]
        else:
            args += ["-c:v", "copy"]
        args += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k",
                 "-shortest", "-movflags", "+faststart", "-y", str(out)]
        run_ffmpeg(args, timeout=1800)

    def _thumbnail(self, video: Path, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(["-ss", "1.0", "-i", str(video), "-frames:v", "1", "-q:v", "2",
                    "-y", str(out)], timeout=120)
        return out

    # ------------------------------------------------------------------
    def _narrated_fallback(self, prompt: str, plan: PromptPlan, opts: StudioOptions, hw: Any,
                           step: Callable[..., None], cancel: CancelFn | None,
                           notes: list[str], warnings: list[str],
                           project: Project | None = None) -> StudioResult:
        """Use the existing, tested narrated-illustration pipeline."""
        from .pipeline import Pipeline, PipelineOptions

        if project is None:
            project = Project.create(
                self.projects_root, name=opts.title or plan.title, prompt=prompt,
                languages={"project": opts.language},
                settings={"studio": opts.to_dict(), "studio_version": 1},
            )
            project.write_json("script", "prompt_plan.json", plan.model_dump())
        background_image: Path | None = None
        if plan.supported_method == "ai_still_image_story":
            w, h = RESOLUTIONS.get(opts.resolution, RESOLUTIONS["480p"])
            bg, source, bg_warnings, bg_limits = self._make_background(
                plan, opts, project, w, h, notes, [])
            warnings.extend(bg_warnings)
            if source.startswith("ai_image_local"):
                background_image = bg
                notes.append("The cinematic story uses a locally generated AI still image "
                             "with procedural camera motion; it is not AI video.")
            else:
                warnings.extend(bg_limits)
        pipeline_opts = PipelineOptions.from_config(
            self.config,
            duration_minutes=max(0.05, (opts.duration_seconds or plan.duration_seconds) / 60.0),
            resolution=opts.resolution if opts.resolution in RESOLUTIONS else "720p",
            fps=opts.fps, tts_engine=opts.tts_engine, tts_voice=opts.tts_voice,
            music=opts.music, burn_subtitles=opts.burn_subtitles, crf=opts.crf,
            x264_preset=opts.preset, background_image=background_image,
        )

        def _cb(stage: str, index: int, total: int, extra: dict | None) -> None:
            if cancel and cancel():
                raise RuntimeError("cancelled")
            step(stage, 0.05 + 0.9 * (index / max(1, total)), None)

        result = Pipeline(project, pipeline_opts, progress=_cb).run(resume=False)
        if background_image is not None:
            method = "ai_still_image_story"
        elif result.report:
            method = result.report.methods_used.get("animation", "illustrated_narration")
        else:
            method = "illustrated_narration"
        return StudioResult(
            ok=result.ok, project=project, output=result.output_path,
            method=method,
            requested_method=plan.requested_method, plan=plan, validation=result.report,
            notes=notes + list(result.notes), warnings=warnings,
            limitations=list(result.report.limitations) if result.report else [],
            duration=float(result.report.actual_duration) if result.report else 0.0,
            error=result.error,
        )


def l_visemes(convo: ConversationResult, line: Any) -> list[Any]:
    """Find the precomputed viseme timeline for a rendered line."""
    for span in convo.spans:
        if span.character_id == line.speaker_id and span.text == line.text \
                and abs(span.start - line.start) < 0.02:
            return span.visemes
    from .lipsync import build_viseme_timeline
    return build_viseme_timeline(line.text, max(0.1, line.end - line.start))
