"""FreeAnimate AI — command-line interface and interactive creation flow.

Commands mirror section 15 of the master prompt:

    freeanimate create [--prompt ... --duration ... --language ...]
    freeanimate plan <project>
    freeanimate scenes <project>
    freeanimate narration <project>
    freeanimate generate <project>
    freeanimate assemble <project>
    freeanimate export <project> --resolution 1080p|4k
    freeanimate status <project>
    freeanimate resume <project>
    freeanimate projects list
    freeanimate providers list
    freeanimate doctor
    freeanimate benchmark
    freeanimate clean-cache

The interactive flow guides a beginner through prompt → duration → languages
→ voice → style → resolution → confirmation, with sensible defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import Config, ConfigError
from .constants import (
    APP_NAME,
    APP_VERSION,
    ANIMATION_MODES,
    DEFAULT_DURATION_TOLERANCE,
    DURATION_CHOICES,
    FRAME_RATES,
    MAX_CUSTOM_DURATION_MIN,
    MIN_CUSTOM_DURATION_MIN,
    PIPELINE_STAGES,
    RESOLUTIONS,
)
from .ffmpeg_utils import (
    FFmpegNotFoundError,
    ffmpeg_version,
    ffprobe_version,
    find_ffmpeg,
    has_filter,
)
from .languages import ROLE_NAMES, LanguageRegistry, LanguageSettings
from .pipeline import Pipeline, PipelineOptions, PipelineResult
from .project import (
    Project,
    ProjectError,
    list_projects,
    resolve_project,
)
from .script_engine import extract_topic
from .tts import probe_all_engines, pick_engine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_ROOT / "config" / "config.json"
LANGUAGES_PATH = APP_ROOT / "config" / "languages.json"


def projects_root(cfg: Config) -> Path:
    p = Path(cfg.get("paths.projects_dir", "projects"))
    if not p.is_absolute():
        p = APP_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_root(cfg: Config) -> Path:
    p = Path(cfg.get("paths.cache_dir", "cache"))
    if not p.is_absolute():
        p = APP_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
def _supports_unicode() -> bool:
    return sys.stdout.encoding is None or "utf" in sys.stdout.encoding.lower()


def out(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default or ""
    return val if val else (default or "")


def ask_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            out(f"Please enter a number between {lo} and {hi}.")
        except ValueError:
            out("Please enter a valid number.")


def ask_choice(prompt: str, options: list[tuple[str, str]], default_key: str | None = None) -> str:
    """Show a numbered menu; accept number or key/name."""
    out("")
    out(prompt)
    keys: list[str] = []
    for i, (key, label) in enumerate(options, start=1):
        keys.append(key)
        marker = " (default)" if key == default_key else ""
        out(f"  [{i}] {label}{marker}")
    out("")
    while True:
        raw = ask("Choice", None)
        if not raw and default_key:
            return default_key
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        for key, label in options:
            if raw.lower() == key.lower() or raw.lower() == label.lower():
                return key
        out(f"Please choose 1..{len(options)} or type a name.")


def hr() -> None:
    out("-" * 64)


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------
def make_progress_printer(project_name: str, requested: str, language: str,
                          mode: str, resolution: str):
    state = {"last_line_empty": True}

    def cb(stage: str, index: int, total: int, extra: dict[str, Any] | None) -> None:
        if index == 1 and stage == PIPELINE_STAGES[0]:
            out("")
            out(f"Project: {project_name}")
            out(f"Requested duration: {requested}")
            out(f"Language: {language}")
            out(f"Animation mode: {mode}")
            out(f"Resolution: {resolution}")
            out("")
        prefix = f"[{min(index, total)}/{total}]"
        detail = ""
        if extra:
            if "scene" in extra and total > 1:
                detail = f" — {extra['scene']}"
            if "engine" in extra:
                detail += f" (engine: {extra['engine']})"
            if "camera" in extra:
                detail += f" ({extra['camera']})"
            if "elapsed" in extra and index > 1:
                elapsed = extra["elapsed"]
                detail += f" — elapsed {elapsed/60:.1f} min"
        out(f"{prefix} {stage}{detail}")

    return cb


# ---------------------------------------------------------------------------
# Command: create
# ---------------------------------------------------------------------------
def cmd_create(args: argparse.Namespace) -> int:
    cfg = Config.load(CONFIG_PATH)
    langs = LanguageRegistry.from_file(LANGUAGES_PATH if LANGUAGES_PATH.exists() else None)
    if not LANGUAGES_PATH.exists():
        langs.save(LANGUAGES_PATH)

    interactive = not args.prompt or args.interactive

    # ---- 1. Prompt
    prompt = args.prompt
    if not prompt:
        out(f"{APP_NAME} — CREATE VIDEO")
        hr()
        prompt = ask("Enter your video prompt")
        while not prompt:
            prompt = ask("A prompt is required. Enter your video prompt")
    topic = extract_topic(prompt)

    # ---- 2. Duration
    duration_minutes: float | None = None
    if args.duration is not None:
        duration_minutes = float(args.duration)
        if duration_minutes <= 0:
            out("Invalid duration; must be a positive number of minutes.")
            return 2
    elif interactive:
        out("")
        out("Choose video duration:")
        opts = [(str(k), f"{v} minutes" if v else "Custom duration")
                for k, v in DURATION_CHOICES.items()]
        choice = ask_choice("Duration option", opts, default_key="2")
        if choice == "9":
            custom = ask_int(f"Custom duration in minutes ({MIN_CUSTOM_DURATION_MIN}-{MAX_CUSTOM_DURATION_MIN})",
                             10, MIN_CUSTOM_DURATION_MIN, MAX_CUSTOM_DURATION_MIN)
            duration_minutes = float(custom)
        else:
            duration_minutes = float(DURATION_CHOICES[int(choice)])

    # ---- 3-7. Languages
    lang_args = {
        "project": args.language,
        "script": args.script_language,
        "narration": args.narration_language,
        "subtitle": args.subtitle_language,
        "onscreen": args.onscreen_language,
        "thumbnail": getattr(args, "thumbnail_language", None),
    }
    missing = [r for r, v in lang_args.items() if not v]
    if interactive and ("project" in missing or len(missing) > 1):
        if not lang_args["project"]:
            lang_args["project"] = ask_choice(
                "Project language (used for everything by default)",
                [(l.code, l.label()) for l in langs],
                default_key=cfg.get("defaults.project_language", "en"),
            )
        # Offer per-role overrides.
        override = ask_choice(
            "Language settings",
            [
                ("all", "Use project language for everything (recommended)"),
                ("custom", "Set script / narration / subtitle / on-screen / thumbnail languages separately"),
            ],
            default_key="all",
        )
        if override == "custom":
            for role in ("script", "narration", "subtitle", "onscreen", "thumbnail"):
                lang_args[role] = ask_choice(
                    f"{role.capitalize()} language",
                    [(l.code, l.label()) for l in langs],
                    default_key=lang_args["project"],
                )
    if not lang_args["project"]:
        lang_args["project"] = cfg.get("defaults.project_language", "en")
    try:
        project_lang = langs.resolve(lang_args["project"])
    except KeyError as exc:
        out(f"Error: {exc}")
        out("Run 'freeanimate languages' to see supported languages.")
        return 2
    lang_args["project"] = project_lang.code
    for role in ("script", "narration", "subtitle", "onscreen", "thumbnail"):
        if lang_args[role]:
            try:
                lang_args[role] = langs.resolve(lang_args[role]).code
            except KeyError as exc:
                out(f"Error: {exc}")
                return 2
    lang_settings = LanguageSettings.from_dict(lang_args)

    # ---- 8. Voice selection (only after narration language chosen)
    narration_lang = lang_settings.resolved("narration")
    tts_choice = args.voice or cfg.get("tts.engine", "auto")
    if interactive and not args.non_interactive_defaults:
        reports = {r.engine: r for r in probe_all_engines()}
        usable = [r for r in reports.values() if r.available and r.engine != "silence"]
        out("")
        out(f"Narration language: {langs.require(narration_lang).label()}")
        if usable:
            voice_opts: list[tuple[str, str]] = []
            for r in usable:
                match_count = len([v for v in r.voices if v.language.split("-")[0].lower() == narration_lang])
                label = f"Use {r.engine} — {r.detail}"
                if r.voices:
                    label += f" ({len(r.voices)} voice(s), {match_count} match language)"
                voice_opts.append((r.engine, label))
            voice_opts.append(("silence", "Continue without narration (silent track)"))
            tts_choice = ask_choice("Voice / engine", voice_opts, default_key=voice_opts[0][0])
        else:
            out("No TTS engines with voices are installed. Install options:")
            out("  - Piper: https://github.com/rhasspy/piper (offline neural TTS)")
            out("  - espeak-ng: https://github.com/espeak-ng/espeak-ng")
            out("  - Windows voices: Settings > Time & Language > Speech")
            tts_choice = ask_choice(
                "Narration",
                [("silence", "Continue without narration (silent track)")],
                default_key="silence",
            )

    # ---- 9. Animation style
    mode = args.animation_mode or cfg.get("defaults.animation_mode", "hybrid")
    if interactive and not args.animation_mode:
        mode = ask_choice(
            "Animation style",
            [(k, f"{k}: {v}") for k, v in ANIMATION_MODES.items()],
            default_key=mode,
        )

    # ---- 10. Resolution
    resolution = args.resolution or cfg.get("defaults.resolution", "1080p")
    if interactive and not args.resolution:
        res_opts = [("1080p", "1080p (1920x1080) — recommended"),
                    ("4k", "4K (3840x2160) — slower on low-end hardware"),
                    ("720p", "720p (1280x720) — fastest preview")]
        resolution = ask_choice("Export resolution", res_opts, default_key=resolution)

    # ---- 11. Generation mode extras
    burn = cfg.get("defaults.burn_subtitles", True)
    music = cfg.get("defaults.music", "none")
    if interactive and not args.no_prompts_extra:
        burn = ask_choice("Burn subtitles into the video?",
                          [("yes", "Yes (burned-in + .srt/.vtt files)"),
                           ("no", "No (subtitle files only)")],
                          default_key="yes" if burn else "no") == "yes"
        music = ask_choice(
            "Background music",
            [("none", "No music (silent background)"),
             ("procedural", "Procedural ambient pad (synthesized, free, labeled as ambience)")],
            default_key="none",
        )

    # Explicit flags win over interactive answers / config defaults.
    if args.music:
        music = args.music
    if getattr(args, "no_burn_subtitles", False):
        burn = False

    # ---- 12. Confirmation
    name = args.name or topic[:60]
    hr()
    out("Summary")
    out(f"  Name:        {name}")
    out(f"  Prompt:      {prompt[:100]}{'…' if len(prompt) > 100 else ''}")
    out(f"  Duration:    {duration_minutes} minutes")
    out(f"  Project lang:{lang_settings.project}")
    out(f"  Script lang: {lang_settings.resolved('script')}")
    out(f"  Narration:   {lang_settings.resolved('narration')} via {tts_choice}")
    out(f"  Subtitles:   {lang_settings.resolved('subtitle')} (burned: {burn})")
    out(f"  On-screen:   {lang_settings.resolved('onscreen')}")
    out(f"  Animation:   {mode}")
    out(f"  Resolution:  {resolution}")
    out(f"  Music:       {music}")
    hr()
    if interactive and not args.yes:
        if ask("Proceed? (yes/no)", "yes").lower() not in ("y", "yes"):
            out("Cancelled.")
            return 1

    # ---- Run
    settings = {
        "duration_minutes": duration_minutes,
        "resolution": resolution,
        "fps": args.fps or cfg.get("defaults.fps"),
        "animation_mode": mode,
        "transition": cfg.get("defaults.transition", "fade"),
        "wpm": cfg.get("defaults.wpm"),
        "burn_subtitles": bool(burn),
        "music": music,
        "tts_engine": tts_choice,
        "low_resource": bool(args.low_resource or cfg.get("defaults.low_resource")),
    }
    project = Project.create(projects_root(cfg), name, prompt, lang_settings.as_dict(), settings)
    out(f"Created project: {project.root}")

    opts = PipelineOptions.from_config(
        cfg,
        duration_minutes=duration_minutes,
        resolution=resolution,
        animation_mode=mode,
        burn_subtitles=bool(burn),
        music=music,
        tts_engine=tts_choice,
        low_resource=settings["low_resource"],
        script_file=args.script_file,
        fps=args.fps,
        thumbnails=not getattr(args, "no_thumbnails", False),
        auto_select_thumbnail=bool(getattr(args, "auto_select_thumbnail", False)),
    )
    progress = make_progress_printer(name, f"{duration_minutes} minutes",
                                     lang_settings.project, mode, resolution)
    result = Pipeline(project, opts, progress=progress, languages=langs).run(resume=False)
    return _report_result(result)


def _report_result(result: PipelineResult) -> int:
    out("")
    if result.ok and result.report and result.output_path:
        r = result.report
        out("EXPORT COMPLETE")
        out(f"  Output:     {result.output_path}")
        out(f"  Duration:   {r.actual_duration:.1f}s requested {r.requested_duration:.1f}s" if r.actual_duration else "  Duration: unknown")
        out(f"  Resolution: {r.resolution}")
        out(f"  Frame rate: {r.frame_rate}")
        out(f"  Audio:      {'present' if r.audio_present else 'missing'}")
        out(f"  Subtitles:  {len(r.subtitle_files)} file(s)")
        out(f"  Methods:    {', '.join(r.methods_used.get('animation_methods', []))}")
        if r.methods_used.get("music") and r.methods_used["music"] != "none":
            out(f"  Music:      {r.methods_used['music']}")
        if r.limitations:
            out("  Notes/limitations:")
            for lim in r.limitations[:8]:
                out(f"    - {lim}")
        out("")
        out(r.summary())
        out("")
        out("VIDEO GENERATION COMPLETE")
        out("")
        out("Video:")
        out(f"  {result.output_path}")
        out("")
        _print_thumbnail_report(result.project)
        return 0
    out("EXPORT FAILED")
    if result.error:
        out(f"  Error: {result.error}")
    for n in result.notes[:5]:
        out(f"  Note: {n}")
    out(f"  Project saved at: {result.project.root} (use 'freeanimate resume {result.project.manifest['slug']}' to retry)")
    return 1


def _sync_thumbnail_manifest(project, report: dict) -> None:
    """Refresh the manifest's thumbnails block from a fresh report.

    Keeps `status` consistent after standalone generate/regenerate runs (the
    pipeline does the same when it runs the stage itself).
    """
    block = project.manifest.setdefault("thumbnails", {})
    block["recommended"] = report["recommendation"]["index"]
    block["reasoning"] = report["recommendation"]["reasoning"]
    block["options"] = [
        {k: o.get(k) for k in ("index", "file", "concept", "concept_label",
                               "total", "passed_quality")}
        for o in report["options"]
    ]
    block["report"] = "thumbnails/thumbnail_recommendations.json"
    block["generated_at"] = report["generated_at"]
    if report.get("selected"):
        block["selected"] = report["selected"].get("index")
    project.save()


def _print_thumbnail_report(project) -> None:
    """Print the thumbnail options + recommendation (section 12.7)."""
    from .thumbnails import read_recommendation_json, SELECTED_NAME

    rep = read_recommendation_json(project)
    if not rep:
        return
    out("")
    out("THUMBNAIL OPTIONS")
    out("")
    rec_idx = (rep.get("recommendation") or {}).get("index")
    for o in rep.get("options", []):
        idx = o.get("index")
        tag = "Recommended — " if idx == rec_idx else ""
        out(f"{idx}. {tag}{o.get('concept_label', o.get('concept'))}")
        out(f"   Score: {o.get('total', 0)}/100")
        strengths = ", ".join(o.get("strengths", [])) or "—"
        weaknesses = ", ".join(o.get("weaknesses", [])) or "—"
        out(f"   Strengths: {strengths}")
        out(f"   Weaknesses: {weaknesses}")
        out(f"   File: thumbnails/{o.get('file', '')}"
            f"{' (failed quality checks)' if not o.get('passed_quality') else ''}")
        out("")
    out("RECOMMENDATION")
    out("")
    reasoning = (rep.get("recommendation") or {}).get("reasoning", "")
    if rec_idx:
        out(f"Choose Thumbnail {rec_idx}: {reasoning}")
    else:
        out(f"No recommendation available: {reasoning}")
    out("")
    out(f"Thumbnail language: {rep.get('thumbnail_language', '?')} | "
        f"Method: {rep.get('generation_method', '?')}")
    selected = rep.get("selected")
    if selected:
        out(f"Selected thumbnail: #{selected.get('index')} -> thumbnails/{SELECTED_NAME}")
    for w in rep.get("warnings", [])[:6]:
        out(f"  warning: {w}")
    slug = project.manifest.get("slug", "")
    out("")
    out(f"Manage thumbnails: freeanimate thumbnails {slug}")
    out("  (view, regenerate, select, open folder — without re-rendering the video)")


# ---------------------------------------------------------------------------
# Command: status / plan / scenes / narration
# ---------------------------------------------------------------------------
def _load_project(cfg: Config, ref: str) -> Project:
    return resolve_project(projects_root(cfg), ref)


def cmd_status(args: argparse.Namespace) -> int:
    cfg = Config.load(CONFIG_PATH)
    try:
        p = _load_project(cfg, args.project)
    except ProjectError as exc:
        out(str(exc))
        return 2
    m = p.manifest
    hr()
    out(f"Project:  {m.get('name')}")
    out(f"Slug:     {m.get('slug')}")
    out(f"Status:   {m.get('status')}")
    out(f"Resume:   stage '{p.resume_point()}'")
    out(f"Prompt:   {(m.get('prompt') or '')[:120]}")
    langs = m.get("languages", {})
    out(f"Languages: project={langs.get('project')} script={langs.get('script')} "
        f"narration={langs.get('narration')} subtitle={langs.get('subtitle')} onscreen={langs.get('onscreen')}")
    settings = m.get("settings", {})
    out(f"Duration: {settings.get('duration_minutes')} min | Resolution: {settings.get('resolution')} | "
        f"Mode: {settings.get('animation_mode')}")
    out("Stages:")
    for stage, info in m.get("stages", {}).items():
        mark = "done" if info.get("done") else ("error" if info.get("error") else "pending")
        err = f" — {info['error']}" if info.get("error") else ""
        out(f"  {stage:<14} {mark}{err}")
    if m.get("scenes"):
        out(f"Scenes:   {m['scenes']}")
    if m.get("narration"):
        n = m["narration"]
        out(f"Narration: engine={n.get('engine')} total={n.get('total_seconds', 0):.1f}s")
    if m.get("export"):
        e = m["export"]
        out(f"Export:   {e.get('path')} (validated={e.get('validated')})")
    if m.get("thumbnails"):
        t = m["thumbnails"]
        selected = t.get("selected")
        if not selected:
            # Fall back to the report JSON (selections made before the manifest
            # was kept in sync, or regenerated reports carrying 'selected').
            try:
                from .thumbnails import read_recommendation_json
                rep = read_recommendation_json(p)
                selected = (rep or {}).get("selected", {}).get("index")
            except Exception:  # noqa: BLE001
                selected = None
        out(f"Thumbnails: recommended=#{t.get('recommended')} | "
            f"{len(t.get('options', []))} option(s) | selected=#{selected or 'none'}")
    elif m.get("stages", {}).get("thumbnails_ready", {}).get("error"):
        out(f"Thumbnails: FAILED — {m['stages']['thumbnails_ready']['error']}")
    if m.get("errors"):
        out("Recent errors:")
        for e in m["errors"][-5:]:
            out(f"  - {e.get('message')}")
    if m.get("warnings"):
        out("Warnings:")
        for w_ in m["warnings"][-5:]:
            out(f"  - {w_.get('message')}")
    hr()
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Show/regenerate the script outline without rendering."""
    cfg = Config.load(CONFIG_PATH)
    langs = LanguageRegistry.from_file(LANGUAGES_PATH if LANGUAGES_PATH.exists() else None)
    try:
        p = _load_project(cfg, args.project)
    except ProjectError as exc:
        out(str(exc))
        return 2
    script_data = p.read_json("script", "script.json")
    if script_data is None:
        opts = PipelineOptions.from_config(cfg)
        pipe = Pipeline(p, opts, languages=langs)
        script = pipe._prepare_script(LanguageSettings.from_dict(p.manifest.get("languages", {})))
        p.write_json("script", "script.json", script.to_dict())
        p.mark_stage("script_ready")
    else:
        from .script_engine import Script
        script = Script.from_dict(script_data)
    out(f"Script: '{script.topic}' ({script.kind}) — {script.actual_words} words, "
        f"~{script.estimated_seconds / 60:.1f} min @ {script.target_wpm} wpm")
    out(f"Source: {script.script_source}")
    for sec in script.sections:
        out(f"  [{sec.id}] {sec.title} ({sec.word_count} words)")
        preview = sec.narration[:120].replace("\n", " ")
        out(f"        {preview}…")
    out(f"\nFull script: {p.root / 'script' / 'full_script.txt'}")
    out(f"Edit section files, then run: freeanimate generate {p.manifest['slug']}")
    return 0


def cmd_scenes(args: argparse.Namespace) -> int:
    cfg = Config.load(CONFIG_PATH)
    try:
        p = _load_project(cfg, args.project)
    except ProjectError as exc:
        out(str(exc))
        return 2
    data = p.read_json("scenes", "scenes.json")
    if data is None:
        out("No scene manifest yet. Run: freeanimate plan <project>")
        return 1
    out(f"Scenes: {len(data['scenes'])} | estimated {data['total_estimated_seconds'] / 60:.1f} min "
        f"| target {data['target_seconds'] / 60:.1f} min")
    for s in data["scenes"]:
        out(f"  {s['scene_id']}  {s['duration_seconds']:5.1f}s  [{s['status']:<13}] "
            f"{s['camera_move']:<14} {s['style']:<16} {s['title'][:40]}")
    return 0


def cmd_narration(args: argparse.Namespace) -> int:
    """Generate (or regenerate) narration only."""
    cfg = Config.load(CONFIG_PATH)
    langs = LanguageRegistry.from_file(LANGUAGES_PATH if LANGUAGES_PATH.exists() else None)
    try:
        p = _load_project(cfg, args.project)
    except ProjectError as exc:
        out(str(exc))
        return 2
    data = p.read_json("scenes", "scenes.json")
    if data is None:
        out("No scene manifest yet. Run: freeanimate plan <project>")
        return 1
    from .scene_planner import SceneManifest
    manifest = SceneManifest.from_dict(data)
    opts = PipelineOptions.from_config(cfg)
    if args.engine:
        opts.tts_engine = args.engine
    pipe = Pipeline(p, opts, languages=langs)
    lang_settings = LanguageSettings.from_dict(p.manifest.get("languages", {}))
    info = pipe._generate_narration(manifest, lang_settings)
    out(f"Narration engine: {info['engine']}")
    out(f"Total narration: {info['total_seconds']:.1f}s across {len(manifest.scenes)} scenes")
    for n in info.get("notes", []):
        out(f"  note: {n}")
    return 0


# ---------------------------------------------------------------------------
# Command: generate (full pipeline, resumes)
# ---------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> int:
    cfg = Config.load(CONFIG_PATH)
    langs = LanguageRegistry.from_file(LANGUAGES_PATH if LANGUAGES_PATH.exists() else None)
    try:
        p = _load_project(cfg, args.project)
    except ProjectError as exc:
        out(str(exc))
        return 2
    settings = p.manifest.get("settings", {})
    overrides: dict[str, Any] = {}
    if args.resolution:
        overrides["resolution"] = args.resolution
    if args.low_resource:
        overrides["low_resource"] = True
    opts = PipelineOptions.from_config(
        cfg,
        duration_minutes=settings.get("duration_minutes"),
        resolution=settings.get("resolution"),
        fps=settings.get("fps"),
        animation_mode=settings.get("animation_mode"),
        burn_subtitles=settings.get("burn_subtitles"),
        music=settings.get("music"),
        tts_engine=settings.get("tts_engine", cfg.get("tts.engine")),
        low_resource=settings.get("low_resource", False),
        **overrides,
    )
    progress = make_progress_printer(
        p.manifest.get("name", "?"),
        f"{opts.duration_minutes} minutes",
        p.manifest.get("languages", {}).get("project", "en"),
        opts.animation_mode,
        opts.resolution,
    )
    result = Pipeline(p, opts, progress=progress, languages=langs).run(resume=not args.fresh)
    return _report_result(result)


def cmd_resume(args: argparse.Namespace) -> int:
    args.resolution = None
    args.low_resource = args.low_resource if hasattr(args, "low_resource") else False
    args.fresh = False
    return cmd_generate(args)


def cmd_export(args: argparse.Namespace) -> int:
    args.resolution = args.resolution
    return cmd_generate(args)


# ---------------------------------------------------------------------------
# Command: thumbnails (view / generate / regenerate / select / open)
# ---------------------------------------------------------------------------
def cmd_thumbnails(args: argparse.Namespace) -> int:
    """Manage YouTube thumbnails for a project without re-rendering (12.8)."""
    cfg = Config.load(CONFIG_PATH)
    try:
        p = _load_project(cfg, args.project)
    except ProjectError as exc:
        out(str(exc))
        return 2
    action = getattr(args, "action", None) or "show"
    index = getattr(args, "index", None)

    from .thumbnails import (
        SELECTED_NAME,
        open_thumbnails_folder,
        read_recommendation_json,
        regenerate_thumbnail,
        select_thumbnail,
    )

    if action == "open":
        open_thumbnails_folder(p)
        out(f"Opened: {p.stage_dir('thumbnails')}")
        return 0

    if action in ("generate", "regenerate"):
        if not (p.manifest.get("export") or p.stage_done("exported")):
            out("This project has no exported video yet; render it first:")
            out(f"  freeanimate generate {p.manifest['slug']}")
            return 1
        what = f"thumbnail #{index}" if index else "all three thumbnails"
        out(f"Regenerating {what} from existing scene assets (the video is NOT re-rendered)...")
        try:
            report = regenerate_thumbnail(
                p, index=index,
                text=getattr(args, "text", None) or None,
                lang_code=getattr(args, "language", None) or None,
            )
        except Exception as exc:  # noqa: BLE001 — reported honestly
            out(f"Thumbnail generation failed: {exc}")
            return 1
        _sync_thumbnail_manifest(p, report)
        out("Current options:")
        for o in report["options"]:
            out(f"  {o['index']}. {o['concept_label']:<28} {o['total']:>3}/100  "
                f"{'quality OK' if o['passed_quality'] else 'FAILED quality checks'}")
        rec = report["recommendation"]["index"]
        out("")
        out(f"Recommendation: {'Thumbnail ' + str(rec) if rec else 'none (all failed quality checks)'}")
        for w in report.get("warnings", [])[:5]:
            out(f"  warning: {w}")
        return 0 if rec else 1

    if action == "select":
        rep = read_recommendation_json(p)
        if not rep:
            out("No thumbnails exist yet for this project. Generate them first:")
            out(f"  freeanimate thumbnails generate {p.manifest['slug']}")
            return 1
        choice: Any = index if index else "accept"
        try:
            sel = select_thumbnail(p, choice,
                                   overwrite_confirmed=bool(getattr(args, "yes", False)))
        except FileExistsError as exc:
            out(str(exc))
            out(f"Re-run with --yes to replace it: freeanimate thumbnails select "
                f"{p.manifest['slug']} --index {index} --yes")
            return 1
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            out(f"Error: {exc}")
            return 1
        out(f"Selected thumbnail #{sel['index']} -> {sel['path']}")
        out("(The file was copied to thumbnails/" + SELECTED_NAME + "; the previously "
            "selected thumbnail is only replaced with confirmation.)")
        # Keep the manifest in sync so 'freeanimate status' shows the selection.
        p.manifest.setdefault("thumbnails", {})["selected"] = sel["index"]
        p.save()
        return 0

    # default: show
    rep = read_recommendation_json(p)
    if not rep:
        if p.manifest.get("export"):
            out("No thumbnails yet for this completed video. Generate them with:")
            out(f"  freeanimate thumbnails generate {p.manifest['slug']}")
        else:
            out("This project has no exported video yet, so no thumbnails exist.")
            out(f"Render it first: freeanimate generate {p.manifest['slug']}")
        return 1
    _print_thumbnail_report(p)
    return 0


# ---------------------------------------------------------------------------
# Command: projects list
# ---------------------------------------------------------------------------
def cmd_projects(args: argparse.Namespace) -> int:
    cfg = Config.load(CONFIG_PATH)
    items = list_projects(projects_root(cfg))
    if not items:
        out("No projects yet. Create one with: freeanimate create")
        return 0
    out(f"{'SLUG':<28} {'STATUS':<14} {'NAME'}")
    hr()
    for it in items:
        out(f"{it['slug']:<28} {it['status']:<14} {it.get('name', '')[:40]}")
    return 0


# ---------------------------------------------------------------------------
# Command: providers list
# ---------------------------------------------------------------------------
def cmd_providers(args: argparse.Namespace) -> int:
    out(f"{APP_NAME} — PROVIDER AVAILABILITY")
    hr()
    out("Text-to-speech engines (probed at runtime):")
    for r in probe_all_engines():
        mark = "AVAILABLE" if r.available else "unavailable"
        out(f"  [{mark:<11}] {r.engine:<8} {r.detail}")
        for v in r.voices[:6]:
            out(f"      voice: {v.name} ({v.language}) offline={v.offline}")
        if len(r.voices) > 6:
            out(f"      … {len(r.voices) - 6} more voices")
        if r.extra.get("licensing"):
            out(f"      license: {r.extra['licensing']}")
    hr()
    out("FFmpeg:")
    out(f"  ffmpeg:  {ffmpeg_version() or 'NOT FOUND'}")
    out(f"  ffprobe: {ffprobe_version() or 'NOT FOUND'}")
    if find_ffmpeg("ffmpeg"):
        for f in ("zoompan", "subtitles", "drawtext", "afade", "loudnorm", "sidechaincompress"):
            out(f"  filter {f}: {'yes' if has_filter(f) else 'NO'}")
    hr()
    out("Visual generation:")
    out("  [AVAILABLE ] procedural illustrations + motion graphics (offline, CPU-only)")
    out("  [unavailable] AI video backends (none installed; core pipeline does not require them)")
    hr()
    out("Script generation:")
    out("  [AVAILABLE ] template engine (offline, deterministic)")
    ollama = _probe_ollama()
    out(f"  [{'AVAILABLE' if ollama else 'unavailable':<11}] ollama local LLM {'(' + ollama + ')' if ollama else '(not running)'}")
    hr()
    out("Translation:")
    out("  [unavailable] none configured (enable mymemory in config; requires network + consent)")
    return 0


def _probe_ollama() -> str | None:
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1) as r:
            data = json.loads(r.read().decode())
            models = [m.get("name", "?") for m in data.get("models", [])]
            return ", ".join(models[:3]) if models else "running (no models)"
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Command: doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    out(f"{APP_NAME} {APP_VERSION} — DOCTOR")
    hr()
    problems = 0

    # Python
    py = sys.version.split()[0]
    out(f"Python:    {py} ({sys.executable})")
    if sys.version_info < (3, 10):
        out("  ERROR: Python 3.10+ required")
        problems += 1

    # FFmpeg
    fv = ffmpeg_version()
    out(f"FFmpeg:    {fv or 'NOT FOUND'}")
    if not fv:
        out("  ERROR: FFmpeg not found. Install: winget install Gyan.FFmpeg")
        problems += 1
    else:
        for f in ("zoompan", "subtitles"):
            ok = has_filter(f)
            out(f"  filter {f}: {'ok' if ok else 'MISSING'}")
            if not ok:
                problems += 1

    # PIL / numpy
    try:
        import PIL
        out(f"Pillow:    {PIL.__version__}")
    except ImportError:
        out("Pillow:    NOT INSTALLED (pip install Pillow)")
        problems += 1
    try:
        import numpy
        out(f"numpy:     {numpy.__version__}")
    except ImportError:
        out("numpy:     not installed (needed for procedural ambience only)")
    try:
        import pyttsx3
        out("pyttsx3:   installed")
    except ImportError:
        out("pyttsx3:   not installed (optional; enables Windows SAPI narration)")

    # Disk space
    try:
        import shutil as _sh
        root = APP_ROOT if str(APP_ROOT)[1:3] == ":\\" else Path(".")
        usage = _sh.disk_usage(str(root))
        out(f"Disk free: {usage.free / 1e9:.1f} GB of {usage.total / 1e9:.1f} GB")
        if usage.free < 2e9:
            out("  WARNING: less than 2 GB free; long 4K projects may fail")
            problems += 1
    except OSError:
        pass

    # TTS
    out("TTS engines:")
    any_tts = False
    for r in probe_all_engines():
        if r.engine == "silence":
            continue
        mark = "ok" if r.available else "—"
        out(f"  {r.engine:<8} {mark:<4} {r.detail[:70]}")
        any_tts = any_tts or r.available
    if not any_tts:
        out("  WARNING: no speech engines found; narration will be silent silence")
        out("  Install espeak-ng or Piper for real narration.")

    # Fonts per script family
    from .fonts import find_font_for_script
    out("Fonts:")
    for family in ("latin", "arabic", "devanagari", "cjk", "cyrillic"):
        res = find_font_for_script(family)
        if res.ok:
            out(f"  {family:<11} {res.font.filename}")
        else:
            out(f"  {family:<11} MISSING — {res.guidance[:60]}")

    hr()
    if problems == 0:
        out("All critical checks passed. Ready to create videos.")
        return 0
    out(f"{problems} issue(s) found. Fix the ERROR lines above before long renders.")
    return 1


# ---------------------------------------------------------------------------
# Command: benchmark
# ---------------------------------------------------------------------------
def cmd_benchmark(args: argparse.Namespace) -> int:
    """Measure real render throughput with a tiny synthetic scene."""
    import tempfile
    cfg = Config.load(CONFIG_PATH)
    out(f"{APP_NAME} — BENCHMARK")
    hr()
    if not find_ffmpeg("ffmpeg"):
        out("FFmpeg not found; cannot benchmark.")
        return 1

    from .scene_planner import Scene
    from .visuals import generate_scene_asset, MotionSpec
    from .renderer import render_scene_segment

    with tempfile.TemporaryDirectory(prefix="freeanimate_bench_") as td:
        tdir = Path(td)
        scene = Scene(scene_id="bench_001", title="Benchmark", narration_text="Benchmark narration.",
                      duration_seconds=5.0, visual_description="benchmark card", camera_move="push_in",
                      style="motion_graphics")
        t0 = time.time()
        res = generate_scene_asset(scene, tdir, width=2560, height=1440, fps=24)
        t_asset = time.time() - t0
        if not res.ok:
            out(f"Asset generation failed: {res.error}")
            return 1
        motion = MotionSpec(camera_move="push_in", duration_seconds=5.0, fps=24,
                            input_w=2560, input_h=1440, out_w=1920, out_h=1080)
        t0 = time.time()
        r = render_scene_segment(res.image_path, tdir / "bench_seg.mp4", motion, preset="veryfast")
        t_render = time.time() - t0
        if not r.ok:
            out(f"Render failed: {r.error}")
            return 1
        seg_size = (tdir / "bench_seg.mp4").stat().st_size
        out(f"Asset generation (2560x1440 PNG): {t_asset:.2f}s")
        out(f"Scene render (5s @ 24fps, 1080p): {t_render:.2f}s")
        out(f"Realtime factor: {5.0 / t_render:.2f}x" if t_render > 0 else "")
        out(f"Segment size: {seg_size / 1e6:.2f} MB")
        for minutes in (5, 10, 30, 45):
            scenes = max(4, int(minutes * 60 / 22))
            est = scenes * t_render / 60.0
            out(f"Estimated render time for {minutes:>2} min video (~{scenes} scenes): {est:.1f} min "
                f"(excl. narration/assembly)")
    hr()
    out("Estimates are extrapolations from this machine's measured throughput.")
    return 0


# ---------------------------------------------------------------------------
# Command: clean-cache
# ---------------------------------------------------------------------------
def cmd_clean_cache(args: argparse.Namespace) -> int:
    cfg = Config.load(CONFIG_PATH)
    croot = cache_root(cfg)
    total = 0
    files = 0
    for p in croot.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            files += 1
    out(f"Cache: {croot}")
    out(f"  {files} file(s), {total / 1e6:.1f} MB")
    if args.yes or ask("Delete cache files? (yes/no)", "no").lower() in ("y", "yes"):
        import shutil
        for p in croot.rglob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
        out("Cache cleaned.")
    else:
        out("Cancelled.")
    return 0


# ---------------------------------------------------------------------------
# Command: languages
# ---------------------------------------------------------------------------
def cmd_languages(args: argparse.Namespace) -> int:
    langs = LanguageRegistry.from_file(LANGUAGES_PATH if LANGUAGES_PATH.exists() else None)
    out("Supported languages (config/languages.json is editable):")
    for line in langs.menu_lines():
        out("  " + line)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freeanimate",
        description=f"{APP_NAME} {APP_VERSION} — free, multilingual, terminal-based AI video creator",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    sub = parser.add_subparsers(dest="command", help="commands")

    # create
    c = sub.add_parser("create", help="create a new video project (interactive by default)")
    c.add_argument("--prompt", "-p", help="video prompt")
    c.add_argument("--name", help="project name (default: derived from prompt)")
    c.add_argument("--duration", "-d", type=float, help="target duration in minutes")
    c.add_argument("--language", "-l", help="project language (code or name)")
    c.add_argument("--script-language", help="script language")
    c.add_argument("--narration-language", help="narration language")
    c.add_argument("--subtitle-language", help="subtitle language")
    c.add_argument("--onscreen-language", help="on-screen text language")
    c.add_argument("--voice", help="TTS engine/voice (auto|piper|espeak|sapi|silence)")
    c.add_argument("--animation-mode", choices=list(ANIMATION_MODES), help="animation mode")
    c.add_argument("--resolution", choices=sorted(RESOLUTIONS), help="export resolution")
    c.add_argument("--fps", type=int, choices=FRAME_RATES, help="frame rate")
    c.add_argument("--script-file", help="use a user-provided script file instead of generation")
    c.add_argument("--thumbnail-language", help="language for thumbnail text")
    c.add_argument("--no-thumbnails", action="store_true",
                   help="skip YouTube thumbnail generation after export")
    c.add_argument("--auto-select-thumbnail", action="store_true",
                   help="copy the recommended thumbnail to youtube_thumbnail.jpg automatically")
    c.add_argument("--low-resource", action="store_true", help="enable low-resource mode")
    c.add_argument("--music", choices=["none", "procedural"],
                   help="background audio: none or synthesized procedural ambience")
    c.add_argument("--no-burn-subtitles", action="store_true",
                   help="write .srt/.vtt files without burning them into the video")
    c.add_argument("--yes", "-y", action="store_true", help="skip final confirmation")
    c.add_argument("--interactive", action="store_true", help="force interactive prompts")
    c.add_argument("--non-interactive-defaults", action="store_true",
                   help="accept defaults for voice/style questions")
    c.add_argument("--no-prompts-extra", action="store_true",
                   help="skip burn-subtitle/music questions (use config defaults)")
    c.set_defaults(func=cmd_create)

    # plan / scenes / narration
    for name, fn, help_text in (
        ("plan", cmd_plan, "generate/inspect the script outline"),
        ("scenes", cmd_scenes, "show the scene manifest"),
        ("narration", cmd_narration, "generate narration audio"),
    ):
        p_ = sub.add_parser(name, help=help_text)
        p_.add_argument("project", help="project slug, name, or path")
        if name == "narration":
            p_.add_argument("--engine", help="override TTS engine")
        p_.set_defaults(func=fn)

    # generate
    g = sub.add_parser("generate", help="run the full render pipeline (resumes where possible)")
    g.add_argument("project", help="project slug, name, or path")
    g.add_argument("--resolution", choices=sorted(RESOLUTIONS), help="override export resolution")
    g.add_argument("--low-resource", action="store_true", help="enable low-resource mode")
    g.add_argument("--fresh", action="store_true", help="ignore previous stage results")
    g.set_defaults(func=cmd_generate)

    # resume
    r = sub.add_parser("resume", help="resume an interrupted project")
    r.add_argument("project", help="project slug")
    r.add_argument("--low-resource", action="store_true")
    r.add_argument("--resolution", choices=sorted(RESOLUTIONS))
    r.add_argument("--fresh", action="store_true")
    r.set_defaults(func=cmd_resume)

    # export
    e = sub.add_parser("export", help="assemble/export a project at a resolution")
    e.add_argument("project", help="project slug")
    e.add_argument("--resolution", choices=sorted(RESOLUTIONS), required=True)
    e.add_argument("--low-resource", action="store_true")
    e.add_argument("--fresh", action="store_true")
    e.set_defaults(func=cmd_export)

    # thumbnails
    th = sub.add_parser(
        "thumbnails",
        help="view, generate, regenerate or select YouTube thumbnails (no re-render)")
    th.add_argument("project", help="project slug, name, or path")
    th.add_argument("action", nargs="?", default="show",
                    choices=["show", "generate", "regenerate", "select", "open"],
                    help="show report | regenerate options | select final | open folder")
    th.add_argument("--index", type=int, choices=[1, 2, 3],
                    help="thumbnail number (select: default accepts the recommendation; "
                         "regenerate: default regenerates all three)")
    th.add_argument("--text", help="override the thumbnail text (generate/regenerate)")
    th.add_argument("--language", help="override the thumbnail language (generate/regenerate)")
    th.add_argument("--yes", "-y", action="store_true",
                    help="allow replacing a previously selected thumbnail")
    th.set_defaults(func=cmd_thumbnails)

    # status
    s = sub.add_parser("status", help="show project status")
    s.add_argument("project", help="project slug")
    s.set_defaults(func=cmd_status)

    # projects
    pr = sub.add_parser("projects", help="list projects")
    pr.add_argument("action", nargs="?", default="list", choices=["list"])
    pr.set_defaults(func=cmd_projects)

    # providers
    prov = sub.add_parser("providers", help="list provider availability")
    prov.add_argument("action", nargs="?", default="list", choices=["list"])
    prov.set_defaults(func=cmd_providers)

    # doctor / benchmark / clean-cache / languages
    sub.add_parser("doctor", help="check installation health").set_defaults(func=cmd_doctor)
    sub.add_parser("benchmark", help="measure render throughput on this machine").set_defaults(func=cmd_benchmark)
    cc = sub.add_parser("clean-cache", help="delete cached files")
    cc.add_argument("--yes", "-y", action="store_true")
    cc.set_defaults(func=cmd_clean_cache)
    sub.add_parser("languages", help="list supported languages").set_defaults(func=cmd_languages)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # No subcommand: run interactive create.
        args = parser.parse_args(["create"])
    try:
        return args.func(args)
    except ConfigError as exc:
        out(f"Configuration error: {exc}")
        return 2
    except FFmpegNotFoundError as exc:
        out(str(exc))
        return 3
    except ProjectError as exc:
        out(str(exc))
        return 2
    except KeyboardInterrupt:
        out("\nInterrupted. Progress is saved; run 'freeanimate resume <project>' to continue.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
