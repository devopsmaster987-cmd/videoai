"""FastAPI application for the local FreeAnimate studio.

The server binds to 127.0.0.1 by default (see :func:`freeanimate.cli.cmd_serve`)
and serves the browser UI, JSON APIs, render jobs and generated media.  Nothing
is uploaded anywhere; jobs run in local worker threads.
"""

from __future__ import annotations

import json
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from ..blender3d import find_blender
from ..characters import CharacterLibrary
from ..config import Config, ConfigError
from ..hardware import detect_hardware, model_suggestions
from ..jobs import JobContext, JobRunner, JobStore
from ..local_models import (MODEL_REGISTRY, check_image_backend, check_video_backend,
                            default_image_model_id, generate_image_diffusers,
                            installed_models, models_dir, plan_download, registry)
from ..prompt_understanding import DialogueLine, understand_prompt
from ..project import Project, ProjectError, list_projects, resolve_project
from ..studio import Studio, StudioOptions, RESOLUTIONS
from ..tts import available_voices

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
STATIC = HERE / "static"


def _payload_options(data: dict[str, Any]) -> StudioOptions:
    allowed = {
        "title", "method", "resolution", "fps", "language", "music", "music_volume",
        "burn_subtitles", "duration_seconds", "crf", "preset", "tts_engine", "tts_voice",
        "tts_speed", "aspect_ratio", "seed", "image_model",
    }
    opts = {k: v for k, v in data.items() if k in allowed}
    if "fps" in opts:
        opts["fps"] = int(opts["fps"])
    if "duration_seconds" in opts and opts["duration_seconds"] in ("", None):
        opts.pop("duration_seconds")
    return StudioOptions(**opts)


def create_app(config_path: Path | str | None = None, *,
               projects_root: Path | str | None = None,
               cache_root: Path | str | None = None) -> FastAPI:
    config = Config.load(config_path)
    studio = Studio(config, projects_root=Path(projects_root) if projects_root else None,
                    cache_root=Path(cache_root) if cache_root else None)
    store = JobStore(studio.cache_root / "jobs.db")
    runner = JobRunner(store)
    generated_dir = studio.cache_root / "generated_images"
    generated_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="FreeAnimate Studio", docs_url="/api/docs", redoc_url=None)
    app.state.studio = studio
    app.state.store = store
    app.state.runner = runner
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    app.mount("/media", StaticFiles(directory=str(studio.projects_root)), name="media")
    app.mount("/generated", StaticFiles(directory=str(generated_dir)), name="generated")
    templates = Jinja2Templates(directory=str(TEMPLATES))

    def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        ctx.setdefault("config", config)
        return templates.TemplateResponse(request, name, ctx)

    def project_or_404(slug: str) -> Project:
        try:
            return resolve_project(studio.projects_root, slug)
        except ProjectError as exc:
            raise HTTPException(404, str(exc)) from exc

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        projects = list_projects(studio.projects_root)
        for p in projects:
            p["url"] = f"/media/{p['slug']}/renders/final.mp4"
            p["thumbnail_url"] = f"/media/{p['slug']}/thumbnails/thumbnail.jpg"
            p["page"] = f"/projects/{p['slug']}"
            p["output_exists"] = (Path(p["path"]) / "renders" / "final.mp4").exists()
        jobs = store.list(limit=8)
        return render(request, "index.html", title="Projects", projects=projects, jobs=jobs)

    @app.get("/new", response_class=HTMLResponse)
    def new_video(request: Request) -> HTMLResponse:
        return render(request, "new.html", title="New video",
                      resolutions=sorted(RESOLUTIONS), methods=["auto", "2d", "3d", "3d_cartoon", "realistic_3d",
                               "cinematic", "photoreal", "still_image", "ai_image"])

    @app.get("/images", response_class=HTMLResponse)
    def images_page(request: Request) -> HTMLResponse:
        backend = check_image_backend()
        return render(request, "images.html", title="Image lab", backend=backend,
                      models=installed_models(), default_model=default_image_model_id(),
                      models_path=str(models_dir()), projects=list_projects(studio.projects_root))

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str) -> HTMLResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return render(request, "job.html", title=f"Job {job_id}", job=job)

    @app.get("/projects/{slug}", response_class=HTMLResponse)
    def project_page(request: Request, slug: str) -> HTMLResponse:
        project = project_or_404(slug)
        plan = project.read_json("script", "prompt_plan.json") or {}
        convo = project.read_json("narration", "conversation.json") or {}
        report = None
        for name in ("validation_report.json", "report.json"):
            p = project.path("reports", name)
            if p.exists():
                report = json.loads(p.read_text(encoding="utf-8"))
                break
        chars: list[dict[str, Any]] = []
        lib_dir = project.path("assets", "characters")
        if (lib_dir / "library.json").exists():
            lib = CharacterLibrary(lib_dir)
            for c in lib.list():
                ref = c.selected_reference or (c.reference_images[0] if c.reference_images else None)
                chars.append({
                    "id": c.id, "name": c.name, "species": c.species,
                    "clothing": c.clothing, "personality": c.personality,
                    "palette": c.color_palette,
                    "portrait": (f"/media/{project.manifest['slug']}/assets/characters/"
                                 f"{ref}" if ref else None),
                })
        media = f"/media/{project.manifest['slug']}"
        renders_dir = project.path("renders")
        scenes = [
            {"index": i + 1, "url": f"{media}/renders/{f.name}",
             "size_mb": round(f.stat().st_size / (1024 * 1024), 1)}
            for i, f in enumerate(sorted(renders_dir.glob("scene_*.mp4")))
        ]
        return render(request, "project.html", title=project.manifest.get("name", slug),
                      project=project, plan=plan, convo=convo, report=report,
                      characters=chars, media=media, scenes=scenes)

    @app.get("/hardware", response_class=HTMLResponse)
    def hardware_page(request: Request) -> HTMLResponse:
        hw = detect_hardware(studio.projects_root)
        suggestions = model_suggestions(hw)
        blender = find_blender()
        return render(request, "hardware.html", title="Hardware & capabilities",
                      hw=hw, suggestions=suggestions, blender=blender)

    @app.get("/models", response_class=HTMLResponse)
    def models_page(request: Request) -> HTMLResponse:
        image = check_image_backend()
        video = check_video_backend()
        return render(request, "models.html", title="Local models",
                      models=[m.to_dict() for m in MODEL_REGISTRY],
                      image_backend=image, video_backend=video)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request) -> HTMLResponse:
        from ..constants import FRAME_RATES

        return render(request, "settings.html", title="Settings",
                      settings=config.data, source_path=str(config.source_path or ""),
                      resolutions=sorted(RESOLUTIONS), frame_rates=sorted(FRAME_RATES),
                      tts_engines=["auto", "sapi", "espeak", "piper", "silence"])

    @app.get("/characters", response_class=HTMLResponse)
    def characters_page(request: Request) -> HTMLResponse:
        found: list[dict[str, Any]] = []
        for info in list_projects(studio.projects_root):
            lib_dir = Path(info["path"]) / "assets" / "characters"
            if not (lib_dir / "library.json").exists():
                continue
            lib = CharacterLibrary(lib_dir)
            for c in lib.list():
                ref = c.selected_reference or (c.reference_images[0] if c.reference_images else None)
                found.append({
                    "name": c.name, "species": c.species, "project": info["slug"],
                    "clothing": c.clothing, "personality": c.personality,
                    "palette": c.color_palette,
                    "portrait": (f"/media/{info['slug']}/assets/characters/{ref}" if ref else None),
                })
        return render(request, "characters.html", title="Characters", characters=found)

    # -------------------------------------------------------------------- API
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": "1.0", "time": time.time(),
                "projects": len(list_projects(studio.projects_root)),
                "jobs": store.counts()}

    @app.post("/api/plan")
    async def api_plan(request: Request) -> JSONResponse:
        data = await request.json()
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "prompt is required")
        opts = _payload_options(data)
        plan, hw = studio.plan(prompt, opts)
        return JSONResponse({
            "title": plan.title, "project_type": plan.project_type,
            "requested_method": plan.requested_method,
            "supported_method": plan.supported_method, "method_note": plan.method_note,
            "environment": plan.environment_type, "setting": plan.setting,
            "characters": [c.model_dump() for c in plan.characters],
            "dialogue": [d.model_dump() for d in plan.dialogue],
            "scenes": [s.model_dump() for s in plan.scenes],
            "duration_seconds": plan.duration_seconds, "aspect_ratio": plan.aspect_ratio,
            "warnings": plan.warnings, "notes": plan.parse_notes,
            "capabilities": {k: c.status for k, c in hw.capabilities.items()},
        })

    @app.post("/api/projects")
    async def api_create_project(request: Request) -> JSONResponse:
        data = await request.json()
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "prompt is required")
        opts = _payload_options(data)
        dialogue_payload = data.get("dialogue")
        dialogue_text = (data.get("dialogue_text") or "").strip()
        if dialogue_payload:
            try:
                opts.dialogue = [DialogueLine(**d) for d in dialogue_payload]
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"invalid dialogue: {exc}") from exc
        elif dialogue_text:
            from ..conversation import parse_dialogue_text

            plan, _ = studio.plan(prompt, opts)
            try:
                opts.dialogue = parse_dialogue_text(dialogue_text, plan.characters)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        params = {**opts.__dict__, "prompt": prompt}
        if opts.dialogue is not None:
            params["dialogue"] = [d.model_dump() for d in opts.dialogue]

        def work(ctx: JobContext, p: dict[str, Any]) -> dict[str, Any]:
            o = StudioOptions(**{k: v for k, v in p.items()
                                 if k in StudioOptions.__dataclass_fields__ and k != "prompt"})
            if "dialogue" in p and p["dialogue"]:
                o.dialogue = [DialogueLine(**d) for d in p["dialogue"]]
            res = studio.run(p["prompt"], o,
                             progress=lambda stage, frac, msg: ctx.progress(stage, frac, msg),
                             cancel=lambda: ctx.store.is_cancelled(ctx.job_id))
            if not res.ok:
                raise RuntimeError(res.error or "render failed")
            return res.to_dict()

        job = runner.submit("new", "conversation_video", work, params)
        return JSONResponse({"job_id": job.id, "status": job.status})

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        d = job.to_dict()
        d["running"] = runner.is_running(job_id)
        return d

    @app.get("/api/jobs")
    def api_jobs(project_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [j.to_dict() for j in store.list(project_id=project_id, limit=limit)]

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel(job_id: str) -> dict[str, Any]:
        if store.get(job_id) is None:
            raise HTTPException(404, "job not found")
        return {"cancelled": runner.cancel(job_id)}

    @app.get("/api/voices")
    def api_voices(language: str = "en") -> dict[str, Any]:
        voices = available_voices(language, config=config.data.get("tts", {}))
        return {"language": language, "count": len({v.id for v in voices}),
                "voices": [v.to_dict() for v in voices]}

    @app.get("/api/hardware")
    def api_hardware() -> dict[str, Any]:
        hw = detect_hardware(studio.projects_root)
        return {"report": hw.to_dict(), "suggestions": model_suggestions(hw),
                "blender": find_blender().to_dict()}

    @app.get("/api/models")
    def api_models(task: str | None = None) -> dict[str, Any]:
        return {"models": [m.to_dict() for m in registry(task)],
                "image_backend": check_image_backend().to_dict(),
                "video_backend": check_video_backend().to_dict()}

    @app.post("/api/models/plan")
    async def api_model_plan(request: Request) -> dict[str, Any]:
        data = await request.json()
        model = next((m for m in MODEL_REGISTRY if m.id == data.get("id")), None)
        if model is None:
            raise HTTPException(404, "unknown model")
        return plan_download(model).to_dict()

    @app.get("/api/image/status")
    def api_image_status() -> dict[str, Any]:
        backend = check_image_backend()
        hw = detect_hardware(studio.projects_root)
        return {
            "backend": backend.to_dict(),
            "models": installed_models(),
            "default_model": default_image_model_id(),
            "models_path": str(models_dir()),
            "hardware": {
                "cpu": hw.cpu_model, "cores": hw.cpu_cores, "ram_gb": hw.ram_gb,
                "gpu": hw.gpu.name, "cuda": hw.gpu.cuda_available,
                "disk_free_gb": hw.disk_free_gb,
            },
        }

    @app.post("/api/image/generate")
    async def api_image_generate(request: Request) -> dict[str, Any]:
        data = await request.json()
        prompt = str(data.get("prompt") or "").strip()
        if len(prompt) < 3:
            raise HTTPException(400, "prompt is required")
        backend = check_image_backend(data.get("model"))
        if not backend.available:
            raise HTTPException(409, backend.detail)
        model_id = data.get("model") or default_image_model_id()
        if not model_id:
            raise HTTPException(409, f"No local model installed under {models_dir()}.")
        seed = data.get("seed")
        try:
            seed = int(seed) if seed not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "seed must be an integer") from exc
        try:
            width = max(256, min(768, int(data.get("width") or 512)))
            height = max(256, min(768, int(data.get("height") or 512)))
            steps = data.get("steps")
            steps = int(steps) if steps not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "width, height and steps must be integers") from exc
        filename = f"image_{int(time.time())}_{seed if seed is not None else 'random'}.png"
        out_path = generated_dir / filename
        result = await run_in_threadpool(
            generate_image_diffusers, prompt, out_path,
            model_id=model_id, negative_prompt=str(data.get("negative_prompt") or ""),
            width=width, height=height, steps=steps, seed=seed,
        )
        if not result.ok or not result.path:
            raise HTTPException(500, result.error or "image generation failed")
        return {
            "ok": True, "image": result.path.name,
            "url": f"/generated/{result.path.name}",
            "model": result.model_id, "backend": result.backend,
            "detail": result.detail, "prompt": prompt,
        }

    @app.post("/api/image/use")
    async def api_image_use(request: Request) -> dict[str, Any]:
        data = await request.json()
        name = Path(str(data.get("image") or "")).name
        source = generated_dir / name
        if not name or not source.is_file():
            raise HTTPException(404, "generated image not found")
        project = project_or_404(str(data.get("project") or ""))
        target = project.path("assets", "custom_background.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return {"ok": True, "project": project.manifest.get("slug", ""), "path": str(target),
                "note": "Next render of this project will reuse this image as its background."}

    @app.post("/api/settings")
    async def api_settings(request: Request) -> dict[str, Any]:
        data = await request.json()
        updates = data.get("updates") or {}
        if not isinstance(updates, dict) or not updates:
            raise HTTPException(400, "updates must be a non-empty object")
        allowed = {
            "defaults.resolution", "defaults.fps", "defaults.wpm", "defaults.music",
            "defaults.music_volume", "defaults.burn_subtitles", "render.crf",
            "render.sample_rate", "tts.engine", "tts.speed",
            "subtitles.max_line_chars", "subtitles.max_lines",
        }
        integer = {"defaults.fps", "defaults.wpm", "render.crf", "render.sample_rate",
                   "subtitles.max_line_chars", "subtitles.max_lines"}
        number = {"defaults.music_volume", "tts.speed"}
        boolean = {"defaults.burn_subtitles"}
        before = deepcopy(config.data)
        for dotted, value in updates.items():
            if dotted not in allowed:
                raise HTTPException(400, f"setting {dotted!r} cannot be changed from the browser")
            section, key = dotted.split(".", 1)
            try:
                if dotted in integer:
                    value = int(value)
                elif dotted in number:
                    value = float(value)
                elif dotted in boolean:
                    value = bool(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"invalid value for {dotted}") from exc
            if dotted == "defaults.music" and value not in ("none", "procedural"):
                raise HTTPException(400, "default music must be 'none' or 'procedural'")
            config.data[section][key] = value
        try:
            config.validate()
        except ConfigError as exc:
            config.data = before
            raise HTTPException(400, str(exc)) from exc
        try:
            target = config.source_path or (HERE.parent.parent / "config" / "config.json")
            config.save(target)
        except OSError as exc:
            config.data = before
            raise HTTPException(500, f"could not save configuration: {exc}") from exc
        return {"ok": True, "source_path": str(config.source_path), "settings": config.data}

    @app.post("/api/projects/{slug}/scenes/{index}/rerender")
    async def api_scene_rerender(request: Request, slug: str, index: int) -> dict[str, Any]:
        project = project_or_404(slug)
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001 — empty body is valid
            data = {}
        overrides = _payload_options(data) if isinstance(data, dict) and data else None

        def work(ctx: JobContext, p: dict[str, Any]) -> dict[str, Any]:
            res = studio.regenerate_scene(
                p["slug"], int(p["index"]), overrides,
                progress=lambda stage, frac, msg: ctx.progress(stage, frac, msg),
                cancel=lambda: ctx.store.is_cancelled(ctx.job_id))
            if not res.ok:
                raise RuntimeError(res.error or "scene re-render failed")
            return res.to_dict()

        job = runner.submit(project.manifest["slug"], f"scene_{index}_rerender",
                            work, {"slug": project.manifest["slug"], "index": index})
        return {"job_id": job.id, "status": job.status}

    @app.get("/api/projects/{slug}/export.zip")
    def api_export_zip(slug: str) -> FileResponse:
        project = project_or_404(slug)
        final = project.path("renders", "final.mp4")
        if not final.exists():
            raise HTTPException(404, "this project has no finished video yet")
        out_dir = studio.cache_root / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{project.manifest['slug']}_export.zip"
        include = [
            ("project.json", project.path("project.json")),
            ("renders/final.mp4", final),
            ("subtitles/subtitles.srt", project.path("subtitles", "subtitles.srt")),
            ("subtitles/subtitles.vtt", project.path("subtitles", "subtitles.vtt")),
            ("audio/mix.wav", project.path("audio", "mix.wav")),
            ("reports/validation_report.json", project.path("reports", "validation_report.json")),
            ("reports/studio_report.json", project.path("reports", "studio_report.json")),
            ("script/prompt_plan.json", project.path("script", "prompt_plan.json")),
            ("narration/conversation.json", project.path("narration", "conversation.json")),
        ]
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for arc, path in include:
                if path.exists():
                    zf.write(path, arc)
        return FileResponse(out, media_type="application/zip",
                            filename=f"{project.manifest['slug']}_export.zip")

    @app.get("/api/projects/{slug}/plan")
    def api_project_plan(slug: str) -> Any:
        project = project_or_404(slug)
        return project.read_json("script", "prompt_plan.json") or {}

    @app.post("/api/projects/{slug}/rerender")
    async def api_rerender(request: Request, slug: str) -> dict[str, Any]:
        project = project_or_404(slug)
        data = await request.json()
        from ..conversation import parse_dialogue_text
        from ..prompt_understanding import CharacterSpec

        plan_data = project.read_json("script", "prompt_plan.json") or {}
        specs = [CharacterSpec(**c) for c in plan_data.get("characters", [])]
        chars_dir = project.path("assets", "characters")
        characters = CharacterLibrary(chars_dir).list() if (chars_dir / "library.json").exists() else []
        text = (data.get("dialogue_text") or "").strip()
        if not text:
            raise HTTPException(400, "dialogue_text is required")
        try:
            lines = parse_dialogue_text(text, characters or specs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        opts = _payload_options(data)
        opts.title = data.get("title") or (project.manifest.get("name") + " (edit)")
        opts.dialogue = lines
        params = {**opts.__dict__, "prompt": project.manifest.get("prompt", ""),
                  "dialogue": [d.model_dump() for d in lines]}

        def work(ctx: JobContext, p: dict[str, Any]) -> dict[str, Any]:
            o = StudioOptions(**{k: v for k, v in p.items()
                                 if k in StudioOptions.__dataclass_fields__ and k != "prompt"})
            o.dialogue = [DialogueLine(**d) for d in p["dialogue"]]
            res = studio.run(p["prompt"], o,
                             progress=lambda stage, frac, msg: ctx.progress(stage, frac, msg),
                             cancel=lambda: ctx.store.is_cancelled(ctx.job_id))
            if not res.ok:
                raise RuntimeError(res.error or "render failed")
            return res.to_dict()

        job = runner.submit(slug, "conversation_video_edit", work, params)
        return {"job_id": job.id, "status": job.status}

    return app
