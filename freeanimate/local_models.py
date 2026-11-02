"""Local open-weight model registry and adapters.

This module never downloads anything automatically and never pretends a model is
available.  It provides:

* a registry of well-known open-weight image/video families with *approximate*
  sizes and requirement fields (licenses must always be checked on the model
  card at download time),
* availability checks (runtime imports, ComfyUI server/dir, model directory),
* real execution adapters for ``diffusers`` and a local ``ComfyUI`` HTTP server,
* a consent-gated download helper built on ``huggingface_hub``.

If the required runtime or GGUF/checkpoint is absent the adapter returns an
honest error; the studio then uses its procedural 2D/3D paths instead and labels
them accurately.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    id: str
    task: str                       # image | video
    family: str
    repo_id: str | None             # Hugging Face repo if applicable
    approx_download_gb: float
    min_ram_gb: float
    recommended_vram_gb: float
    cpu_capable: bool
    license_note: str
    supports: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        id="tiny_sd",
        task="image",
        family="Tiny SD (Segmind distilled SD 1.5)",
        repo_id="segmind/tiny-sd",
        approx_download_gb=1.1,
        min_ram_gb=6,
        recommended_vram_gb=4,
        cpu_capable=True,
        license_note="CreativeML OpenRAIL-M. Read the model card for the full use restrictions.",
        supports=["text-to-image", "512x512", "CPU-friendly"],
        notes="About 1.1 GB. Best fit for this CPU-only machine; quality is modest.",
    ),
    ModelSpec(
        id="sd_turbo",
        task="image",
        family="SD-Turbo (Stability AI)",
        repo_id="stabilityai/sd-turbo",
        approx_download_gb=13.0,
        min_ram_gb=8,
        recommended_vram_gb=4,
        cpu_capable=True,
        license_note="Released as a research artifact; commercial use requires checking "
                     "Stability AI's current license. Verify the model card before use.",
        supports=["text-to-image", "1-4 steps", "512x512"],
        notes="Full repository is ~13 GB; the fp16 subset is smaller. Research-oriented license.",
    ),
    ModelSpec(
        id="sd15",
        task="image",
        family="Stable Diffusion 1.5 family",
        repo_id="runwayml/stable-diffusion-v1-5",
        approx_download_gb=4.0,
        min_ram_gb=12,
        recommended_vram_gb=6,
        cpu_capable=True,
        license_note="CreativeML OpenRAIL-M style; read the checkpoint card and its use restrictions.",
        supports=["text-to-image", "img2img", "inpainting (pipeline-dependent)", "512x512"],
        notes="Wide ecosystem; practical on CPU only with patience.",
    ),
    ModelSpec(
        id="sdxl_base",
        task="image",
        family="Stable Diffusion XL base",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        approx_download_gb=7.0,
        min_ram_gb=16,
        recommended_vram_gb=8,
        cpu_capable=False,
        license_note="CreativeML OpenRAIL++-M style; verify the model card.",
        supports=["text-to-image", "1024x1024"],
        notes="Better quality; a dedicated GPU is strongly recommended.",
    ),
    ModelSpec(
        id="wan_t2v_1_3b",
        task="video",
        family="Wan 2.1 Text-to-Video 1.3B",
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        approx_download_gb=8.0,
        min_ram_gb=24,
        recommended_vram_gb=12,
        cpu_capable=False,
        license_note="Check the model card for the current license and commercial-use terms.",
        supports=["text-to-video", "short clips"],
        notes="Needs a capable GPU; not practical on CPU-only or low-RAM machines.",
    ),
]


def registry(task: str | None = None) -> list[ModelSpec]:
    return [m for m in MODEL_REGISTRY if task is None or m.task == task]


# ---------------------------------------------------------------------------
# Environment / availability
# ---------------------------------------------------------------------------
def models_dir() -> Path:
    env = os.environ.get("FREEANIMATE_MODELS_DIR")
    p = Path(env) if env else Path("models")
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_spec(model_id: str | None) -> ModelSpec | None:
    if not model_id:
        return None
    return next((m for m in MODEL_REGISTRY if m.id == model_id or m.repo_id == model_id), None)


def _model_roots() -> list[Path]:
    roots = [models_dir()]
    cache_models = Path("cache") / "models"
    if cache_models.resolve() != roots[0].resolve():
        roots.append(cache_models)
    return roots


def installed_models() -> list[dict[str, Any]]:
    """Return locally installed diffusers folders, without touching the network."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in _model_roots():
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not (child / "model_index.json").exists():
                continue
            key = str(child.resolve())
            if key in seen:
                continue
            seen.add(key)
            spec = model_spec(child.name)
            found.append({
                "id": spec.id if spec else child.name,
                "family": spec.family if spec else child.name,
                "path": str(child),
                "approx_gb": spec.approx_download_gb if spec else None,
                "license": spec.license_note if spec else "Unknown; check the model card.",
                "repo_id": spec.repo_id if spec else None,
            })
    return found


def resolve_local_model(model_id: str | None = None) -> tuple[ModelSpec | None, Path | None]:
    """Find an installed model by registry id, HF repo id, or filesystem path."""
    if model_id:
        direct = Path(model_id).expanduser()
        if direct.is_dir() and (direct / "model_index.json").exists():
            return model_spec(model_id), direct.resolve()
    spec = model_spec(model_id)
    ids = [model_id] if model_id else []
    if spec:
        ids += [spec.id, spec.repo_id, spec.repo_id.split("/")[-1] if spec.repo_id else None]
    for root in _model_roots():
        if not root.is_dir():
            continue
        for name in [x for x in ids if x]:
            candidate = root / name
            if candidate.is_dir() and (candidate / "model_index.json").exists():
                return spec, candidate.resolve()
        if not model_id:
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "model_index.json").exists():
                    return model_spec(child.name), child.resolve()
    return spec, None


def default_image_model_id() -> str | None:
    spec, path = resolve_local_model()
    return (spec.id if spec else (path.name if path else None))


def comfyui_dir() -> Path | None:
    env = os.environ.get("COMFYUI_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        Path.cwd() / "ComfyUI",
        Path.home() / "ComfyUI",
        Path("C:/ComfyUI"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "main.py").exists():
            return c
    return None


def comfyui_url() -> str:
    return os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class Availability:
    available: bool
    backend: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_image_backend(model_id: str | None = None) -> Availability:
    """Report which (if any) real local image backend is usable right now."""
    if _has_module("diffusers") and _has_module("torch"):
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            spec, path = resolve_local_model(model_id)
            if path is None:
                return Availability(
                    False, "diffusers",
                    f"diffusers + torch detected (device: {device}), but no local diffusers model "
                    f"was found under {models_dir()}. Nothing is downloaded automatically.")
            label = spec.family if spec else path.name
            return Availability(True, "diffusers",
                                f"diffusers + torch detected (device: {device}); model: {label} "
                                f"at {path}. "
                                + ("GPU inference available." if device == "cuda"
                                   else "CPU inference only; expect slow generation."))
        except Exception as exc:  # noqa: BLE001
            return Availability(False, None, f"torch import failed: {exc}")
    if comfyui_dir() is not None or _comfyui_reachable():
        return Availability(True, "comfyui", f"ComfyUI detected at {comfyui_url()}.")
    missing = [m for m in ("diffusers", "torch") if not _has_module(m)]
    return Availability(
        False, None,
        "No local image backend is usable. Missing: " + ", ".join(missing) +
        ". Install a compatible PyTorch build for this Python version and diffusers, or run "
        "ComfyUI locally. A model checkpoint is also required.",
    )


def check_video_backend() -> Availability:
    if comfyui_dir() is not None or _comfyui_reachable():
        return Availability(True, "comfyui",
                            "ComfyUI is reachable; a compatible video workflow and checkpoint are "
                            "still required.")
    if _has_module("diffusers") and _has_module("torch"):
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return Availability(False, "diffusers",
                                    "diffusers is installed but no CUDA GPU is available; open-weight "
                                    "video models are not practical on CPU.")
            return Availability(True, "diffusers",
                                "diffusers + CUDA detected; a compatible video model checkpoint is required.")
        except Exception:  # noqa: BLE001
            pass
    return Availability(False, None,
                        "No usable local video backend. A capable GPU, a compatible runtime and a "
                        "licensed video model are required.")


def _comfyui_reachable(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{comfyui_url()}/system_stats", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Real adapters (lazy imports; honest errors)
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    ok: bool
    path: Path | None = None
    error: str | None = None
    backend: str | None = None
    model_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def generate_image_diffusers(
    prompt: str,
    out_path: Path,
    *,
    model_id: str | None = None,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Generate an image with a local diffusers pipeline.

    This performs real inference only from a model already installed under
    ``models/`` (or an explicit local path).  It never downloads, never calls a
    cloud API, and never returns a bundled sample image as if the model made it.
    """
    if not (_has_module("diffusers") and _has_module("torch")):
        return GenerationResult(False, error="diffusers/torch not installed",
                                backend="diffusers", model_id=model_id)
    spec, local_path = resolve_local_model(model_id)
    if local_path is None:
        return GenerationResult(
            False,
            error=(f"No local diffusers model found under {models_dir()}. "
                   "Use 'python -m freeanimate models tiny_sd --download --yes' after "
                   "reviewing the size, license and disk space."),
            backend="diffusers", model_id=model_id)
    try:
        import time

        import torch  # type: ignore
        from diffusers import AutoPipelineForText2Image  # type: ignore

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            str(local_path), torch_dtype=dtype, local_files_only=True,
        ).to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(int(seed))
        label = (spec.family if spec else local_path.name)
        turbo = "turbo" in label.lower() or "turbo" in local_path.name.lower()
        default_steps = 1 if turbo else (8 if (spec and spec.id == "tiny_sd") else 20)
        n_steps = steps if steps is not None else default_steps
        started = time.time()
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": n_steps,
            "guidance_scale": 0.0 if turbo else 7.0,
            "generator": gen,
            "width": max(64, int(width) // 8 * 8),
            "height": max(64, int(height) // 8 * 8),
        }
        if negative_prompt and not turbo:
            kwargs["negative_prompt"] = negative_prompt
        image = pipe(**kwargs).images[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return GenerationResult(
            True, out_path, backend="diffusers",
            model_id=spec.id if spec else local_path.name,
            detail={"device": device, "steps": n_steps, "size": [image.width, image.height],
                    "model_path": str(local_path), "seconds": round(time.time() - started, 2),
                    "license": spec.license_note if spec else "Check the model card."})
    except Exception as exc:  # noqa: BLE001 � surface the real error
        return GenerationResult(False, error=f"{type(exc).__name__}: {exc}",
                                backend="diffusers", model_id=model_id)


def generate_image_comfyui(
    prompt: str,
    out_path: Path,
    *,
    checkpoint: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    seed: int = 0,
    timeout: float = 300.0,
) -> GenerationResult:
    """Generate an image through a local ComfyUI server using a standard workflow.

    ``checkpoint`` must be a checkpoint filename already installed in ComfyUI's
    ``models/checkpoints`` directory.  This is a real HTTP submit/poll/download
    implementation; it is not exercised unless ComfyUI is running locally.
    """
    if not _comfyui_reachable():
        return GenerationResult(False, error=f"ComfyUI not reachable at {comfyui_url()}",
                                backend="comfyui")
    base = comfyui_url()
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": 7.0, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0, "model": ["4", 0],
            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "freeanimate", "images": ["8", 0]}},
    }
    try:
        payload = json.dumps({"prompt": workflow}).encode("utf-8")
        req = urllib.request.Request(f"{base}/prompt", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            return GenerationResult(False, error=f"ComfyUI rejected the workflow: {data}",
                                    backend="comfyui")
        import time

        deadline = time.time() + timeout
        images: list[dict[str, Any]] = []
        while time.time() < deadline:
            with urllib.request.urlopen(f"{base}/history/{prompt_id}", timeout=30) as resp:
                hist = json.loads(resp.read().decode("utf-8"))
            entry = hist.get(prompt_id)
            if entry and entry.get("outputs"):
                for node in entry["outputs"].values():
                    for img in node.get("images", []):
                        images.append(img)
                if images:
                    break
            time.sleep(1.0)
        if not images:
            return GenerationResult(False, error="ComfyUI produced no images before the timeout",
                                    backend="comfyui")
        img = images[0]
        url = f"{base}/view?filename={urllib.parse.quote(img['filename'])}&subfolder={urllib.parse.quote(img.get('subfolder',''))}&type={img.get('type','output')}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            out_path.write_bytes(resp.read())
        return GenerationResult(True, out_path, backend="comfyui",
                                detail={"checkpoint": checkpoint})
    except Exception as exc:  # noqa: BLE001
        return GenerationResult(False, error=f"{type(exc).__name__}: {exc}", backend="comfyui")


def generate_video_comfyui(
    prompt: str,
    out_path: Path,
    *,
    workflow_template: Path,
    timeout: float = 1800.0,
) -> GenerationResult:
    """Run a user-supplied ComfyUI *video* workflow template.

    Video workflows differ per model and ComfyUI version, so we do not invent a
    universal node graph.  Provide a workflow JSON exported from ComfyUI with a
    ``{{PROMPT}}`` placeholder; it is submitted unchanged otherwise.
    """
    if not _comfyui_reachable():
        return GenerationResult(False, error=f"ComfyUI not reachable at {comfyui_url()}",
                                backend="comfyui")
    if not workflow_template.exists():
        return GenerationResult(False, error=f"workflow template not found: {workflow_template}",
                                backend="comfyui")
    try:
        text = workflow_template.read_text(encoding="utf-8").replace("{{PROMPT}}", prompt)
        workflow = json.loads(text)
        base = comfyui_url()
        req = urllib.request.Request(f"{base}/prompt",
                                     data=json.dumps({"prompt": workflow}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            return GenerationResult(False, error=f"ComfyUI rejected workflow: {data}",
                                    backend="comfyui")
        import time

        deadline = time.time() + timeout
        videos: list[dict[str, Any]] = []
        while time.time() < deadline:
            with urllib.request.urlopen(f"{base}/history/{prompt_id}", timeout=30) as resp:
                hist = json.loads(resp.read().decode("utf-8"))
            entry = hist.get(prompt_id)
            if entry and entry.get("outputs"):
                for node in entry["outputs"].values():
                    videos.extend(node.get("gifs", []) or node.get("videos", []))
                if videos:
                    break
            time.sleep(2.0)
        if not videos:
            return GenerationResult(False, error="no video output before timeout", backend="comfyui")
        v = videos[0]
        url = f"{base}/view?filename={urllib.parse.quote(v['filename'])}&subfolder={urllib.parse.quote(v.get('subfolder',''))}&type={v.get('type','output')}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            out_path.write_bytes(resp.read())
        return GenerationResult(True, out_path, backend="comfyui")
    except Exception as exc:  # noqa: BLE001
        return GenerationResult(False, error=f"{type(exc).__name__}: {exc}", backend="comfyui")


# ---------------------------------------------------------------------------
# Consent-gated model download
# ---------------------------------------------------------------------------
@dataclass
class DownloadPlan:
    model_id: str
    repo_id: str | None
    approx_gb: float
    license_note: str
    target_dir: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_download(spec: ModelSpec, target_root: Path | None = None) -> DownloadPlan:
    root = Path(target_root) if target_root else models_dir()
    return DownloadPlan(
        model_id=spec.id,
        repo_id=spec.repo_id,
        approx_gb=spec.approx_download_gb,
        license_note=spec.license_note,
        target_dir=str(root / spec.id),
        note="Nothing is downloaded until you explicitly confirm. Verify the license on the model "
             "card and ensure enough free disk space first.",
    )


def download_model(spec: ModelSpec, target_root: Path | None = None, *,
                   confirm: bool = False, allow_patterns: list[str] | None = None) -> DownloadPlan:
    """Download a model only when ``confirm=True``.

    Raises ``PermissionError`` otherwise.  Requires ``huggingface_hub``.
    """
    plan = plan_download(spec, target_root)
    if not confirm:
        raise PermissionError("model download requires explicit confirmation (confirm=True)")
    if not spec.repo_id:
        raise ValueError(f"{spec.id} has no Hugging Face repo; download manually")
    if not _has_module("huggingface_hub"):
        raise RuntimeError("huggingface_hub is not installed; pip install huggingface_hub")
    from huggingface_hub import snapshot_download  # type: ignore

    target = Path(plan.target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec.repo_id, local_dir=str(target),
                      allow_patterns=allow_patterns)
    return plan


# urllib.parse is used inside the ComfyUI adapters.
import urllib.parse  # noqa: E402  (kept near use for clarity)
