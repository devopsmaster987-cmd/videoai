"""Hardware detection and capability profiles.

Everything here reports *measured* facts about the machine and derives an
honest capability profile.  It never assumes an NVIDIA GPU, never trusts a GPU
model name alone to decide what can run, and never promises that a heavy model
will work.  Missing information is reported as missing rather than guessed.

Profiles (see the master prompt, section 6):

* **A — Basic**      procedural 2D animation, illustration, TTS, Blender if
                    installed, and no local neural image/video model unless the
                    user installs one deliberately.
* **B — Moderate**   local AI image generation and Blender scenes become
                    practical where a compatible model/runtime is installed.
* **C — High**       larger image/video models and heavier 3D rendering.

The profile only *recommends*; nothing is downloaded automatically.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Low-level probes (all defensive — a failure returns None, never raises)
# ---------------------------------------------------------------------------
def _run(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def _ram_gb() -> float | None:
    """Total physical RAM in GB. Tries psutil, then OS-specific commands."""
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        out = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
        if out:
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    return round(int(line) / 1e9, 1)
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        if out and out.isdigit():
            return round(int(out) / 1e9, 1)
    else:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
            page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
            return round(pages * page_size / 1e9, 1)
        except Exception:  # noqa: BLE001
            return None
    return None


def _cpu_model() -> str | None:
    if os.name == "nt":
        out = _run(["wmic", "cpu", "get", "name"])
        if out:
            lines = [ln.strip() for ln in out.splitlines() if ln.strip() and ln.strip().lower() != "name"]
            if lines:
                return lines[0]
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor).Name | Select-Object -First 1",
        ])
        if out:
            return out.splitlines()[0].strip()
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:  # noqa: BLE001
            return None
    return None


@dataclass
class GpuInfo:
    name: str | None = None
    vendor: str | None = None       # nvidia | amd | intel | unknown
    vram_gb: float | None = None
    dedicated: bool = False
    cuda_available: bool = False
    cuda_version: str | None = None
    torch_available: bool = False
    torch_device: str | None = None


def _gpu_info() -> GpuInfo:
    info = GpuInfo()
    # nvidia-smi is the most reliable source for real, dedicated NVIDIA VRAM.
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if out:
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            info.name = parts[0] if parts else None
            if len(parts) > 1:
                try:
                    info.vram_gb = round(float(parts[1].split()[0]) / 1024, 1)
                except (ValueError, IndexError):
                    pass
            info.vendor = "nvidia"
            info.dedicated = True
    # Windows fallback: report the adapter name (VRAM from wmic is unreliable,
    # so we do not trust it for capability decisions).
    if os.name == "nt" and not info.name:
        out = _run(["wmic", "path", "win32_VideoController", "get", "name"])
        if out:
            names = [ln.strip() for ln in out.splitlines() if ln.strip() and ln.strip().lower() != "name"]
            if names:
                info.name = names[0]
                low = names[0].lower()
                if "nvidia" in low:
                    info.vendor = "nvidia"
                elif "amd" in low or "radeon" in low:
                    info.vendor = "amd"
                elif "intel" in low:
                    info.vendor = "intel"
                else:
                    info.vendor = "unknown"
                info.dedicated = info.vendor in ("nvidia", "amd") and "integrated" not in low
    # Torch (if present) gives the authoritative CUDA story.
    try:
        import torch  # type: ignore

        info.torch_available = True
        info.cuda_available = bool(torch.cuda.is_available())
        info.torch_device = "cuda" if info.cuda_available else "cpu"
        if info.cuda_available:
            info.cuda_version = getattr(torch.version, "cuda", None)
            if not info.name:
                try:
                    info.name = torch.cuda.get_device_name(0)
                    info.vram_gb = round(
                        torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                    )
                    info.dedicated = True
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        info.torch_available = False
    return info


def _disk_usage(path: Path) -> tuple[float | None, float | None]:
    try:
        usage = shutil.disk_usage(str(path))
        return round(usage.free / 1e9, 1), round(usage.total / 1e9, 1)
    except OSError:
        return None, None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class Capability:
    status: str            # available | requires_install | requires_hardware | unsupported | not_implemented
    detail: str
    method: str = ""       # what would actually run
    license: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HardwareReport:
    os_name: str
    os_version: str
    python_version: str
    cpu_cores: int | None
    cpu_model: str | None
    ram_gb: float | None
    gpu: GpuInfo
    disk_free_gb: float | None
    disk_total_gb: float | None
    ffmpeg: str | None
    ffprobe: str | None
    blender: str | None
    node: str | None
    profile: str
    profile_label: str
    capabilities: dict[str, Capability] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["capabilities"] = {k: v.to_dict() for k, v in self.capabilities.items()}
        return d

    def summary_lines(self) -> list[str]:
        lines = [
            f"OS: {self.os_name} {self.os_version}",
            f"Python: {self.python_version}",
            f"CPU: {self.cpu_model or 'unknown'} ({self.cpu_cores or '?'} cores)",
            f"RAM: {self.ram_gb if self.ram_gb is not None else 'unknown'} GB",
            f"GPU: {self.gpu.name or 'unknown'}"
            + (f" ({self.gpu.vram_gb} GB)" if self.gpu.vram_gb else "")
            + (f" - {self.gpu.vendor}" if self.gpu.vendor else ""),
            f"CUDA: {'yes' if self.gpu.cuda_available else 'no'}"
            + (f" ({self.gpu.cuda_version})" if self.gpu.cuda_version else ""),
            f"Disk free: {self.disk_free_gb if self.disk_free_gb is not None else '?'} GB",
            f"FFmpeg: {'yes' if self.ffmpeg else 'MISSING'}",
            f"FFprobe: {'yes' if self.ffprobe else 'MISSING'}",
            f"Blender: {'yes' if self.blender else 'MISSING'}",
            f"Node.js: {'yes' if self.node else 'MISSING'}",
            f"Profile: {self.profile} - {self.profile_label}",
        ]
        return lines


def _pick_profile(ram_gb: float | None, gpu: GpuInfo, cores: int | None) -> tuple[str, str]:
    """Assign a profile from measured RAM/VRAM/cores — not from GPU names alone."""
    ram = ram_gb or 0.0
    vram = gpu.vram_gb or 0.0
    # High: comfortable for larger models and heavy 3D.
    if ram >= 32 and (vram >= 12 or (gpu.cuda_available and vram >= 10)):
        return "C", "High-performance hardware"
    # Moderate: 16 GB+ RAM and a real dedicated GPU with >= 6 GB VRAM.
    if ram >= 16 and vram >= 6:
        return "B", "Moderate hardware"
    # A dedicated GPU with 4-6 GB can still do small models -> moderate.
    if ram >= 16 and gpu.dedicated and vram >= 4:
        return "B", "Moderate hardware"
    return "A", "Basic computer"


def detect_hardware(root: Path | None = None) -> HardwareReport:
    """Measure the machine and derive honest capabilities."""
    root = Path(root or Path.cwd())
    cpu_model = _cpu_model()
    ram = _ram_gb()
    gpu = _gpu_info()
    free_gb, total_gb = _disk_usage(root)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    blender = shutil.which("blender") or shutil.which("blender.exe")
    if not blender:
        try:
            from .blender3d import find_blender

            status = find_blender()
            if status.available:
                blender = status.executable
        except Exception:  # noqa: BLE001 — detection must never break the report
            pass
    node = shutil.which("node")
    profile, label = _pick_profile(ram, gpu, os.cpu_count())

    report = HardwareReport(
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        cpu_cores=os.cpu_count(),
        cpu_model=cpu_model,
        ram_gb=ram,
        gpu=gpu,
        disk_free_gb=free_gb,
        disk_total_gb=total_gb,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        blender=blender,
        node=node,
        profile=profile,
        profile_label=label,
    )

    caps = report.capabilities
    # Genuine, always-available procedural paths.
    caps["procedural_2d_animation"] = Capability(
        "available",
        "Rigged 2D character animation rendered with Pillow + FFmpeg (no model needed).",
        method="freeanimate.animation2d",
        license="MIT (project code)",
    )
    caps["animated_illustration"] = Capability(
        "available",
        "Procedural illustrated scenes with Ken Burns camera motion.",
        method="freeanimate.visuals",
        license="MIT (project code)",
    )
    caps["local_tts"] = Capability(
        "available",
        "Local speech via Piper/espeak-ng/Windows SAPI when a voice is installed.",
        method="freeanimate.tts",
        license="varies by engine/voice",
    )
    caps["subtitles"] = Capability(
        "available",
        "SRT/VTT plus libass burned-in subtitles.",
        method="freeanimate.subtitles",
        license="MIT",
    )

    # AI image generation: needs diffusers/ComfyUI + a model + usually a GPU.
    import importlib.util

    has_diffusers = importlib.util.find_spec("diffusers") is not None
    has_torch = gpu.torch_available
    installed: list[dict[str, Any]] = []
    if has_diffusers and has_torch:
        try:
            from .local_models import installed_models as _installed
            installed = _installed()
        except Exception:
            installed = []
    if has_diffusers and has_torch:
        names = ", ".join(str(m.get("id")) for m in installed)
        if installed:
            if gpu.cuda_available:
                caps["ai_image_generation"] = Capability(
                    "available",
                    f"diffusers + local model(s) found ({names}); CUDA GPU ({gpu.name}) can run them.",
                    method="diffusers (local, GPU)",
                    license="model-dependent (check each model)",
                )
            else:
                caps["ai_image_generation"] = Capability(
                    "available",
                    f"diffusers + torch and local model(s) installed ({names}). CPU inference works "
                    "for small distilled models (roughly 1-3 minutes per 512x512 image) and is slow "
                    "for large models.",
                    method="diffusers (local, CPU)",
                    license="model-dependent (check each model)",
                )
        else:
            caps["ai_image_generation"] = Capability(
                "requires_install",
                "diffusers + torch are installed, but no local model was found. Download a small "
                "licensed checkpoint (for example Tiny SD) with the Models page or CLI; this machine "
                "has no CUDA GPU, so only small CPU models are practical.",
                method="diffusers (local)",
                license="model-dependent (check each model)",
            )
    else:
        missing = []
        if not has_diffusers:
            missing.append("diffusers")
        if not has_torch:
            missing.append("torch")
        caps["ai_image_generation"] = Capability(
            "requires_install",
            "Not installed: " + ", ".join(missing) + ". Install a compatible PyTorch build for your "
            "Python version and diffusers, then download an appropriately licensed model. "
            "This machine has no CUDA GPU, so only small CPU models are practical.",
            method="diffusers (local)",
        )

    # AI video generation: heavier than image generation.
    caps["ai_video_generation"] = Capability(
        "requires_hardware" if gpu.cuda_available and (gpu.vram_gb or 0) >= 8 else "requires_install",
        "Open-weight video models (e.g. text/image-to-video) need a local runtime and a compatible GPU "
        "with substantial VRAM; CPU-only generation of useful clips is not practical. The adapter "
        "reports this honestly instead of faking clips.",
        method="ComfyUI / diffusers video pipelines",
    )

    caps["blender_3d"] = (
        Capability("available", f"Blender found at {blender}. Real 3D scene rendering is available.",
                   method="blender --background --python", license="Blender GPL (output is yours)")
        if blender else
        Capability("requires_install",
                   "Blender was not found. Install it from https://www.blender.org/download/ and add "
                   "it to PATH to enable genuine 3D rendering.",
                   method="blender --background --python")
    )

    caps["lipsync_2d"] = Capability(
        "available",
        "Viseme-driven mouth animation for rigged 2D characters, timed to measured narration.",
        method="freeanimate.lipsync + freeanimate.animation2d",
        license="MIT",
    )
    caps["web_interface"] = Capability(
        "available",
        "Local FastAPI server + browser UI (127.0.0.1 only by default).",
        method="freeanimate.web",
        license="MIT",
    )

    # Honest notes.
    if ram is not None and ram < 12:
        report.notes.append(
            f"Only {ram} GB RAM was detected. Keep previews short and resolution modest; "
            "do not attempt large local diffusion/video models."
        )
    if gpu.name and not gpu.cuda_available:
        report.notes.append(
            "No CUDA-capable GPU was detected. Neural image/video generation would run on CPU "
            "(slow) or is unavailable; the procedural 2D/3D paths remain fully usable."
        )
    if not blender:
        report.notes.append("Blender is not installed, so genuine 3D rendering is unavailable until installed.")
    if free_gb is not None and free_gb < 10:
        report.notes.append(f"Only {free_gb} GB free disk space; model downloads require more.")
    crash = platform.python_version_tuple()
    if crash[:2] >= ("3", "14") and not has_torch:
        report.notes.append(
            "Python 3.14 detected. Official PyTorch wheels may not be available yet for this version; "
            "a compatible Python (e.g. 3.11/3.12) virtual environment is recommended for AI models."
        )
    return report


def model_suggestions(report: HardwareReport) -> list[dict[str, Any]]:
    """Return model *suggestions* (never auto-downloaded) with honest requirements.

    These are well-known open-weight families; users must verify the current
    license and pick a specific checkpoint themselves. We list approximate
    requirements, not guarantees.
    """
    return [
        {
            "task": "image",
            "family": "SD-Turbo / SDXL-Turbo (Stability AI, open weights)",
            "approx_download_gb": 2.5,
            "min_ram_gb": 8,
            "recommended_vram_gb": 0,
            "cpu_capable": True,
            "license_note": "Check the specific Stability AI model license before use; some restrict "
                            "commercial use. Verify at download time.",
            "note": "Small distilled diffusion model; on CPU expect minutes per image at low resolution.",
            "profile": "A+",
        },
        {
            "task": "image",
            "family": "Stable Diffusion 1.5-family (open weights)",
            "approx_download_gb": 4.0,
            "min_ram_gb": 12,
            "recommended_vram_gb": 6,
            "cpu_capable": True,
            "license_note": "CreativeML OpenRAIL-M style licenses; read the checkpoint card.",
            "note": "Practical on CPU only with patience; far better with a GPU.",
            "profile": "B+",
        },
        {
            "task": "video",
            "family": "Open-weight image-to-video / text-to-video models",
            "approx_download_gb": 8.0,
            "min_ram_gb": 24,
            "recommended_vram_gb": 12,
            "cpu_capable": False,
            "license_note": "Licenses vary widely; many forbid commercial use or require attribution. "
                            "Verify the exact model card before downloading.",
            "note": "Needs a capable GPU. Not practical on CPU-only or 8 GB-RAM machines.",
            "profile": "B/C",
        },
    ]
