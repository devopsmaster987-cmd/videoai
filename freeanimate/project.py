"""Project directories, manifest, persistence, and resume support.

Every project owns one directory (section 14):

    projects/<slug>/
        project.json      manifest: prompt, languages, settings, status
        script/           script sections + full script
        scenes/           scene manifest (scenes.json) + per-scene assets
        assets/           generated illustrations, diagrams
        narration/        per-scene WAV/MP3 narration + narration.json
        subtitles/        .srt / .vtt
        audio/            mixed audio, music
        renders/          scene segments + final export
        logs/             stage logs
        reports/          final validation report (JSON + TXT)
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_MANIFEST = "project.json"
PROJECT_VERSION = 1

SUBDIRS = (
    "script",
    "scenes",
    "assets",
    "narration",
    "subtitles",
    "audio",
    "renders",
    "thumbnails",
    "logs",
    "reports",
)

# Completion statuses used in the manifest
STATUS_PLANNED = "planned"
STATUS_SCRIPT_READY = "script_ready"
STATUS_SCENES_READY = "scenes_ready"
STATUS_NARRATED = "narrated"
STATUS_RENDERED = "rendered"
STATUS_EXPORTED = "exported"
STATUS_THUMBNAILS = "thumbnails_ready"   # tracked outside STAGE_ORDER (non-blocking)
STATUS_FAILED = "failed"

STAGE_ORDER = (
    STATUS_PLANNED,
    STATUS_SCRIPT_READY,
    STATUS_SCENES_READY,
    STATUS_NARRATED,
    STATUS_RENDERED,
    STATUS_EXPORTED,
)


def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem-safe project slug that keeps Unicode letters/digits."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    if not text:
        text = "project"
    return text[:max_len]


class ProjectError(RuntimeError):
    pass


@dataclass
class Project:
    """In-memory view of a project directory + its manifest."""

    root: Path
    manifest: dict[str, Any]

    # ---------------------------------------------------------------- create
    @classmethod
    def create(
        cls,
        projects_root: Path,
        name: str,
        prompt: str,
        languages: dict[str, str],
        settings: dict[str, Any],
    ) -> "Project":
        slug = slugify(name) or "project"
        root = Path(projects_root) / slug
        if root.exists():
            # Avoid clobbering: append a numeric suffix.
            for i in range(2, 500):
                candidate = Path(projects_root) / f"{slug}_{i}"
                if not candidate.exists():
                    root = candidate
                    slug = candidate.name
                    break
            else:
                raise ProjectError(f"could not allocate a project directory under {projects_root}")
        for sub in SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "project_version": PROJECT_VERSION,
            "name": name,
            "slug": slug,
            "created_at": time.time(),
            "updated_at": time.time(),
            "prompt": prompt,
            "languages": languages,
            "settings": settings,
            "status": STATUS_PLANNED,
            "pipeline_stage": 0,
            "stages": {s: {"done": False, "error": None} for s in STAGE_ORDER},
            "script": None,
            "scenes": None,
            "narration": None,
            "render": None,
            "export": None,
            "errors": [],
            "warnings": [],
            "providers_used": {},
            "methods_used": {},
        }
        proj = cls(root=root, manifest=manifest)
        # The project exists -> the "planned" stage is complete by definition;
        # resume therefore starts at script generation.
        proj.manifest["stages"][STATUS_PLANNED] = {"done": True, "error": None}
        proj.save()
        return proj

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Path | str) -> "Project":
        p = Path(path)
        if p.is_dir():
            manifest_path = p / PROJECT_MANIFEST
            root = p
        else:
            manifest_path = p
            root = p.parent
        if not manifest_path.exists():
            raise ProjectError(f"not a FreeAnimate project (missing {PROJECT_MANIFEST}): {root}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"corrupt project manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or "slug" not in manifest:
            raise ProjectError(f"invalid project manifest: {manifest_path}")
        return cls(root=root, manifest=manifest)

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        self.manifest["updated_at"] = time.time()
        path = self.root / PROJECT_MANIFEST
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic on Windows for small files

    # ---------------------------------------------------------------- paths
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def stage_dir(self, stage: str) -> Path:
        if stage not in SUBDIRS:
            raise KeyError(stage)
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --------------------------------------------------------------- status
    @property
    def status(self) -> str:
        return self.manifest.get("status", STATUS_PLANNED)

    def mark_stage(self, stage: str, *, error: str | None = None) -> None:
        stages = self.manifest.setdefault("stages", {})
        entry = stages.setdefault(stage, {"done": False, "error": None})
        entry["done"] = error is None
        entry["error"] = error
        if error is None:
            if stage in STAGE_ORDER:
                self.manifest["status"] = stage
                self.manifest["pipeline_stage"] = STAGE_ORDER.index(stage) + 1
        else:
            self.manifest["status"] = STATUS_FAILED
            self.add_error(f"{stage}: {error}")
        self.save()

    def stage_done(self, stage: str) -> bool:
        return bool(self.manifest.get("stages", {}).get(stage, {}).get("done"))

    def add_error(self, message: str) -> None:
        errs = self.manifest.setdefault("errors", [])
        errs.append({"ts": time.time(), "message": message})
        self.save()

    def add_warning(self, message: str) -> None:
        warns = self.manifest.setdefault("warnings", [])
        warns.append({"ts": time.time(), "message": message})
        self.save()

    # ------------------------------------------------------- json read/write
    def write_json(self, rel_dir: str, name: str, data: Any) -> Path:
        d = self.stage_dir(rel_dir)
        p = d / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def read_json(self, rel_dir: str, name: str) -> Any | None:
        p = self.root / rel_dir / name
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    # --------------------------------------------------------------- resume
    def resume_point(self) -> str:
        """Return the name of the first stage that still needs work."""
        for stage in STAGE_ORDER:
            if not self.stage_done(stage):
                return stage
        return "done"

    def can_render(self) -> bool:
        return self.stage_done(STATUS_SCENES_READY) or self.stage_done(STATUS_NARRATED)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def list_projects(projects_root: Path) -> list[dict[str, Any]]:
    """List projects under a root directory; tolerate broken entries."""
    out: list[dict[str, Any]] = []
    root = Path(projects_root)
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        manifest = child / PROJECT_MANIFEST
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            out.append({"slug": child.name, "path": str(child), "status": "corrupt", "name": child.name})
            continue
        out.append(
            {
                "slug": data.get("slug", child.name),
                "name": data.get("name", child.name),
                "status": data.get("status", "unknown"),
                "path": str(child),
                "created_at": data.get("created_at"),
                "prompt": (data.get("prompt") or "")[:120],
            }
        )
    return out


def resolve_project(projects_root: Path, ref: str) -> Project:
    """Resolve a project by slug, name, or path."""
    p = Path(ref)
    if p.exists():
        return Project.load(p)
    candidate = Path(projects_root) / ref
    if candidate.exists():
        return Project.load(candidate)
    # Fuzzy: match slug prefix or name
    for info in list_projects(projects_root):
        if info["slug"] == ref or info.get("name") == ref:
            return Project.load(Path(info["path"]))
    for info in list_projects(projects_root):
        if info["slug"].startswith(ref):
            return Project.load(Path(info["path"]))
    raise ProjectError(f"project not found: {ref!r} (try: freeanimate projects list)")


def delete_project(project: Project, *, confirm: bool = False) -> None:
    """Delete a project directory.  Requires explicit confirmation (14)."""
    if not confirm:
        raise ProjectError("refusing to delete a project without confirmation")
    shutil.rmtree(project.root, ignore_errors=False)
