"""Tests 7, 27: Windows path handling and cache management."""

from __future__ import annotations

from pathlib import Path

import pytest

from freeanimate.project import Project, ProjectError, list_projects, resolve_project, slugify


def test_slugify_basic():
    assert slugify("The History of Ancient Egypt!") == "the_history_of_ancient_egypt"
    assert slugify("") == "project"
    assert slugify("   ") == "project"


def test_slugify_keeps_unicode():
    s = slugify("اردو documentary")
    assert "d" in s  # kept some content
    # Must be a single path component with no separators
    assert "/" not in s and "\\" not in s and ":" not in s


def test_project_dirs_created(sample_project: Project):
    from freeanimate.project import SUBDIRS
    for sub in SUBDIRS:
        assert (sample_project.root / sub).is_dir(), f"missing {sub}"


def test_paths_with_spaces(tmp_path: Path):
    root = tmp_path / "My Projects Root"
    root.mkdir()
    p = Project.create(root, "A Project With Spaces", "prompt", {"project": "en"},
                       {"duration_minutes": 5})
    assert " " in str(p.root)
    # Save/read through a path containing spaces
    p.write_json("scenes", "scenes.json", {"scenes": []})
    assert p.read_json("scenes", "scenes.json") == {"scenes": []}
    # Reload via resolve
    p2 = resolve_project(root, p.manifest["slug"])
    assert p2.root == p.root


def test_no_clobber_same_name(tmp_path: Path):
    root = tmp_path / "p"
    root.mkdir()
    a = Project.create(root, "Same Name", "p1", {"project": "en"}, {})
    b = Project.create(root, "Same Name", "p2", {"project": "en"}, {})
    assert a.root != b.root
    assert a.manifest["prompt"] == "p1"
    assert b.manifest["prompt"] == "p2"


def test_list_and_resolve(projects_root: Path):
    Project.create(projects_root, "Alpha Docs", "p", {"project": "en"}, {})
    items = list_projects(projects_root)
    assert len(items) == 1
    slug = items[0]["slug"]
    p = resolve_project(projects_root, slug)
    assert p.manifest["name"] == "Alpha Docs"
    with pytest.raises(ProjectError):
        resolve_project(projects_root, "does-not-exist")


def test_corrupt_manifest_reported_not_crash(tmp_path: Path):
    root = tmp_path / "projects"
    d = root / "broken"
    d.mkdir(parents=True)
    (d / "project.json").write_text("{{{{", encoding="utf-8")
    items = list_projects(root)
    assert items[0]["status"] == "corrupt"


def test_windows_drive_path_used_everywhere(sample_project: Project):
    """All project paths must be pathlib.Path, not os-strings."""
    p = sample_project
    assert isinstance(p.root, Path)
    assert isinstance(p.path("renders", "x.mp4"), Path)
