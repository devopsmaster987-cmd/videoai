"""Shared pytest fixtures for FreeAnimate AI tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from freeanimate.config import Config  # noqa: E402
from freeanimate.languages import LanguageRegistry  # noqa: E402
from freeanimate.project import Project  # noqa: E402


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Config:
    return Config.load(tmp_path / "config.json")


@pytest.fixture()
def languages() -> LanguageRegistry:
    return LanguageRegistry.builtin()


@pytest.fixture()
def projects_root(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def sample_project(projects_root: Path) -> Project:
    return Project.create(
        projects_root,
        "Test Project",
        "Explain the history of ancient Egypt with engaging narration.",
        {"project": "en", "script": "en", "narration": "en", "subtitle": "en", "onscreen": "en"},
        {"duration_minutes": 5, "resolution": "1080p", "fps": 24,
         "animation_mode": "hybrid", "transition": "fade",
         "burn_subtitles": True, "music": "none", "tts_engine": "silence",
         "low_resource": True},
    )


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    from freeanimate.ffmpeg_utils import find_ffmpeg
    return find_ffmpeg("ffmpeg") is not None
