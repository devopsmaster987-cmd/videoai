"""Tests for film mode: duration limits, planning, direction, resume and QA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from freeanimate.animation2d import (CharacterPlacement, CharacterRig, DialogueSpan,
                                     SceneAnimation, render_scene_animation)
from freeanimate.characters import CharacterLibrary, ensure_project_characters
from freeanimate.config import Config
from freeanimate.film import (FILM_MAX_DURATION_MIN, FILM_MIN_DURATION_MIN,
                              FilmOptions, FilmPlan, FilmScene, FilmStudio,
                              build_camera_cues, clamp_film_minutes, direct_scene,
                              plan_film)
from freeanimate.project import Project
from freeanimate.prompt_understanding import CharacterSpec, DialogueLine
from freeanimate.subtitles import SubtitleCue, SubtitleTrack

CHILDREN_PROMPT = (
    "Three children play in a detailed garden. Lily runs after a ball, Ben waves "
    "and laughs, and Mia dances near a tree. They walk, approach each other and talk."
)


# ---------------------------------------------------------------------------
# Duration limits (section 15)
# ---------------------------------------------------------------------------
def test_duration_defaults_and_limits():
    assert clamp_film_minutes(None) == (15.0, [])
    minutes, notes = clamp_film_minutes(2)
    assert minutes == FILM_MIN_DURATION_MIN and notes
    minutes, notes = clamp_film_minutes(40)
    assert minutes == FILM_MAX_DURATION_MIN and notes
    assert clamp_film_minutes(12) == (12.0, [])
    assert clamp_film_minutes(90)[0] == FILM_MAX_DURATION_MIN


def test_quality_presets_are_honest():
    res, crf, preset = FilmOptions(quality="pc_friendly").render_settings()
    assert res == "480p" and preset == "ultrafast"
    res, _crf, preset = FilmOptions(quality="high").render_settings()
    assert res == "1080p" and preset not in ("ultrafast",)
    # An explicit resolution overrides the preset.
    assert FilmOptions(quality="pc_friendly", resolution="720p").render_settings()[0] == "720p"


# ---------------------------------------------------------------------------
# Director + camera
# ---------------------------------------------------------------------------
def test_director_detects_actions_and_props():
    actions, props = direct_scene(
        1, "Lily runs after the ball and waves. Ben walks over and laughs.",
        ["a", "b"], on_screen=["a", "b"])
    verbs = {c["action"] for c in actions}
    assert {"run", "wave", "walk", "laugh"} <= verbs
    assert [p["kind"] for p in props] == ["ball"]
    assert all(0.0 <= c["t0"] < c["t1"] <= 1.0 for c in actions)
    # Movement cues must carry a real travel path (no foot sliding by design).
    for cue in actions:
        if cue["action"] in ("run", "walk"):
            assert cue["x_from"] is not None and cue["x_to"] is not None


def test_camera_cues_cover_scene_without_overlap():
    cues = build_camera_cues([(1.0, 4.0, "a"), (5.0, 9.0, "b"), (10.0, 14.0, "a")],
                             20.0, n_chars=2)
    assert cues
    assert cues[0].start == 0.0
    assert abs(cues[-1].end - 20.0) < 0.01
    covered = sum(c.end - c.start for c in cues)
    assert covered >= 19.5
    for prev, nxt in zip(cues, cues[1:]):
        assert nxt.start >= prev.start
        assert nxt.start <= prev.end + 0.01
    assert cues[0].kind == "establishing"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def children_plan(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    film, _plan, _script = plan_film(
        CHILDREN_PROMPT, FilmOptions(duration_minutes=5, quality="pc_friendly"),
        config=Config(), cache_root=cache)
    return film


def test_plan_structure_is_complete(children_plan: FilmPlan):
    assert children_plan.scene_count >= 3
    assert children_plan.target_minutes == 5.0
    assert children_plan.target_seconds == 300.0
    assert children_plan.estimated_seconds > 0
    ids = {c.id for c in children_plan.characters}
    assert "narrator" in ids  # narration is spoken by an explicit narrator
    for scene in children_plan.scenes:
        assert scene.lines, f"scene {scene.index} has no spoken lines"
        assert scene.duration_planned > 0
        for line in scene.lines:
            assert line.speaker in ids
        assert scene.characters and set(scene.characters) <= ids
        for cue in scene.actions:
            assert cue["character_id"] in ids
            assert 0.0 <= cue["t0"] < cue["t1"] <= 1.0
    # The garden prompt yields garden scenes and a ball prop somewhere.
    assert any(s.environment == "garden" for s in children_plan.scenes)
    assert any(p["kind"] == "ball" for s in children_plan.scenes for p in s.props)
    assert children_plan.script["sections"]
    assert children_plan.warnings or children_plan.notes


def test_story_prompt_is_written_from_the_prompt(children_plan: FilmPlan):
    assert children_plan.script["script_source"] == "prompt_story"
    spoken = " ".join(ln.text for s in children_plan.scenes for ln in s.lines)
    assert "garden" in spoken.lower()
    assert "ball" in spoken.lower()
    # Prompt names must reach the cast and the narration.
    names = {c.name for c in children_plan.characters}
    assert {"Lily", "Ben", "Mia"} <= names
    assert "Lily" in spoken and "Ben" in spoken
    # Documentary filler about a single topic noun must not leak into a story.
    assert "in this video" not in spoken.lower()
    assert {s.environment for s in children_plan.scenes} == {"garden"}


def test_story_runtime_tracks_the_request(children_plan: FilmPlan):
    minutes = children_plan.estimated_seconds / 60.0
    assert 4.2 <= minutes <= 5.8


def test_story_scenes_include_character_dialogue(children_plan: FilmPlan):
    speakers = {ln.speaker for s in children_plan.scenes for ln in s.lines}
    assert speakers - {"narrator"}, "story scenes should let characters speak"


def test_plan_respects_user_script(tmp_path, children_plan):
    script = tmp_path / "story.txt"
    script.write_text("# Chapter One\nOnce upon a time three friends met in a garden.\n"
                      "They played together until the sun went down.\n", encoding="utf-8")
    film, _p, parsed = plan_film(
        "A short story.", FilmOptions(duration_minutes=5, script_path=str(script)),
        config=Config(), cache_root=tmp_path / "cache")
    assert parsed.script_source == "user"
    assert film.scenes and film.scenes[0].title.startswith("Chapter One")
    spoken = " ".join(ln.text for s in film.scenes for ln in s.lines)
    assert "once upon a time" in spoken.lower()


# ---------------------------------------------------------------------------
# Measured scene windows
# ---------------------------------------------------------------------------
def test_scene_windows_follow_measured_audio():
    scenes = [
        FilmScene(0, "sc_001", "A", lines=[DialogueLine(speaker="a", text="one")]),
        FilmScene(1, "sc_002", "B", lines=[DialogueLine(speaker="b", text="two")]),
        FilmScene(2, "sc_003", "C", lines=[DialogueLine(speaker="a", text="three")]),
    ]
    from freeanimate.conversation import ConversationResult, RenderedLine
    lines = [
        RenderedLine("a", "A", "one", "neutral", None, 0.3, 1.5, None, "sapi", "v1"),
        RenderedLine("b", "B", "two", "neutral", None, 2.0, 3.4, None, "sapi", "v2"),
        RenderedLine("a", "A", "three", "neutral", None, 4.0, 5.0, None, "sapi", "v1"),
    ]
    convo = ConversationResult(ok=True, lines=lines, total_seconds=5.6)
    windows = FilmStudio._scene_windows(scenes, convo)
    assert len(windows) == 3
    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(5.6)
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
        assert s1 < e1 <= s2 < e2


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------
def _studio(tmp_path) -> FilmStudio:
    return FilmStudio(Config.load(tmp_path / "config.json"),
                      projects_root=tmp_path / "projects",
                      cache_root=tmp_path / "cache")


def test_resume_state_and_fingerprint(tmp_path):
    studio = _studio(tmp_path)
    project = Project.create(studio.projects_root, "Resume", "p",
                             {"project": "en"}, {"film": {}})
    plan = FilmPlan(prompt="p", title="Resume", scenes=[
        FilmScene(0, "sc_001", "A", duration_planned=5.0)])
    opts = FilmOptions()
    fp1 = studio._fingerprint(project, plan, opts)
    studio._save_state(project, {"fingerprint": fp1,
                                 "scenes": {"1": {"status": "done", "frames": 30}},
                                 "stage": "rendering"})
    loaded = studio._load_state(project)
    assert loaded["scenes"]["1"]["status"] == "done"
    assert loaded["fingerprint"] == fp1
    plan.scenes[0].title = "Changed"
    assert studio._fingerprint(project, plan, opts) != fp1


# ---------------------------------------------------------------------------
# Real render + QA
# ---------------------------------------------------------------------------
@pytest.fixture()
def tiny_film(tmp_path):
    from freeanimate.backgrounds import EnvironmentSpec, render_environment

    studio = _studio(tmp_path)
    project = Project.create(studio.projects_root, "Tiny Film", "tiny",
                             {"project": "en"}, {"film": FilmOptions().to_dict()})
    project.write_json("script", "script.json", {"topic": "tiny", "sections": []})

    specs = [CharacterSpec(id="a", name="Ana", age_group="child"),
             CharacterSpec(id="b", name="Bo", age_group="child")]
    characters = ensure_project_characters(
        CharacterLibrary(project.path("assets", "characters")), specs,
        project.manifest["slug"])
    placements = [CharacterPlacement(rig=CharacterRig.from_character(characters[0]),
                                     x_frac=0.38, ground_frac=0.94, height_frac=0.58,
                                     facing=1),
                  CharacterPlacement(rig=CharacterRig.from_character(characters[1]),
                                     x_frac=0.62, ground_frac=0.94, height_frac=0.58,
                                     facing=-1)]
    bg = render_environment(EnvironmentSpec(env_type="garden", time_of_day="day",
                                            weather="", lighting="warm",
                                            elements=["flowers"], seed="tiny"),
                            project.path("assets", "background_scene_01.png"),
                            width=320, height=180)
    duration = 1.6
    anim = SceneAnimation(
        background_path=bg, placements=placements,
        dialogue=[DialogueSpan(character_id="a", text="Hello", start=0.2, end=1.1)],
        duration=duration, fps=12, out_w=320, out_h=180, seed="tiny",
        actions=[__import__("freeanimate.animation2d", fromlist=["ActorCue"]).ActorCue(
            character_id="b", action="walk", start=0.2, end=1.4,
            x_from=0.62, x_to=0.44)],
        camera_shots=[__import__("freeanimate.animation2d",
                                 fromlist=["CameraCue"]).CameraCue(
            kind="establishing", movement="push_in", start=0.0, end=0.6),
            __import__("freeanimate.animation2d", fromlist=["CameraCue"]).CameraCue(
                kind="medium", movement="push_in", start=0.6, end=duration)],
    )
    seg = project.path("renders", "scene_01.mp4")
    result = render_scene_animation(anim, seg, crf=30, preset="ultrafast")
    assert result.ok, result.error
    final = project.path("renders", "final.mp4")
    final.write_bytes(seg.read_bytes())

    srt = project.path("subtitles", "subtitles.srt")
    srt.write_text("1\n00:00:00,200 --> 00:00:01,100\nAna: Hello\n", encoding="utf-8")

    plan = FilmPlan(
        prompt="tiny", title="Tiny Film", target_minutes=5, target_seconds=duration,
        characters=specs,
        scenes=[FilmScene(
            index=0, scene_id="sc_001", title="A", characters=["a", "b"],
            lines=[DialogueLine(speaker="a", text="Hello")],
            actions=[{"character_id": "b", "action": "walk", "t0": 0.125,
                      "t1": 0.875, "x_from": 0.62, "x_to": 0.44}],
            props=[], duration_planned=duration, duration_actual=duration,
            status="done", segment=str(seg), background=str(bg),
            start=0.0, end=duration)],
        script={"sections": []})
    project.write_json("script", "film_plan.json", plan.to_dict())
    return studio, project, plan, final


def test_qa_checks_a_real_render(tiny_film):
    from freeanimate.qa import write_qa_report, run_qa
    from freeanimate.validation import validate_output

    studio, project, plan, final = tiny_film
    validation = validate_output(final, expected_width=320, expected_height=180,
                                 expected_fps=12, require_audio=False)
    qa = run_qa(project, plan, final, validation=validation,
                work_dir=project.path("reports", "qa_frames"))
    names = {c.name for c in qa.checks}
    assert {"scene_segments", "characters_present", "action_motion",
            "foot_sliding", "camera_paths", "export_readable", "subtitles"} <= names
    assert qa.ok, [c.to_dict() for c in qa.failures]
    assert qa.contact_sheet and Path(qa.contact_sheet).exists()
    jp, tp = write_qa_report(qa, project.stage_dir("reports"))
    assert jp.exists() and tp.exists()
    assert json.loads(jp.read_text(encoding="utf-8"))["ok"] is True


def test_preview_renders_a_real_mp4(tiny_film):
    studio, project, plan, _final = tiny_film
    result = studio.preview(project.manifest["slug"],
                            FilmOptions(preview_seconds=4.0, quality="pc_friendly"))
    assert result.ok, result.error
    assert result.output and result.output.exists() and result.output.stat().st_size > 1024
    assert result.perception and result.perception["measured_fps"] > 0
    assert result.estimate_seconds > 0


def test_status_reports_scene_progress(tiny_film):
    studio, project, plan, _final = tiny_film
    info = studio.status(project.manifest["slug"])
    assert info["scene_count"] == 1
    assert info["scenes_done"] == 0  # no state file yet
    assert info["scenes"][0]["title"] == "A"
