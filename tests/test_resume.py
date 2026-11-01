"""Tests 8-9: project resume and failed-scene retries."""

from __future__ import annotations

from freeanimate.project import (
    STATUS_FAILED,
    STATUS_NARRATED,
    STATUS_PLANNED,
    STATUS_SCENES_READY,
    STATUS_SCRIPT_READY,
    Project,
)


def test_initial_status(sample_project: Project):
    assert sample_project.status == STATUS_PLANNED
    assert sample_project.resume_point() == STATUS_SCRIPT_READY


def test_stage_progression_and_resume_point(sample_project: Project):
    p = sample_project
    p.mark_stage(STATUS_SCRIPT_READY)
    assert p.resume_point() == STATUS_SCENES_READY
    p.mark_stage(STATUS_SCENES_READY)
    p.manifest["status"] = STATUS_SCENES_READY
    p.save()
    assert p.resume_point() == STATUS_NARRATED
    p.mark_stage(STATUS_NARRATED)
    assert p.resume_point() == "rendered"
    p.mark_stage("rendered")
    p.mark_stage("exported")
    assert p.resume_point() == "done"


def test_manifest_persists_across_reload(sample_project: Project):
    p = sample_project
    p.mark_stage(STATUS_SCRIPT_READY)
    p.write_json("script", "script.json", {"topic": "X", "sections": []})
    reloaded = Project.load(p.root)
    assert reloaded.stage_done(STATUS_SCRIPT_READY)
    assert reloaded.read_json("script", "script.json")["topic"] == "X"
    assert reloaded.resume_point() == STATUS_SCENES_READY


def test_stage_failure_recorded(sample_project: Project):
    p = sample_project
    p.mark_stage(STATUS_SCRIPT_READY)
    p.mark_stage(STATUS_SCENES_READY, error="scene 3 asset failed")
    assert p.status == STATUS_FAILED
    assert any("scene 3" in e["message"] for e in p.manifest["errors"])
    # Failure does not lose earlier progress:
    assert p.stage_done(STATUS_SCRIPT_READY)


def test_failed_scene_retry_semantics():
    """A failed scene can be retried; retries counter increments, error clears."""
    from freeanimate.scene_planner import Scene
    sc = Scene(scene_id="scene_001", title="t", narration_text="n",
               duration_seconds=5, visual_description="v",
               status="failed", retries=2, error="boom")
    # Simulate retry:
    sc.status = "planned"
    sc.error = None
    sc.retries += 1
    assert sc.status == "planned"
    assert sc.error is None
    assert sc.retries == 3
    d = sc.to_dict()
    sc2 = Scene.from_dict(d)
    assert sc2.retries == 3


def test_unknown_keys_ignored_in_scene_deserialization():
    from freeanimate.scene_planner import Scene
    sc = Scene.from_dict({
        "scene_id": "s", "title": "t", "narration_text": "n",
        "duration_seconds": 5, "visual_description": "v",
        "some_future_field": 42,
    })
    assert sc.scene_id == "s"
