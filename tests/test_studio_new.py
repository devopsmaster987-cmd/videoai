"""Tests for the conversation studio: planning, lip-sync, jobs, web, honesty."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from freeanimate import blender3d, conversation
from freeanimate.animation2d import (
    CharacterPlacement, CharacterRig, DialogueSpan, SceneAnimation,
    render_preview_frame,
)
from freeanimate.characters import Character
from freeanimate.conversation import parse_dialogue_text, render_conversation
from freeanimate.jobs import JobRunner, JobStore
from freeanimate.lipsync import build_viseme_timeline, mouth_openness, viseme_at
from freeanimate.local_models import download_model, plan_download, registry
from freeanimate.project import Project
from freeanimate.prompt_understanding import CharacterSpec, DialogueLine, understand_prompt
from freeanimate.studio import StudioResult
from freeanimate.tts import TTSResult, VoiceInfo, write_silence_wav

CONVERSATION_PROMPT = (
    "Two distinct stylized cartoon characters are sitting at a wooden table in a "
    "beautiful sunlit garden. One character tells a funny story. The second character "
    "listens, smiles, laughs, and responds. They wear different clothes and use "
    "distinct voices. Include flowers, trees, a fountain, warm lighting and camera shots."
)


def _char(spec_id: str, name: str) -> Character:
    return Character.from_spec(CharacterSpec(id=spec_id, name=name))


def test_prompt_plan_is_a_conversation_with_two_characters():
    plan = understand_prompt(CONVERSATION_PROMPT,
                             available={"blender_3d": False,
                                        "ai_image_generation": False,
                                        "ai_video_generation": False})
    assert plan.project_type == "conversation"
    assert len(plan.characters) >= 2
    assert plan.environment_type == "garden"
    assert plan.dialogue
    assert plan.supported_method == "conversation_2d"
    assert plan.scenes


def test_lipsync_timeline_is_monotonic_and_in_range():
    events = build_viseme_timeline("Hello there, this is a lip sync test.", 3.0)
    assert events
    starts = [e.start for e in events]
    assert starts == sorted(starts)
    assert 0.0 <= events[0].start and events[-1].end <= 3.0 + 1e-6
    for event in events:
        assert 0.0 <= mouth_openness(event.viseme) <= 1.0
    assert viseme_at(events, 0.0)
    assert viseme_at(events, 99.0, default="rest") == "rest"


def test_parse_dialogue_text_matches_names_and_emotions():
    chars = [_char("char_1", "Maya"), _char("char_2", "Leo")]
    lines = parse_dialogue_text("Maya: [happy] Hello there.\nLeo: [curious] Who are you?\n", chars)
    assert [l.speaker for l in lines] == ["char_1", "char_2"]
    assert lines[0].emotion == "happy"
    assert lines[0].text == "Hello there."
    with pytest.raises(ValueError):
        parse_dialogue_text("Nobody: hi", chars)


def test_single_voice_is_reported_honestly(monkeypatch):
    chars = [_char("a", "A"), _char("b", "B")]
    monkeypatch.setattr(conversation, "available_voices", lambda *a, **k: [
        VoiceInfo(id="one", name="One", language="en", engine="sapi")])
    assignments, notes = conversation.assign_voices(chars, language="en")
    assert len(assignments) == 2
    assert all(a.voice_id == "one" for a in assignments)
    assert all(not a.distinct for a in assignments)
    assert any("Only 1 distinct voice" in n for n in notes)


def test_no_installed_voice_means_silence_not_fake_speech(monkeypatch):
    chars = [_char("a", "A")]
    monkeypatch.setattr(conversation, "available_voices", lambda *a, **k: [])
    assignments, notes = conversation.assign_voices(chars, language="zz")
    assert assignments[0].engine == "silence"
    assert any("No local voice" in n for n in notes)


def test_render_conversation_builds_real_timeline(tmp_path, monkeypatch, ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg is required for audio normalization")
    chars = [_char("a", "Maya"), _char("b", "Leo")]
    lines = [
        DialogueLine(speaker="a", text="Hello Leo, how are you today?", emotion="happy"),
        DialogueLine(speaker="b", text="I am fine, thank you for asking!", emotion="curious"),
    ]
    monkeypatch.setattr(conversation, "available_voices", lambda *a, **k: [])

    def fake_synth(text, out_wav, assignment, language, cfg):
        seconds = max(0.4, len(text.split()) * 0.28)
        write_silence_wav(out_wav, seconds, rate=22050)
        return TTSResult(True, out_wav, seconds, "silence", "silence")

    monkeypatch.setattr(conversation, "_synth_line", fake_synth)
    result = render_conversation(lines, chars, tmp_path / "narration", language="en")
    assert result.ok, result.error
    assert result.voice_track and result.voice_track.exists()
    assert len(result.lines) == 2
    assert result.lines[1].start > result.lines[0].end
    assert len(result.spans) == 2 and result.spans[0].visemes
    assert result.subtitles and len(result.subtitles.cues) == 2
    assert result.total_seconds > result.lines[-1].end


def test_blender_missing_is_reported_not_faked(tmp_path, monkeypatch):
    monkeypatch.setattr(blender3d, "find_blender", lambda: blender3d.BlenderStatus(
        False, error="not installed for test"))
    result = blender3d.render_conversation_3d([_char("a", "A")], [], tmp_path, duration=1.0)
    assert result.ok is False
    assert "3D" in (result.error or "") or result.notes


def test_model_download_requires_explicit_confirmation():
    model = registry("image")[0]
    plan = plan_download(model)
    assert plan.approx_gb > 0 and plan.repo_id
    with pytest.raises(PermissionError):
        download_model(model, confirm=False)


def test_job_store_and_runner_lifecycle(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.create("p1", "render", {"x": 1})
    assert job.status == "queued"
    store.mark_running(job.id)
    store.set_progress(job.id, "tts", 0.5, "half")
    mid = store.get(job.id)
    assert mid.status == "running" and mid.progress == pytest.approx(0.5)
    store.mark_done(job.id, {"ok": True})
    done = store.get(job.id)
    assert done.status == "done" and done.result == {"ok": True}
    assert store.counts()["done"] == 1

    runner = JobRunner(store)
    failed = runner.submit("p1", "boom", lambda ctx, p: (_ for _ in ()).throw(RuntimeError("no")))
    runner.wait(failed.id, timeout=5)
    assert store.get(failed.id).status == "failed"

    def slow(ctx, p):
        for i in range(100):
            ctx.progress("work", i / 100, force=True)
            time.sleep(0.02)
        return {}

    running = runner.submit("p1", "slow", slow)
    time.sleep(0.15)
    assert runner.cancel(running.id)
    runner.wait(running.id, timeout=5)
    assert store.get(running.id).status == "cancelled"


def test_animation_preview_renders_png(tmp_path):
    spec = CharacterSpec(id="a", name="Maya", color_palette=["#f2c9a0", "#5b3a29", "#3b82f6", "#fff"])
    char = Character.from_spec(spec)
    bg = tmp_path / "bg.png"
    Image.new("RGB", (640, 360), (110, 160, 90)).save(bg)
    anim = SceneAnimation(
        background_path=bg,
        placements=[CharacterPlacement(rig=CharacterRig.from_character(char), x_frac=0.5)],
        dialogue=[DialogueSpan(character_id="a", text="Hello!", start=0.0, end=1.5,
                               visemes=build_viseme_timeline("Hello!", 1.5))],
        duration=1.5, fps=24, out_w=320, out_h=180, show_table=False, title=None,
    )
    out = render_preview_frame(anim, tmp_path / "preview.png", t=0.5)
    assert out.exists() and out.stat().st_size > 1000
    with Image.open(out) as im:
        assert im.size == (320, 180)


def test_web_pages_and_apis(tmp_path):
    from fastapi.testclient import TestClient
    from freeanimate.web import create_app

    app = create_app("config/config.json", projects_root=tmp_path / "projects",
                     cache_root=tmp_path / "cache")
    client = TestClient(app)
    for url in ("/", "/new", "/hardware", "/models", "/characters", "/api/health"):
        assert client.get(url).status_code == 200
    plan = client.post("/api/plan", json={"prompt": CONVERSATION_PROMPT}).json()
    assert plan["project_type"] == "conversation"
    assert plan["characters"] and plan["supported_method"] == "conversation_2d"
    assert client.post("/api/models/plan", json={"id": "sd_turbo"}).status_code == 200
    assert client.get("/api/voices?language=en").status_code == 200


def test_web_job_flow_marks_success(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from freeanimate.web import create_app

    app = create_app("config/config.json", projects_root=tmp_path / "projects",
                     cache_root=tmp_path / "cache")
    client = TestClient(app)

    def fake_run(self, prompt, options=None, *, progress=None, cancel=None):
        if progress:
            progress("render", 0.5, "half")
            progress("complete", 1.0, "done")
        return StudioResult(ok=True, method="rigged_2d_character_animation",
                            duration=1.0, scene_count=1)

    monkeypatch.setattr(app.state.studio, "run", fake_run.__get__(app.state.studio))
    response = client.post("/api/projects", json={"prompt": CONVERSATION_PROMPT,
                                                  "resolution": "480p"})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["result"]["method"] == "rigged_2d_character_animation"


def test_settings_page_validates_and_saves_real_config(tmp_path):
    from fastapi.testclient import TestClient
    from freeanimate.web import create_app

    config_path = tmp_path / "config.json"
    app = create_app(config_path, projects_root=tmp_path / "projects",
                     cache_root=tmp_path / "cache")
    client = TestClient(app)
    assert client.get("/settings").status_code == 200
    assert client.post("/api/settings", json={"updates": {"defaults.fps": 999}}).status_code == 400
    assert client.post("/api/settings",
                       json={"updates": {"defaults.not_a_setting": 1}}).status_code == 400
    response = client.post("/api/settings", json={"updates": {
        "defaults.wpm": 155, "defaults.resolution": "480p",
        "defaults.burn_subtitles": False, "render.crf": 24,
    }})
    assert response.status_code == 200, response.text
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["defaults"]["wpm"] == 155
    assert saved["defaults"]["resolution"] == "480p"
    assert saved["defaults"]["burn_subtitles"] is False
    assert saved["render"]["crf"] == 24


def test_export_zip_contains_real_project_artifacts(tmp_path):
    from fastapi.testclient import TestClient
    from freeanimate.web import create_app

    app = create_app(tmp_path / "config.json", projects_root=tmp_path / "projects",
                     cache_root=tmp_path / "cache")
    client = TestClient(app)
    project = Project.create(tmp_path / "projects", name="Zip Bundle", prompt="test prompt",
                             languages={"project": "en"}, settings={})
    slug = project.manifest["slug"]
    files = {
        "renders/final.mp4": b"fake-mp4-bytes",
        "subtitles/subtitles.srt": b"1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        "audio/mix.wav": b"RIFFfake",
        "reports/validation_report.json": b"{}",
    }
    for rel, payload in files.items():
        path = project.path(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    response = client.get(f"/api/projects/{slug}/export.zip")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = set(zf.namelist())
    assert "renders/final.mp4" in names
    assert "subtitles/subtitles.srt" in names
    assert "audio/mix.wav" in names
    assert "project.json" in names


def test_scene_rerender_endpoint_wires_partial_render_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from freeanimate.web import create_app

    app = create_app(tmp_path / "config.json", projects_root=tmp_path / "projects",
                     cache_root=tmp_path / "cache")
    client = TestClient(app)
    project = Project.create(tmp_path / "projects", name="Scene Job", prompt="test",
                             languages={"project": "en"}, settings={})
    slug = project.manifest["slug"]
    (project.path("renders")).mkdir(parents=True, exist_ok=True)
    (project.path("renders", "scene_01.mp4")).write_bytes(b"one")
    (project.path("renders", "scene_02.mp4")).write_bytes(b"two")

    calls = {}

    def fake_regenerate(self, slug_arg, index, options=None, *, progress=None, cancel=None):
        calls["slug"] = slug_arg
        calls["index"] = index
        if progress:
            progress("render2d", 0.5, "half")
            progress("complete", 1.0, "done")
        return StudioResult(ok=True, project=project,
                            method="rigged_2d_character_animation",
                            duration=2.0, scene_count=2)

    monkeypatch.setattr(app.state.studio, "regenerate_scene",
                        fake_regenerate.__get__(app.state.studio))
    response = client.post(f"/api/projects/{slug}/scenes/2/rerender", json={})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert calls == {"slug": slug, "index": 2}
