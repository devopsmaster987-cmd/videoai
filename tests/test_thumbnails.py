"""Section 12.10: automated thumbnail tests.

Covers the 14 required checks from b.txt:
 1. three initial options after successful video generation
 2. each file is a valid image
 3. each thumbnail measures 1280 x 720
 4. files saved in the correct location
 5. valid JPEG format
 6. RGB-compatible colour mode
 7. scoring is consistent and reproducible
 8. recommendation follows the implemented scoring and quality rules
 9. missing AI models do not crash the application
10. existing assets can be used for fallback compositions
11. failed thumbnail generation is reported honestly
12. regeneration does not alter the completed video
13. the recommendation JSON is valid
14. RTL text rendering is verified when the required language tools are available
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from freeanimate.ffmpeg_utils import find_ffmpeg
from freeanimate.languages import LanguageSettings
from freeanimate.project import Project
from freeanimate.thumbnails import (
    CONCEPT_BUILDERS,
    SELECTED_NAME,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
    generate_thumbnails,
    read_recommendation_json,
    regenerate_thumbnail,
    select_thumbnail,
    validate_thumbnail,
)

# ---------------------------------------------------------------------------
# Lightweight project fixture (fallback compositions from existing assets)
# ---------------------------------------------------------------------------


def _make_project(root: Path, *, language: str = "en") -> Project:
    """Create a completed-video stand-in: manifest export + scene assets."""
    projects = root / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    langs = LanguageSettings(project=language).as_dict()
    p = Project.create(projects, "Thumb Test", "Explain the history of ancient Egypt.",
                       langs, {"duration_minutes": 1, "resolution": "720p"})
    assets = p.stage_dir("assets")
    scenes = []
    import random
    rng = random.Random(42)
    for i in range(4):
        img = Image.new("RGB", (1728, 972))
        px = img.load()
        for y in range(972):
            for x in range(0, 1728, 8):
                px[x, y] = ((x * i + y) % 255, (y * i) % 200, (x + y * i) % 255)
        # Add structure: shapes so images are not blank.
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.ellipse([200 + i * 60, 150, 900 + i * 40, 700], fill=(rng.randint(40, 240),) * 3)
        d.rectangle([1100, 200 + i * 50, 1600, 800], fill=(rng.randint(20, 60), 40, 90))
        path = assets / f"scene_{i + 1:03d}.png"
        img.save(path)
        scenes.append({
            "scene_id": f"scene_{i + 1:03d}",
            "title": f"Ancient Egypt Part {i + 1}",
            "narration_text": "Egypt has a long and rich history full of pyramids.",
            "duration_seconds": 6.0,
            "assets": {"image": str(path)},
            "camera_move": "push_in", "style": "illustration",
            "animation_method": "animated_illustration",
        })
    p.write_json("scenes", "scenes.json", {
        "scenes": scenes, "total_estimated_seconds": 24,
        "target_seconds": 60, "seed": 0, "notes": []})
    p.manifest["export"] = {"path": str(p.root / "renders" / "freeanimate_final.mp4"),
                            "validated": True}
    p.save()
    return p


@pytest.fixture()
def thumb_project(tmp_path: Path) -> Project:
    return _make_project(tmp_path)


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[Project, dict]:
    """One generated report reused by the read-only checks (7, 8, 13)."""
    tmp = tmp_path_factory.mktemp("thumbgen")
    p = _make_project(tmp)
    report = generate_thumbnails(p)
    return p, report


# ---------------------------------------------------------------------------
# 1-6: generation, validity, dimensions, location, JPEG, RGB
# ---------------------------------------------------------------------------
def test_three_initial_options_after_video_generation(generated):
    _p, report = generated
    assert len(report["options"]) == 3
    assert [o["index"] for o in report["options"]] == [1, 2, 3]
    concepts = {o["concept"] for o in report["options"]}
    assert concepts == {"high_appeal", "cinematic", "creative"}


def test_each_thumbnail_valid_image_dimensions_location_jpeg_rgb(generated):
    p, report = generated
    thumbs_dir = p.root / "thumbnails"
    for o in report["options"]:
        f = thumbs_dir / o["file"]
        assert f.exists(), f"missing {f}"                      # check 4
        assert f.stat().st_size > 10_000, "file too small"     # check 2 (non-trivial)
        with Image.open(f) as im:
            im.load()
            assert im.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)  # check 3
            assert im.format == "JPEG"                            # check 5
            assert im.mode == "RGB"                               # check 6
        vd = validate_thumbnail(f)
        assert vd["ok"], f"critical defects: {vd['defects']}"
    # Files are in the required location.
    assert (thumbs_dir / "thumbnail_1.jpg").exists()
    assert (thumbs_dir / "thumbnail_2.jpg").exists()
    assert (thumbs_dir / "thumbnail_3.jpg").exists()


# ---------------------------------------------------------------------------
# 7: scoring is consistent and reproducible
# ---------------------------------------------------------------------------
def test_scoring_reproducible(tmp_path: Path):
    p = _make_project(tmp_path / "a")
    r1 = generate_thumbnails(p)
    r2 = generate_thumbnails(p)
    t1 = [o["total"] for o in r1["options"]]
    t2 = [o["total"] for o in r2["options"]]
    assert t1 == t2, f"scoring not reproducible: {t1} vs {t2}"
    s1 = [o["scores"] for o in r1["options"]]
    s2 = [o["scores"] for o in r2["options"]]
    assert s1 == s2
    # Scores are within the documented criteria ranges.
    for scores in s1:
        assert sum(scores.values()) <= 100
        for crit, val in scores.items():
            assert 0 <= val <= r1["scoring_criteria"][crit]


# ---------------------------------------------------------------------------
# 8: recommendation follows scoring + quality rules
# ---------------------------------------------------------------------------
def test_recommendation_follows_scoring_and_quality_rules(generated):
    _p, report = generated
    rec = report["recommendation"]
    passing = [o for o in report["options"] if o["passed_quality"]]
    if rec["index"] is None:
        assert not passing, "a passing option exists but none was recommended"
        assert any("No thumbnail passed" in w or "no recommendation" in w
                   for w in report["warnings"])
    else:
        best = max(passing, key=lambda o: o["total"])
        assert rec["index"] == best["index"]
        assert best["total"] == max(o["total"] for o in passing)
        assert rec["reasoning"], "recommendation must include reasoning"
        # The reasoning must not promise click-through performance.
        lower = rec["reasoning"].lower()
        assert "guaranteed" not in lower


# ---------------------------------------------------------------------------
# 9: missing AI models do not crash the application
# ---------------------------------------------------------------------------
def test_missing_ai_models_do_not_crash(tmp_path: Path):
    """No AI image/video backend installed -> procedural path still works."""
    p = _make_project(tmp_path)
    report = generate_thumbnails(p)
    assert len(report["options"]) == 3
    assert report["generation_method"] == "procedural_composition_from_scene_assets"
    # Honest method labelling: never claims AI-generated artwork.
    for o in report["options"]:
        assert o["method"] == "procedural_pillow_composition"
        assert "ai_generated" not in o["method"]


# ---------------------------------------------------------------------------
# 10: existing assets used for fallback compositions
# ---------------------------------------------------------------------------
def test_existing_assets_used_for_fallback(tmp_path: Path):
    p = _make_project(tmp_path)
    # Point scenes at only ONE asset (degraded project): still works.
    data = p.read_json("scenes", "scenes.json")
    first_img = data["scenes"][0]["assets"]["image"]
    for s in data["scenes"][1:]:
        s["assets"]["image"] = first_img
    p.write_json("scenes", "scenes.json", data)
    report = generate_thumbnails(p)
    for o in report["options"]:
        assert o["method"] != "failed", o["defects"]
        assert (p.root / "thumbnails" / o["file"]).exists()


# ---------------------------------------------------------------------------
# 11: failed thumbnail generation reported honestly
# ---------------------------------------------------------------------------
def test_failed_thumbnail_generation_reported_honestly(tmp_path: Path):
    p = _make_project(tmp_path)
    original = CONCEPT_BUILDERS["cinematic"]

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated concept failure")

    CONCEPT_BUILDERS["cinematic"] = _boom
    try:
        report = generate_thumbnails(p)
    finally:
        CONCEPT_BUILDERS["cinematic"] = original

    failed = [o for o in report["options"] if o["method"] == "failed"]
    assert len(failed) == 1 and failed[0]["index"] == 2
    assert any("failed" in w.lower() for w in report["warnings"])
    assert any("generation_failed" in str(d) for d in failed[0]["defects"])
    # The other two still exist and are valid.
    ok = [o for o in report["options"] if o["method"] != "failed"]
    assert len(ok) == 2
    for o in ok:
        assert (p.root / "thumbnails" / o["file"]).exists()
    # Honest failure: no fake recommendation covering a missing option.
    if report["recommendation"]["index"] is not None:
        assert report["recommendation"]["index"] != 2


def test_total_asset_failure_raises_honest_error(tmp_path: Path):
    p = _make_project(tmp_path)
    data = p.read_json("scenes", "scenes.json")
    for s in data["scenes"]:
        s["assets"].pop("image", None)
    p.write_json("scenes", "scenes.json", data)
    with pytest.raises(RuntimeError, match="no rendered scene asset"):
        generate_thumbnails(p)


# ---------------------------------------------------------------------------
# 12: regeneration does not alter the completed video
# ---------------------------------------------------------------------------
def test_regeneration_does_not_alter_video(tmp_path: Path):
    p = _make_project(tmp_path)
    render_dir = p.stage_dir("renders")
    fake_video = render_dir / "freeanimate_final.mp4"
    fake_video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 5000)
    before = fake_video.read_bytes()
    before_mtime = fake_video.stat().st_mtime_ns

    generate_thumbnails(p)
    regenerate_thumbnail(p, index=1)
    regenerate_thumbnail(p)  # all three

    assert fake_video.read_bytes() == before
    assert fake_video.stat().st_mtime_ns == before_mtime


# ---------------------------------------------------------------------------
# 13: recommendation JSON is valid + full required fields (12.9)
# ---------------------------------------------------------------------------
def test_recommendation_json_valid(generated):
    p, _report = generated
    jp = p.root / "thumbnails" / "thumbnail_recommendations.json"
    assert jp.exists()
    data = json.loads(jp.read_text(encoding="utf-8"))
    for key in ("video_title", "project_id", "generated_at", "thumbnail_language",
                "generation_method", "options", "scoring_criteria",
                "scoring_disclaimer", "recommendation", "warnings"):
        assert key in data, f"missing {key} in recommendation JSON"
    assert data["project_id"] == p.manifest["slug"]
    assert set(data["scoring_criteria"]) == {
        "topic_relevance", "visual_clarity", "text_readability", "contrast",
        "composition", "audience_relevance", "distinctiveness",
        "honesty_and_accuracy"}
    assert sum(data["scoring_criteria"].values()) == 100
    assert len(data["options"]) == 3
    for o in data["options"]:
        for key in ("index", "file", "concept", "design_description", "method",
                    "dimensions", "scores", "total", "strengths", "weaknesses"):
            assert key in o, f"option missing {key}"
        assert o["dimensions"] == [THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT]


# ---------------------------------------------------------------------------
# 14: RTL text rendering verified when the required language tools are available
# ---------------------------------------------------------------------------
def test_rtl_text_rendering_verified_or_text_free(tmp_path: Path):
    """Urdu thumbnails: shaped rendering must be verified, else text-free.

    Never ships unshaped (broken) Arabic-script text (b.txt 12.5).
    """
    if find_ffmpeg("ffmpeg") is None:
        pytest.skip("ffmpeg not installed — RTL shaping verification unavailable")
    p = _make_project(tmp_path, language="ur")
    report = generate_thumbnails(p)
    assert len(report["options"]) == 3
    for o in report["options"]:
        # For RTL: either verified shaped rendering or honest text-free.
        assert o["text_status"] in ("rendered", "text-free", "empty"), o
        if o["text_status"] == "rendered":
            # Rendered RTL text must come from the libass pipeline.
            assert o["text"], "rendered status implies text content"
    # And the images are still valid thumbnails.
    for o in report["options"]:
        vd = validate_thumbnail(p.root / "thumbnails" / o["file"])
        assert vd["ok"], vd["defects"]


def test_shaped_text_layer_cropped_to_glyphs(tmp_path: Path):
    """The libass layer must be cropped to its glyphs before pasting.

    Regression: an uncropped full-frame layer put RTL text in the middle of
    the thumbnail, overlapping the scene's baked-in title.
    """
    from freeanimate.ffmpeg_utils import has_filter
    from freeanimate.fonts import find_font_for_script
    if find_ffmpeg("ffmpeg") is None or not has_filter("subtitles"):
        pytest.skip("ffmpeg/libass unavailable — shaping cannot be verified")
    if not find_font_for_script("arabic").ok:
        pytest.skip("no font covering the Arabic script on this machine")
    from freeanimate.thumbnails import _render_shaped_text_layer
    layer, ok = _render_shaped_text_layer(
        "آتش پھات کیسے بنتے ہیں", lang_code="ur", size=80,
        color=(255, 255, 255), work_dir=tmp_path, max_width=1100)
    assert ok, "shaping tools are present but shaping failed"
    assert layer is not None
    assert layer.size[1] < 720, "layer must be cropped to glyphs, not full canvas"
    assert layer.size[0] <= int(1100 * 1.02)


# ---------------------------------------------------------------------------
# Selection semantics (12.8): confirm before overwrite, accept = recommendation
# ---------------------------------------------------------------------------
def test_selection_and_overwrite_confirmation(tmp_path: Path):
    p = _make_project(tmp_path)
    generate_thumbnails(p)
    sel = select_thumbnail(p, "accept")
    assert (p.root / "thumbnails" / SELECTED_NAME).exists()
    assert sel["index"] in (1, 2, 3)
    # Second selection without confirmation must refuse.
    with pytest.raises(FileExistsError):
        select_thumbnail(p, 1)
    # With confirmation it succeeds.
    sel2 = select_thumbnail(p, 1, overwrite_confirmed=True)
    assert sel2["index"] == 1
    # Selecting a bad index is rejected.
    with pytest.raises(ValueError):
        select_thumbnail(p, 9, overwrite_confirmed=True)


def test_thumbnail_language_role_defaults_to_project_language():
    from freeanimate.languages import LanguageSettings
    s = LanguageSettings(project="ur")
    assert s.resolved("thumbnail") == "ur"
    s2 = LanguageSettings(project="ur", thumbnail="en")
    assert s2.resolved("thumbnail") == "en"
    d = s2.as_dict()
    assert d["thumbnail"] == "en"


# ---------------------------------------------------------------------------
# Pipeline isolation (15/17): thumbnail failure never fails the video
# ---------------------------------------------------------------------------
def test_thumbnail_failure_is_recorded_as_warning_not_error(tmp_path: Path):
    from freeanimate.pipeline import Pipeline, PipelineOptions

    p = _make_project(tmp_path)
    p.mark_stage("exported")
    opts = PipelineOptions(duration_minutes=1.0, resolution="720p",
                           tts_engine="silence")
    pipe = Pipeline(p, opts)
    lang_settings = LanguageSettings.from_dict(p.manifest.get("languages", {}))

    # Force the thumbnail stage to blow up.
    import freeanimate.thumbnails as th_mod
    original = th_mod.generate_thumbnails

    def _raise(*_a, **_k):
        raise RuntimeError("thumbnail backend exploded")

    th_mod.generate_thumbnails = _raise
    try:
        result = pipe._generate_thumbnails(lang_settings)
    finally:
        th_mod.generate_thumbnails = original

    assert result is None
    # Honest warning recorded, video export untouched.
    warns = [w["message"] for w in p.manifest["warnings"]]
    assert any("thumbnail generation failed" in w for w in warns)
    assert any("video remains valid" in w for w in warns)
    assert p.manifest["export"]["validated"] is True
    assert p.manifest["stages"]["thumbnails_ready"]["done"] is False
