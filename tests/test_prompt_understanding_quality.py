from freeanimate.animation2d import DialogueSpan
from freeanimate.blender3d import build_shot_plan
from freeanimate.characters import Character
from freeanimate.prompt_understanding import understand_prompt


def test_two_men_are_parsed_as_two_male_characters():
    plan = understand_prompt(
        "Two fictional young adult men sit at a garden cafe. "
        'One says: "Hello there." The other says: "Hi!"')
    assert len(plan.characters) == 2
    assert [c.gender for c in plan.characters] == ["male", "male"]
    assert plan.dialogue[0].speaker != plan.dialogue[1].speaker


def test_ordinal_dialogue_attribution_alternates_correctly():
    plan = understand_prompt(
        "Two friends meet. One says: \"Hello.\" The other says: \"Hi.\" "
        "The first friend says: \"How are you?\" The second friend says: \"Well.\"")
    speakers = [line.speaker for line in plan.dialogue]
    assert speakers == ["char_1", "char_2", "char_1", "char_2"]


def test_girl_and_robot_are_distinct_species():
    plan = understand_prompt("A girl and a robot bake a cake in a cozy kitchen.")
    assert [c.species for c in plan.characters] == ["human", "robot"]


def test_realistic_prompt_selects_realistic_blender_when_available():
    plan = understand_prompt(
        "A cinematic realistic young woman walks through a rainy city street.",
        available={"blender_3d": True})
    assert plan.visual_style == "cinematic_realistic"
    assert plan.supported_method == "blender_3d_realistic"


def test_shot_plan_has_two_shot_and_close_ups():
    chars = [Character(id="a", name="A"), Character(id="b", name="B")]
    spans = [DialogueSpan("a", "Hello there", 0.0, 1.0),
             DialogueSpan("b", "Hi, how are you?", 1.2, 1.0)]
    shots = build_shot_plan(chars, spans, 3.0)
    assert len(shots) >= 3
    assert shots[0]["kind"] == "two"
    assert any(s["kind"] == "speaker" and s["speaker"] == "a" for s in shots)
    assert any(s["kind"] == "speaker" and s["speaker"] == "b" for s in shots)
    assert shots[-1]["end"] == 3.0


def test_3d_story_without_script_gets_placeholder_dialogue():
    plan = understand_prompt(
        "Create a 3D cartoon about a girl and a robot baking a cake in a cozy kitchen.",
        available={"blender_3d": True})
    assert plan.supported_method == "blender_3d_cartoon"
    assert len(plan.dialogue) >= 2
    assert all(line.source == "generated" for line in plan.dialogue)
    assert len(plan.scenes) >= 3
