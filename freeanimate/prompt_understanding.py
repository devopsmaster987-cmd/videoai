"""Structured prompt understanding.

Turns a free-form natural-language prompt into validated, editable production
data: project type, characters, dialogue, environment, style, camera plan,
duration, audio requirements, continuity notes, and a fidelity checklist.

This is a deterministic, offline heuristic parser (no LLM, no network).  It is
intentionally transparent: every field it fills is derived from the prompt
text, and anything it cannot determine is reported as a warning rather than
invented.  When a prompt requests a production method the machine cannot
actually run, the parser records a *supported alternative* and explains the
limitation instead of pretending the original request is possible.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Vocabulary tables (data-driven; extend freely)
# ---------------------------------------------------------------------------
STYLE_CUES: dict[str, tuple[str, ...]] = {
    "stylized_3d_cartoon": ("stylized 3d", "3d cartoon", "3d animation", "3d scene",
                            "three-dimensional", "pixar", "disney-style", "cgi cartoon", "cgi"),
    "cartoon_2d": ("2d cartoon", "2d animation", "cartoon", "flat animation", "cel animation"),
    "anime": ("anime", "manga", "japanese animation"),
    "storybook": ("storybook", "children's book", "children's story", "kids story",
                  "bedtime story", "watercolor", "picture book", "nursery rhyme"),
    "cinematic_realistic": ("cinematic", "photorealistic", "photoreal", "realistic",
                            "realistic human", "realistic people", "real people", "lifelike",
                            "live action", "live-action", "natural skin"),
    "fantasy": ("fantasy", "magical", "dragon", "wizard", "enchanted"),
    "science_fiction": ("sci-fi", "science fiction", "futuristic", "cyberpunk", "space age"),
    "educational": ("educational", "explainer", "whiteboard", "diagram"),
    "horror": ("horror", "scary", "eerie", "unsettling", "dark and moody"),
}

# Explicit cues that decide ambiguous mixed prompts (e.g. "realistic 3D humans").
REALISTIC_CUES = (
    "photorealistic", "photoreal", "realistic human", "realistic people", "real people",
    "lifelike", "live action", "live-action", "natural skin", "believable human",
)
THREE_D_CUES = (
    "3d", "3-d", "three-dimensional", "three dimensional", "blender", "cgi",
    "stylized 3d", "3d cartoon", "3d animation", "3d scene",
)

ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "garden": {
        "cues": ("garden", "flower", "flowers", "fountain", "backyard", "rose", "blossom"),
        "elements": ["wooden table", "fountain", "flower beds", "trees", "hedges", "sunlight"],
        "default_time": "day",
    },
    "kitchen": {
        "cues": ("kitchen", "cooking", "stove", "oven", "counter", "frying", "recipe"),
        "elements": ["counter", "stove", "shelves", "pots", "window", "steam"],
        "default_time": "day",
    },
    "beach": {
        "cues": ("beach", "ocean", "sea", "sand", "waves", "shore", "seaside"),
        "elements": ["sand", "waves", "palm trees", "sun", "shells", "horizon"],
        "default_time": "day",
    },
    "forest": {
        "cues": ("forest", "woods", "jungle", "trees", "trail", "clearing"),
        "elements": ["tall trees", "ferns", "path", "moss", "light shafts", "birds"],
        "default_time": "day",
    },
    "space": {
        "cues": ("space", "galaxy", "planet", "stars", "cosmos", "nebula", "spaceship", "orbit"),
        "elements": ["stars", "planets", "nebula", "asteroids", "starfield", "distant sun"],
        "default_time": "night",
    },
    "school": {
        "cues": ("school", "classroom", "teacher", "students", "lesson", "blackboard"),
        "elements": ["desks", "blackboard", "windows", "books", "clock", "maps"],
        "default_time": "day",
    },
    "city": {
        "cues": ("city", "street", "urban", "skyscraper", "downtown", "building", "traffic"),
        "elements": ["buildings", "street", "lamps", "skyline", "windows", "crosswalk"],
        "default_time": "night",
    },
    "castle": {
        "cues": ("castle", "palace", "throne", "kingdom", "fortress", "knight"),
        "elements": ["stone walls", "towers", "banners", "torches", "gate", "courtyard"],
        "default_time": "day",
    },
    "mountains": {
        "cues": ("mountain", "mountains", "peak", "cliff", "valley", "summit", "hill"),
        "elements": ["peaks", "clouds", "snow caps", "valley", "ridge", "sun"],
        "default_time": "day",
    },
    "room": {
        "cues": ("room", "living room", "bedroom", "indoors", "inside", "apartment", "house"),
        "elements": ["furniture", "window", "lamp", "rug", "walls", "door"],
        "default_time": "day",
    },
    "cafe": {
        "cues": ("cafe", "coffee shop", "restaurant", "diner", "bar"),
        "elements": ["tables", "chairs", "counter", "cups", "window", "warm lights"],
        "default_time": "day",
    },
    "park": {
        "cues": ("park", "bench", "playground", "meadow", "grass"),
        "elements": ["benches", "grass", "trees", "path", "sky", "flowers"],
        "default_time": "day",
    },
    "underwater": {
        "cues": ("underwater", "coral", "reef", "submarine", "fish", "deep sea"),
        "elements": ["coral", "bubbles", "fish", "light rays", "sand floor", "kelp"],
        "default_time": "day",
    },
    "snow": {
        "cues": ("snow", "snowy", "winter", "ice", "frozen", "arctic"),
        "elements": ["snow drifts", "bare trees", "ice", "grey sky", "snowflakes", "mountains"],
        "default_time": "day",
    },
    "desert": {
        "cues": ("desert", "dunes", "sandstorm", "oasis", "cactus"),
        "elements": ["dunes", "cactus", "sun", "rocks", "heat haze", "sky"],
        "default_time": "day",
    },
    "horror_mansion": {
        "cues": ("haunted", "mansion", "graveyard", "ghost", "monster", "abandoned"),
        "elements": ["crooked walls", "fog", "dead trees", "dim windows", "moon", "shadows"],
        "default_time": "night",
    },
}

SPECIES_CUES: dict[str, tuple[str, ...]] = {
    "robot": ("robot", "android", "machine", "droid"),
    "animal": ("dog", "cat", "bird", "rabbit", "horse", "wolf", "bear", "fox", "animal"),
    "creature": ("dragon", "monster", "creature", "alien", "fairy", "elf", "goblin", "beast"),
    "human": ("person", "man", "woman", "boy", "girl", "child", "kid", "character", "people", "teacher", "student", "chef", "chef", "king", "queen", "wizard"),
}

GENDER_CUES = {
    "male": (" man", " boy", " father", " king", " he ", " his ", " mr "),
    "female": (" woman", " girl", " mother", " queen", " she ", " her ", " mrs ", " ms "),
}

CLOTHING_RE = re.compile(
    r"\b(wearing|wears|wear|dressed in|in a|in an)\s+([a-zA-Z\- ]{2,40}?)(?:[,.;]|$|\band\b|\bwhile\b|\bwith\b)",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(r"[\"“”'](.+?)[\"“”']")
SPEECH_VERBS = ("says", "said", "tells", "told", "asks", "asked", "replies", "replied",
                "responds", "responded", "shouts", "whispers", "adds", "answers", "explains",
                "laughs", "jokes", "continues")

DURATION_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?)\b",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "a pair of": 2, "a couple of": 2, "both": 2,
    "several": 3, "a few": 2, "multiple": 2,
}

COMMON_CAPITALIZED = {
    "The", "A", "An", "Two", "Three", "One", "This", "That", "They", "He", "She",
    "It", "We", "I", "In", "On", "At", "With", "When", "While", "As", "Then",
    "Create", "Generate", "Make", "Include", "Use", "Both", "Each", "Their",
    "His", "Her", "Our", "Your", "And", "But", "For", "From", "Into", "Over",
}

CONVERSATION_CUES = ("talking", "conversation", "converse", "dialogue", "tells", "telling",
                     "listens", "listening", "responds", "responding", "laughs", "laughing",
                     "asks", "answers", "replies", "speaks", "speaking", "chats", "chatting")

CAMERA_CUES = {
    "wide shot": ("wide shot", "wide angle", "establishing shot"),
    "medium shot": ("medium shot", "mid shot"),
    "close-up": ("close-up", "close up", "closeup"),
    "over-the-shoulder": ("over the shoulder", "over-the-shoulder"),
    "two-shot": ("two shot", "two-shot", "both characters"),
    "tracking": ("tracking", "dolly", "follow"),
    "pan": ("pan", "pans"),
    "zoom": ("zoom", "push in", "pull out"),
    "static": ("static", "locked off", "still camera"),
}

MOOD_CUES = {
    "happy": ("happy", "joy", "smile", "smiles", "cheerful", "funny", "laugh"),
    "sad": ("sad", "cry", "tear", "sorrow"),
    "surprised": ("surprised", "shocked", "astonished"),
    "angry": ("angry", "furious", "rage"),
    "calm": ("calm", "peaceful", "gentle", "relaxed"),
    "excited": ("excited", "energetic", "thrilled"),
}


# ---------------------------------------------------------------------------
# Validated data models
# ---------------------------------------------------------------------------
class CharacterSpec(BaseModel):
    id: str
    name: str
    role: str = "character"                 # speaker | listener | protagonist | ...
    species: str = "human"                  # human | robot | animal | creature
    description: str = ""
    appearance: str = ""
    clothing: str = ""
    color_palette: list[str] = Field(default_factory=list)
    personality: str = ""
    gender: str | None = None
    voice: str | None = None
    reference_images: list[str] = Field(default_factory=list)
    # Script-driven design details.  These are real inputs to the 2D/3D
    # renderers, not documentation-only fields.
    age: int | None = None
    age_group: str = "adult"                # child | teen | young_adult | adult | elderly
    build: str = "average"                  # slim | average | stocky | tall | short
    hair_style: str = "short"               # short | long | curly | ponytail | bun | bald | hat
    accessories: list[str] = Field(default_factory=list)  # glasses | beard | hat | ...
    height_scale: float = 1.0


class DialogueLine(BaseModel):
    speaker: str                            # character id
    text: str
    emotion: str = "neutral"
    action: str | None = None
    source: str = "prompt"                  # prompt | generated


class CameraShot(BaseModel):
    shot: str = "medium shot"
    movement: str = "slow push in"
    composition: str = "rule of thirds"


class SceneBeat(BaseModel):
    scene_id: str
    title: str
    purpose: str = ""
    characters: list[str] = Field(default_factory=list)
    action: str = ""
    dialogue: list[DialogueLine] = Field(default_factory=list)
    environment: str = ""
    camera: CameraShot = Field(default_factory=CameraShot)
    lighting: str = ""
    audio: str = ""
    continuity: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class PromptPlan(BaseModel):
    prompt: str
    title: str = "Untitled"
    project_type: str = "story"             # documentary | story | tutorial | explainer | conversation | ...
    requested_method: str = "animated_story"  # what the prompt asked for / implies
    supported_method: str = "animated_story"  # what will actually run on this machine
    method_note: str = ""

    characters: list[CharacterSpec] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)

    setting: str = ""
    environment_type: str = "generic"
    background_elements: list[str] = Field(default_factory=list)
    time_of_day: str = "day"
    weather: str = ""
    lighting: str = ""
    visual_style: str = "illustration"

    camera_shots: list[CameraShot] = Field(default_factory=list)
    duration_seconds: float = 30.0
    aspect_ratio: str = "16:9"
    audio: dict[str, Any] = Field(default_factory=dict)
    transitions: str = "fade"
    continuity: list[str] = Field(default_factory=list)

    scenes: list[SceneBeat] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parse_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _clamp_duration(seconds: float) -> float:
    # The UI offers up to 45 minutes. Actual conversation length still follows
    # measured speech; long narrated projects are possible but slow on CPU.
    return float(max(4.0, min(2700.0, seconds)))


def _parse_duration_seconds(prompt: str, default: float = 30.0) -> float:
    m = DURATION_RE.search(prompt)
    if not m:
        if re.search(r"\bshort\b", prompt, re.IGNORECASE):
            return 15.0
        return default
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("h"):
        value *= 3600
    elif unit.startswith("m"):
        value *= 60
    return _clamp_duration(value)


_PEOPLE_NOUNS = (r"characters?|people|persons?|friends?|men|women|boys|girls|"
                 r"adults?|children|kids|humans?|actors?|partners?|siblings?|students?|"
                 r"workers?|customers?|couples?|guests?")
_NUMBER_PATTERN = (r"\d+|one|two|three|four|five|six|seven|eight|"
                   r"a pair of|a couple of|a few|several|multiple|both")


def _detect_character_count(prompt: str, found_names: int) -> int:
    """Count *people nouns* rather than any stray number word.

    The old parser returned 1 for "Two men ... One tells a story" because it
    matched the discourse word "One" first.  Count only numerals attached to a
    people noun, then explicit group words, then detected proper names.
    """
    low = prompt.lower()
    m = re.search(rf"\b({_NUMBER_PATTERN})\s+(?:[a-z\-]+\s+){{0,4}}(?:{_PEOPLE_NOUNS})\b", low)
    if m:
        token = m.group(1)
        if token.isdigit():
            return max(1, min(8, int(token)))
        mapped = NUMBER_WORDS.get(token)
        if mapped:
            return mapped
    if re.search(rf"\b(?:a pair of|couple of|both)\b[^.!?]{{0,60}}(?:{_PEOPLE_NOUNS})", low):
        return 2
    # Distinct character nouns ("a girl and a robot", "a man and his dog").
    kinds = re.findall(
        r"\b(girl|boy|man|woman|robot|creature|animal|dog|cat|dragon|wizard|"
        r"king|queen|teacher|student|chef|person|child|kid|adult|human|actor)\b", low)
    if len(set(kinds)) >= 2:
        return min(4, len(set(kinds)))
    if found_names:
        return found_names
    # Plurals imply at least two even when no numeral is present.
    if re.search(rf"\b(?:{_PEOPLE_NOUNS})\b", low) and re.search(
            r"\b(characters|people|friends|men|women|boys|girls|adults|children|kids|humans|actors)\b",
            low):
        return 2
    if re.search(r"\b(character|person|man|woman|boy|girl|child|kid|robot|creature|animal|"
                 r"adult|human|actor)\b", low):
        return 1
    return 0


def _detect_species(context: str, default: str = "human") -> str:
    low = context.lower()
    for species, cues in SPECIES_CUES.items():
        if any(re.search(rf"\b{re.escape(c)}\b", low) for c in cues):
            return species
    return default


def _detect_gender(context: str) -> str | None:
    low = " " + context.lower() + " "
    for gender, cues in GENDER_CUES.items():
        if any(c in low for c in cues):
            return gender
    return None


def _detect_environment(prompt: str) -> tuple[str, str, list[str], str]:
    low = prompt.lower()
    best: tuple[str, int] | None = None
    for name, data in ENVIRONMENTS.items():
        score = sum(1 for cue in data["cues"] if re.search(rf"\b{re.escape(cue)}\b", low))
        if score and (best is None or score > best[1]):
            best = (name, score)
    if best is None:
        return "generic", "", [], "day"
    name = best[0]
    data = ENVIRONMENTS[name]
    setting = name.replace("_", " ")
    # Extract explicit setting phrase if present.
    m = re.search(r"\b(?:in|at|inside|on)\s+(?:a|an|the)?\s*([a-zA-Z\- ]{3,40}?)"
                  r"(?:[,.!;]|\s+(?:with|while|where|and|as)\b|$)", prompt, re.IGNORECASE)
    if m:
        phrase = m.group(1).strip()
        if phrase and len(phrase.split()) <= 5:
            setting = phrase
    return name, setting, list(data["elements"]), data.get("default_time", "day")


def _detect_time_of_day(prompt: str, default: str) -> str:
    low = prompt.lower()
    for key, cues in {
        "night": ("night", "midnight", "starry", "moonlit", "dark"),
        "evening": ("evening", "sunset", "dusk", "twilight"),
        "morning": ("morning", "sunrise", "dawn"),
        "day": ("day", "sunlit", "daylight", "afternoon", "sunny"),
    }.items():
        if any(re.search(rf"\b{re.escape(c)}\b", low) for c in cues):
            return key
    return default


def _detect_weather(prompt: str) -> str:
    low = prompt.lower()
    for w in ("rain", "rainy", "snow", "snowy", "storm", "stormy", "fog", "foggy",
              "sunny", "cloudy", "windy", "clear"):
        if re.search(rf"\b{w}\b", low):
            return w
    return ""


def _detect_lighting(prompt: str, time_of_day: str) -> str:
    low = prompt.lower()
    for key, cues in {
        "warm": ("warm lighting", "golden", "warm light", "sunlit"),
        "cool": ("cool lighting", "blue light", "moonlight", "cold"),
        "dramatic": ("dramatic lighting", "rim light", "high contrast", "moody"),
        "soft": ("soft lighting", "diffused", "gentle light"),
        "bright": ("bright lighting", "daylight", "sunny"),
    }.items():
        if any(c in low for c in cues):
            return key
    return "warm" if time_of_day in ("day", "morning", "evening") else "cool"


def _detect_style(prompt: str) -> str:
    """Choose the visual style, with explicit precedence for mixed prompts.

    A request for "realistic humans" must never be downgraded to a cartoon just
    because the prompt also says "cinematic" or "3D".  Conversely, a "3D cartoon"
    request stays a 3D cartoon even though it mentions 3D.
    """
    low = prompt.lower()
    realistic = sum(1 for c in REALISTIC_CUES if c in low)
    three_d = sum(1 for c in THREE_D_CUES if re.search(rf"(?<![a-z0-9]){re.escape(c)}(?![a-z0-9])", low))
    cartoon = any(c in low for c in ("cartoon", "pixar", "disney", "anime", "stylized"))
    if realistic and not (three_d and cartoon):
        return "cinematic_realistic"
    if three_d and cartoon:
        return "stylized_3d_cartoon"
    if three_d and not realistic:
        return "stylized_3d_cartoon"
    best, best_score = "illustration", 0
    for style, cues in STYLE_CUES.items():
        score = sum(1 for c in cues if c in low)
        if score > best_score:
            best, best_score = style, score
    return best


def _detect_clothing(sentence: str) -> str:
    m = CLOTHING_RE.search(sentence)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()
    if "different clothes" in sentence.lower():
        return "different, distinct outfits"
    if "wear" in sentence.lower():
        return "clothing mentioned in the prompt"
    return ""


def _detect_names(prompt: str) -> list[str]:
    """Find likely proper-name character names. Conservative to avoid noise."""
    names: list[str] = []
    for m in re.finditer(r"\bnamed\s+([A-Z][a-z]{2,})\b", prompt):
        names.append(m.group(1))
    # "X and Y" where both are capitalized non-common words.
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\s+and\s+([A-Z][a-z]{2,})\b", prompt):
        for g in m.groups():
            if g not in COMMON_CAPITALIZED:
                names.append(g)
    # Dialogue tags "Name says:"
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\s*(?::|says|said|asks|replies|responds)\b", prompt):
        g = m.group(1)
        if g not in COMMON_CAPITALIZED:
            names.append(g)
    seen: list[str] = []
    for n in names:
        if n not in seen and n not in COMMON_CAPITALIZED:
            seen.append(n)
    return seen[:8]


def _split_character_description(sentence: str) -> list[str]:
    """Split a multi-character description into per-character chunks."""
    parts = re.split(
        r",\s*and\s+|\s+while\s+|\s+whereas\s+|;\s*|"
        r"\s+and\s+(?=the (?:first|second|other)|one|another)|"
        r"\s+and\s+(?=(?:a|an|the|his|her|their|my)\s+(?:[a-z\-]+\s+){0,2}"
        r"(?:girl|boy|man|woman|robot|creature|animal|dog|cat|dragon|wizard|king|queen|"
        r"teacher|student|chef|person|child|kid|adult|human|actor))|"
        r"\s+(?:one|the first)\s+|\s+(?:the other|the second)\s+",
        sentence, flags=re.IGNORECASE)
    return [p.strip(" ,.;") for p in parts if p.strip(" ,.;")]


def _detect_plural_gender(context: str) -> str | None:
    low = " " + context.lower() + " "
    if re.search(r"\b(men|boys|males|gentlemen|fathers|brothers)\b", low):
        return "male"
    if re.search(r"\b(women|girls|females|ladies|mothers|sisters)\b", low):
        return "female"
    return None


def _natural_palettes() -> list[list[str]]:
    """Plausible skin/hair/clothing palettes for realistic human requests."""
    return [
        ["#e8b98a", "#2b1d14", "#4f6f8f", "#f2f2f2"],
        ["#c99b74", "#1f1712", "#7a4b3a", "#e8e2d6"],
        ["#f0d0b0", "#5a3a24", "#4b5d67", "#dadfe3"],
        ["#8d5a3b", "#15110f", "#a35d3b", "#eeeeee"],
        ["#d8a87a", "#3b2a1f", "#2f4858", "#f7f1e3"],
        ["#b57b50", "#241812", "#5d6b3f", "#e9e4da"],
    ]


_APPEARANCE_WORDS = (
    "young adult", "young", "middle-aged", "old", "elderly", "tall", "short", "slim",
    "stocky", "round", "athletic", "curly hair", "dark hair", "blonde", "red hair",
    "beard", "glasses", "freckles", "friendly", "cheerful", "serious", "fashionable",
    "stylized", "cartoon", "colorful", "furry", "metallic", "elegant", "scruffy", "cute",
)


_AGE_CUES: dict[str, tuple[str, tuple[str, ...]]] = {
    "child": ("child", ("child", "children", "kid", "kids", "boy", "girl", "toddler",
                        "young child", "schoolchild", "little")),
    "teen": ("teen", ("teen", "teenager", "teenage", "adolescent", "high school")),
    "young_adult": ("young adult", ("young adult", "young man", "young woman", "student",
                                   "twenties", "college")),
    "elderly": ("elderly", ("elderly", "old man", "old woman", "grandmother", "grandfather",
                           "grandma", "grandpa", "senior", "aged", "retired")),
    "adult": ("adult", ("adult", "man", "woman", "mother", "father", "teacher", "chef",
                        "worker", "parent")),
}


def _detect_age(context: str, default: str = "adult") -> tuple[str, int | None, float]:
    low = " " + context.lower() + " "
    for group, (_label, cues) in _AGE_CUES.items():
        if any(re.search(rf"\b{re.escape(c)}\b", low) for c in cues):
            age = {"child": 8, "teen": 15, "young_adult": 24,
                   "adult": 35, "elderly": 72}.get(group)
            scale = {"child": 0.72, "teen": 0.88, "young_adult": 1.0,
                     "adult": 1.0, "elderly": 0.97}.get(group, 1.0)
            return group, age, scale
    m = re.search(r"\b(\d{1,2})[- ]?(?:years?|yrs?)[- ]?old\b", low)
    if m:
        years = int(m.group(1))
        if years < 13:
            return "child", years, 0.62 + years * 0.012
        if years < 20:
            return "teen", years, 0.9
        if years >= 65:
            return "elderly", years, 0.97
        return "young_adult" if years < 30 else "adult", years, 1.0
    return default, None, 1.0


def _detect_build(context: str) -> str:
    low = context.lower()
    for build, cues in {
        "slim": ("slim", "thin", "slender", "lean"),
        "stocky": ("stocky", "heavy", "broad", "muscular", "strong"),
        "tall": ("tall", "lanky"),
        "short": ("short", "small", "petite"),
    }.items():
        if any(re.search(rf"\b{re.escape(c)}\b", low) for c in cues):
            return build
    return "average"


def _detect_hair_style(context: str) -> str:
    low = context.lower()
    for style, cues in {
        "long": ("long hair", "long-haired", "shoulder-length"),
        "curly": ("curly", "afro", "wavy hair"),
        "ponytail": ("ponytail", "pigtail", "braid", "braids"),
        "bun": ("bun", "top knot"),
        "bald": ("bald", "shaved head"),
        "hat": ("hat", "cap", "beanie"),
        "short": ("short hair", "cropped"),
    }.items():
        if any(c in low for c in cues):
            return style
    return "short"


def _detect_accessories(context: str) -> list[str]:
    low = context.lower()
    found: list[str] = []
    for item, cues in {
        "glasses": ("glasses", "spectacles", "eyeglasses"),
        "beard": ("beard", "bearded", "moustache", "mustache"),
        "hat": ("hat", "cap", "beanie"),
        "scarf": ("scarf",),
        "backpack": ("backpack", "school bag"),
        "necklace": ("necklace", "pendant"),
        "earrings": ("earrings",),
        "watch": ("watch", "wristwatch"),
    }.items():
        if any(re.search(rf"\b{re.escape(c)}\b", low) for c in cues):
            found.append(item)
    return found


def _extract_characters(prompt: str, count: int, style: str) -> list[CharacterSpec]:
    sentences = _sentences(prompt)
    names = _detect_names(prompt)
    chars: list[CharacterSpec] = []
    if count <= 0:
        return chars

    # Gather only character-relevant sentences, then split into per-person chunks.
    char_sentences = [
        s for s in sentences
        if re.search(r"\b(character|person|man|woman|boy|girl|child|kid|robot|creature|animal|"
                     r"dog|cat|dragon|wizard|king|queen|teacher|student|chef|friend|people|"
                     r"men|women|adults?|actors?|humans?)\b", s, re.IGNORECASE)
    ]
    # Dialogue sentences must never become a character's visual description.
    description_sentences = [
        s for s in char_sentences
        if not re.search(r"\b(says|said|asks|asked|replies|replied|responds|tells|told|"
                         r"whispers|shouts|explains)\b|[\"“”]", s, re.IGNORECASE)
    ]
    joined = " ".join(char_sentences) or prompt
    described = " ".join(description_sentences) or joined
    clothing = _detect_clothing(joined)
    default_species = _detect_species(joined, "human")
    plural_gender = _detect_plural_gender(joined)
    realistic = style in ("cinematic_realistic", "stylized_3d_cartoon")
    palette_options = _natural_palettes() if realistic else [
        ["#f2c9a0", "#5b3a29", "#3b82f6", "#f5f5f5"],
        ["#e8b98a", "#2f2f2f", "#ef4444", "#fde68a"],
        ["#c99b74", "#4b2e1f", "#10b981", "#e5e7eb"],
        ["#f0d0b0", "#1f2937", "#8b5cf6", "#fbcfe8"],
        ["#d8a87a", "#3f3f46", "#f59e0b", "#bae6fd"],
        ["#c58b62", "#27272a", "#06b6d4", "#fca5a5"],
    ]
    chunks = [c for c in _split_character_description(described) if len(c.split()) >= 2]
    wardrobe = ["", "a contrasting outfit", "a different shirt and trousers",
                "a distinct jacket or dress"]
    default_age_group, default_age, default_height = _detect_age(joined)
    hair_rotation = ["short", "curly", "long", "ponytail", "bun", "short"]
    accessory_rotation = [[], ["backpack"], ["glasses"], ["scarf"], ["hat"], ["watch"]]

    for i in range(count):
        # Prefer explicit names, then stable generic labels.
        name = names[i] if i < len(names) else ("Character" if count == 1 else f"Character {i + 1}")
        desc = chunks[i] if i < len(chunks) else (chunks[-1] if chunks else described[:220])
        if len(desc.split()) < 2:
            desc = described[:220]
        if count > 1 and len(chunks) < count:
            desc = re.sub(r"\b(men)\b", "man", desc, flags=re.IGNORECASE)
            desc = re.sub(r"\b(women)\b", "woman", desc, flags=re.IGNORECASE)
            desc = re.sub(r"\b(friends)\b", "friend", desc, flags=re.IGNORECASE)
            desc = re.sub(r"\b(people)\b", "person", desc, flags=re.IGNORECASE)
        gender = (_detect_gender(desc) or plural_gender
                  or (_detect_gender(joined) if count == 1 else None))
        species = _detect_species(desc, "human")
        if species == "human" and not re.search(
                r"\b(man|woman|boy|girl|person|child|kid|adult|human|friend|teacher|"
                r"student|chef|actor)\b", desc, re.IGNORECASE):
            species = default_species
        age_group, age, height_scale = _detect_age(desc, default_age_group)
        if age is None:
            age = default_age
            height_scale = default_height
        build = _detect_build(desc)
        hair_style = _detect_hair_style(desc)
        if hair_style == "short" and count > 1:
            # Give multi-character casts visibly distinct silhouettes.
            hair_style = hair_rotation[i % len(hair_rotation)]
        accessories = _detect_accessories(desc)
        if count > 1:
            for extra in accessory_rotation[i % len(accessory_rotation)]:
                if extra not in accessories:
                    accessories.append(extra)
        appearance_bits = []
        low_desc = desc.lower()
        for phrase in _APPEARANCE_WORDS:
            if phrase in low_desc and phrase not in appearance_bits:
                appearance_bits.append(phrase)
        appearance_bits.extend(x for x in (age_group.replace("_", " "), build,
                                           f"{hair_style} hair") if x)
        appearance_bits.extend(accessories)
        if not appearance_bits:
            appearance_bits.append("natural proportions" if realistic else "friendly expression")
        if species != "human":
            appearance_bits.append(species)
        style_label = style.replace("_", " ")
        appearance = ", ".join(dict.fromkeys(appearance_bits)) + f" ({style_label})"

        char_clothing = clothing
        if "different" in joined.lower() or "contrasting" in joined.lower():
            if i > 0:
                char_clothing = f"{clothing or 'clothing'}; {wardrobe[i % len(wardrobe)]}".strip("; ")
        if i > 0 and not char_clothing:
            char_clothing = wardrobe[i % len(wardrobe)]

        personality = ""
        for mood in ("funny", "friendly", "shy", "bold", "kind", "curious",
                     "grumpy", "cheerful", "wise", "serious"):
            if mood in desc.lower():
                personality = mood
                break
        chars.append(CharacterSpec(
            id=f"char_{i + 1}",
            name=name,
            role="speaker" if i == 0 else "listener",
            species=species,
            description=desc[:400],
            appearance=appearance.strip(),
            clothing=char_clothing,
            color_palette=list(palette_options[i % len(palette_options)]),
            personality=personality,
            gender=gender,
            voice=None,
            age=age,
            age_group=age_group,
            build=build,
            hair_style=hair_style,
            accessories=accessories,
            height_scale=height_scale,
        ))
    return chars


def _extract_dialogue(prompt: str, characters: list[CharacterSpec]) -> list[DialogueLine]:
    """Pull explicit dialogue and attribute each quote to the right speaker."""
    lines: list[DialogueLine] = []
    names = {c.name.lower(): c.id for c in characters}
    ids = [c.id for c in characters]
    if not ids:
        return lines

    # 1) "Name: [emotion] text" line format.
    for raw in prompt.splitlines():
        m = re.match(r"\s*([A-Za-z][A-Za-z0-9 ]{0,28}?)\s*(?:\((\w+)\))?\s*:\s*(.+)", raw)
        if m:
            who, emotion, text = m.group(1).strip(), (m.group(2) or "").lower(), m.group(3).strip()
            cid = names.get(who.lower())
            if cid is None:
                for i, c in enumerate(characters, start=1):
                    if who.lower() in (f"character {i}", f"char {i}",
                                       f"character{i}", c.name.lower()):
                        cid = c.id
                        break
            if cid:
                tag = re.match(r"^\[(\w+)\]\s*(.*)$", text)
                if tag:
                    emotion, text = tag.group(1).lower(), tag.group(2)
                lines.append(DialogueLine(speaker=cid, text=text.strip(),
                                          emotion=emotion or "neutral", source="prompt"))

    if lines:
        return lines

    # 2) Scan every quoted segment in order.  The text between quotes usually
    # contains the attribution ("One says", "The other replies", "Maya asks").
    quote_re = re.compile(r"[\"“]([^\"“”]+)[\"”]")
    matches = list(quote_re.finditer(prompt))
    if not matches:
        return lines

    def speaker_from(context: str, fallback_index: int) -> str:
        low = " " + context.lower() + " "
        # Longest explicit character name wins.
        for name, cid in sorted(names.items(), key=lambda kv: len(kv[0]), reverse=True):
            if name and re.search(rf"\b{re.escape(name)}\b", low):
                return cid
        # Ordinal / anaphoric references.
        if re.search(r"\b(the first|first|one|former)\b", low):
            return ids[0]
        if re.search(r"\b(the second|second|the other|other|latter|next)\b", low):
            return ids[1] if len(ids) > 1 else ids[0]
        # Unique gender reference.
        for gender in ("man", "boy", "male", "father", "brother", "woman",
                       "girl", "female", "mother", "sister"):
            if re.search(rf"\b{gender}\b", low):
                wanted = "male" if gender in ("man", "boy", "male", "father", "brother") else "female"
                matches_gender = [c.id for c in characters if c.gender == wanted]
                if len(matches_gender) == 1:
                    return matches_gender[0]
        # Stable alternation keeps a two-person exchange coherent.
        return ids[fallback_index % len(ids)]

    previous_end = 0
    last_speaker: str | None = None
    for i, m in enumerate(matches):
        prefix = prompt[previous_end:m.start()]
        previous_end = m.end()
        text = m.group(1).strip().strip(".,;")
        if not text:
            continue
        speaker = speaker_from(prefix, i)
        if i > 0 and not re.search(
                r"\b(says|said|asks|asked|replies|replied|responds|told|tells|adds|"
                r"explains|first|second|other|one|man|woman|boy|girl|robot|friend)\b",
                prefix, re.IGNORECASE) and last_speaker:
            speaker = ids[1 - ids.index(last_speaker)] if len(ids) == 2 else last_speaker
        emotion = "neutral"
        emo = re.search(r"\[(\w+)\]|\((\w+)\)", prefix)
        if emo:
            emotion = (emo.group(1) or emo.group(2)).lower()
        lines.append(DialogueLine(speaker=speaker, text=text, emotion=emotion,
                                  source="prompt"))
        last_speaker = speaker
    return lines


def _generate_dialogue(characters: list[CharacterSpec], setting: str, prompt: str) -> list[DialogueLine]:
    """Create a short, honest placeholder conversation when none is supplied.

    Marked ``source='generated'`` so the report never claims the user wrote it.
    """
    if not characters:
        return []
    place = setting or "here"
    if len(characters) == 1:
        a = characters[0]
        return [
            DialogueLine(speaker=a.id, text=f"I have been thinking about {place} all day.",
                         emotion="thoughtful", action="looks around", source="generated"),
            DialogueLine(speaker=a.id, text="There is something I want to try today.",
                         emotion="curious", action="nods", source="generated"),
            DialogueLine(speaker=a.id, text="It might be difficult, but it could also be wonderful.",
                         emotion="hopeful", action="smiles", source="generated"),
            DialogueLine(speaker=a.id, text="Let us see how it turns out.",
                         emotion="happy", action="steps forward", source="generated"),
        ]
    a, b = characters[0], characters[1]
    place = setting or "here"
    topic = "the day"
    m = re.search(r"\b(?:about|telling|story about|talking about|explains?)\s+"
                  r"([a-zA-Z0-9\- ]{3,50}?)(?:[,.!;]|\s+(?:while|and|to)\b|$)", prompt,
                  re.IGNORECASE)
    if m:
        topic = m.group(1).strip()
    low = prompt.lower()
    mood = "funny" if "funny" in low else "surprising" if "surprise" in low else "pleasant"
    # Honest, editable placeholders: they react to the prompt's mood/setting and
    # are marked source=generated.  Turn-taking is explicit and assigned by id.
    lines = [
        DialogueLine(speaker=a.id,
                     text=f"You will not believe the {mood} thing that happened to me today.",
                     emotion="happy", action="leans forward", source="generated"),
        DialogueLine(speaker=b.id, text="Tell me everything. I could use a good story.",
                     emotion="curious", action="smiles", source="generated"),
        DialogueLine(speaker=a.id,
                     text=f"Well, it started this morning, right before I came to the {place}.",
                     emotion="amused", action="gestures with hand", source="generated"),
        DialogueLine(speaker=b.id,
                     text=f"Ha! That is a great story about {topic}. You always find the bright side.",
                     emotion="laughing", action="laughs", source="generated"),
        DialogueLine(speaker=a.id, text="And the best part? It turned the whole day around.",
                     emotion="happy", action="nods", source="generated"),
        DialogueLine(speaker=b.id, text="Then let us make the rest of the day just as good.",
                     emotion="happy", action="raises cup", source="generated"),
    ]
    return lines


def _detect_camera(prompt: str) -> list[CameraShot]:
    low = prompt.lower()
    shots: list[CameraShot] = []
    for shot, cues in CAMERA_CUES.items():
        if any(c in low for c in cues):
            movement = "slow push in"
            if "pan" in shot:
                movement = "smooth pan"
            elif "tracking" in shot:
                movement = "tracking follow"
            elif "zoom" in shot:
                movement = "push in"
            elif "static" in shot:
                movement = "locked off"
            shots.append(CameraShot(shot=shot, movement=movement,
                                    composition="rule of thirds" if "close" not in shot else "centered"))
    if not shots:
        shots = [
            CameraShot(shot="wide shot", movement="slow push in", composition="rule of thirds"),
            CameraShot(shot="two-shot", movement="gentle drift", composition="balanced"),
            CameraShot(shot="close-up", movement="slow zoom", composition="centered"),
        ]
    return shots


def _project_type(prompt: str, has_dialogue: bool) -> str:
    low = prompt.lower()
    if has_dialogue or any(c in low for c in CONVERSATION_CUES):
        return "conversation"
    if re.search(r"\b(story|tale|fable|adventure|once upon)\b", low):
        return "story"
    if re.search(r"\b(documentary|history of|biography)\b", low):
        return "documentary"
    if re.search(r"\b(tutorial|how to|step by step|guide)\b", low):
        return "tutorial"
    if re.search(r"\b(explain|what is|how does|why)\b", low):
        return "explainer"
    if re.search(r"\b(poem|poetry)\b", low):
        return "poem"
    return "story"


def _choose_method(requested_style: str, project_type: str, has_dialogue: bool,
                   available: dict[str, bool]) -> tuple[str, str, str]:
    """Return (requested_method, supported_method, note).

    The supported method is always something the detected hardware can actually
    run.  When a request cannot be met, the note says exactly what will happen.
    ``realistic_3d_human`` is genuine Blender 3D animation with realistic
    proportions/materials — it is *not* labelled photoreal neural video.
    """
    ai_video = available.get("ai_video_generation", False)
    ai_image = available.get("ai_image_generation", False)
    blender = available.get("blender_3d", False)
    fallback = "conversation_2d" if project_type == "conversation" else "animated_story_2d"

    # --- Realistic human requests -----------------------------------------
    if requested_style == "cinematic_realistic":
        if ai_video:
            return ("realistic_human_video", "ai_video",
                    "A genuine local text/image-to-video model is installed and will be used.")
        if blender:
            return ("realistic_human_scene", "blender_3d_realistic",
                    "Blender is available. This produces genuine 3D human animation with "
                    "realistic proportions, skin/fabric materials and cinematic lighting; it is "
                    "not photoreal neural video and the report labels it accordingly.")
        if ai_image and project_type != "conversation":
            return ("realistic_human_images", "ai_still_image_story",
                    "A local diffusion model can generate realistic still images. The result is a "
                    "labelled AI still-image story with camera motion, not AI video.")
        return ("realistic_human_scene", fallback,
                "No local photorealistic video or 3D engine is available, so clearly labelled "
                "2D animation is used instead.")

    # --- Explicit 3D cartoon requests -------------------------------------
    if requested_style == "stylized_3d_cartoon":
        if blender:
            return ("blender_3d_cartoon", "blender_3d_cartoon",
                    "Genuine Blender 3D cartoon rendering is available (armature-rigged procedural "
                    "characters, 3D environments and multi-shot camera).")
        return ("blender_3d_cartoon", fallback,
                "The prompt requests 3D cartoon animation, but Blender is not installed. "
                "Real rigged 2D animation is used and labelled as 2D.")

    # --- Optional genuine AI video ----------------------------------------
    if ai_video and project_type in ("conversation", "story", "conversation_2d"):
        return ("ai_video", "ai_video", "A compatible local video model is available.")

    # --- AI still-image backgrounds (not presented as video) --------------
    if ai_image and project_type != "conversation":
        return ("ai_image", "animated_story_2d",
                "Local AI image generation is available for backgrounds/illustrations; the motion "
                "is procedural camera animation, not AI video.")

    return "animated_story_2d", fallback, ""


def _build_scenes(plan: PromptPlan) -> list[SceneBeat]:
    """Distribute dialogue/action across short scenes with varied camera shots."""
    shots = plan.camera_shots or [CameraShot()]
    scenes: list[SceneBeat] = []
    dialogue = plan.dialogue
    char_ids = [c.id for c in plan.characters]

    if plan.project_type == "conversation" and dialogue:
        # Group dialogue into exchanges of 2-3 lines per scene.
        idx = 0
        scene_no = 1
        while idx < len(dialogue):
            group = dialogue[idx: idx + 2]
            idx += 2
            speakers = sorted({d.speaker for d in group})
            shot = shots[(scene_no - 1) % len(shots)]
            scenes.append(SceneBeat(
                scene_id=f"scene_{scene_no:03d}",
                title="Conversation" if scene_no == 1 else "The Story Continues",
                purpose="dialogue exchange",
                characters=speakers or char_ids,
                action="Characters face each other and react naturally.",
                dialogue=list(group),
                environment=plan.environment_type,
                camera=shot.model_copy(),
                lighting=plan.lighting,
                audio="speech + ambience",
                continuity=["same characters", "same environment", "consistent clothing"],
            ))
            scene_no += 1
        if not scenes:
            scenes.append(SceneBeat(scene_id="scene_001", title="Conversation",
                                    characters=char_ids, environment=plan.environment_type,
                                    camera=shots[0].model_copy()))
        total = plan.duration_seconds
        per = max(3.0, total / len(scenes))
        for s in scenes:
            s.duration_seconds = round(per, 2)
        return scenes

    # Narrated story/documentary/explainer: 4-6 beats.
    n = max(3, min(6, int(round(plan.duration_seconds / 8.0)) or 3))
    purposes = ["establish the setting", "introduce the characters", "develop the action",
                "a turning point", "resolution", "closing thought"]
    titles = ["Setting the Scene", "Meet the Characters", "The Story Unfolds",
              "A Turning Point", "Resolution", "Closing Thoughts"]
    for i in range(n):
        shot = shots[i % len(shots)]
        scenes.append(SceneBeat(
            scene_id=f"scene_{i + 1:03d}",
            title=titles[i % len(titles)],
            purpose=purposes[i % len(purposes)],
            characters=char_ids,
            action=plan.setting or "movement through the scene",
            environment=plan.environment_type,
            camera=shot.model_copy(),
            lighting=plan.lighting,
            audio="narration + ambience",
            continuity=["consistent characters", "consistent environment"],
            duration_seconds=round(plan.duration_seconds / n, 2),
        ))
    return scenes


def _checklist(plan: PromptPlan) -> list[str]:
    items = [
        f"Character count: {len(plan.characters)}",
        f"Characters: {', '.join(c.name for c in plan.characters) or 'none detected'}",
        f"Environment: {plan.environment_type} ({plan.setting or 'as described'})",
        f"Visual style: {plan.visual_style.replace('_', ' ')}",
        f"Production method: {plan.supported_method.replace('_', ' ')}",
        f"Dialogue lines: {len(plan.dialogue)}",
        f"Target duration: {plan.duration_seconds:.0f}s",
        f"Lighting: {plan.lighting}",
    ]
    if plan.background_elements:
        items.append("Background elements: " + ", ".join(plan.background_elements))
    return items


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def understand_prompt(
    prompt: str,
    *,
    available: dict[str, bool] | None = None,
    default_duration: float = 30.0,
) -> PromptPlan:
    """Parse a prompt into a validated :class:`PromptPlan`.

    ``available`` maps capability keys (from :mod:`freeanimate.hardware`) to
    booleans so the plan can select a method that will actually run.
    """
    available = available or {}
    prompt = (prompt or "").strip()
    warnings: list[str] = []
    notes: list[str] = []

    style = _detect_style(prompt)
    env_type, setting, elements, default_time = _detect_environment(prompt)
    time_of_day = _detect_time_of_day(prompt, default_time)
    weather = _detect_weather(prompt)
    lighting = _detect_lighting(prompt, time_of_day)
    duration = _parse_duration_seconds(prompt, default_duration)

    names = _detect_names(prompt)
    count = _detect_character_count(prompt, len(names))
    characters = _extract_characters(prompt, count, style)
    dialogue = _extract_dialogue(prompt, characters)
    project_type = _project_type(prompt, bool(dialogue))

    if project_type == "conversation" and len(characters) < 2:
        # Conversation requested but only one detected — add a second listener.
        if len(characters) == 1:
            base = characters[0]
            characters.append(CharacterSpec(
                id="char_2", name="Character 2", role="listener",
                species=base.species, description="A second conversation partner.",
                appearance=base.appearance, clothing="different outfit",
                color_palette=["#e8b98a", "#2f2f2f", "#ef4444", "#fde68a"],
            ))
            warnings.append("The prompt implies a conversation but named fewer than two characters; "
                            "a second character was added so dialogue can be assigned.")
        else:
            warnings.append("A conversation was requested but no characters were detected; "
                            "a default pair was created.")

    wants_character_animation = (project_type == "conversation" or
                                 style in ("stylized_3d_cartoon", "cinematic_realistic"))
    if wants_character_animation and characters and not dialogue:
        dialogue = _generate_dialogue(characters, setting or env_type, prompt)
        if dialogue:
            notes.append("No explicit dialogue was supplied, so short placeholder lines were "
                         "generated (marked source=generated). Edit them in the scene editor.")
        # Assign first/second speaker roles.
        for line in dialogue:
            for c in characters:
                if c.id == line.speaker:
                    c.role = "speaker"

    if project_type == "conversation" and dialogue:
        words = sum(len(line.text.split()) for line in dialogue)
        spoken = words / 2.6 + max(0, len(dialogue) - 1) * 0.35
        if duration > spoken * 1.8 + 8.0:
            warnings.append(
                f"The requested {duration:.0f}s is longer than the measured dialogue "
                f"(approximately {spoken:.0f}s). Conversation videos are not padded with "
                "repeated or invented speech; the export will be shorter unless you add lines.")

    requested, supported, method_note = _choose_method(style, project_type, bool(dialogue), available)

    # Voice plan: distinct voices per character.
    audio: dict[str, Any] = {
        "narration": True,
        "music": "procedural" if "music" in prompt.lower() else "none",
        "sound_effects": "ambient" if env_type != "generic" else "none",
        "distinct_voices": len({c.id for c in characters}) > 1,
    }
    for c in characters:
        c.voice = c.voice or ("auto" if len(characters) == 1 else f"voice:{c.id}")

    aspect = "16:9"
    low = prompt.lower()
    if "vertical" in low or "portrait" in low or "9:16" in low or "shorts" in low or "tiktok" in low:
        aspect = "9:16"
    elif "square" in low or "1:1" in low:
        aspect = "1:1"

    plan = PromptPlan(
        prompt=prompt,
        title=_derive_title(prompt),
        project_type=project_type,
        requested_method=requested,
        supported_method=supported,
        method_note=method_note,
        characters=characters,
        dialogue=dialogue,
        setting=setting or env_type,
        environment_type=env_type,
        background_elements=elements,
        time_of_day=time_of_day,
        weather=weather,
        lighting=lighting,
        visual_style=style,
        camera_shots=_detect_camera(prompt),
        duration_seconds=duration,
        aspect_ratio=aspect,
        audio=audio,
        transitions="fade",
        continuity=["characters keep the same appearance", "environment stays consistent within a scene"]
        + (["clothing and colors stay consistent"] if characters else []),
        warnings=warnings,
        parse_notes=notes,
    )
    plan.scenes = _build_scenes(plan)
    plan.checklist = _checklist(plan)
    if not characters:
        plan.warnings.append("No characters were detected in the prompt; the project will be a "
                             "narrated visual without character animation.")
    return plan


def _derive_title(prompt: str) -> str:
    first = _sentences(prompt)
    if not first:
        return "Untitled"
    text = first[0]
    # Strip leading imperative verbs.
    text = re.sub(r"^(create|generate|make|produce|build|show)\s+(a|an|the)?\s*", "", text, flags=re.IGNORECASE)
    words = text.split()
    title = " ".join(words[:8]).strip(" .,:;")
    return (title[:60] or "Untitled")


def plan_to_dict(plan: PromptPlan) -> dict[str, Any]:
    """Serialize a plan for JSON APIs (Pydantic already gives clean dicts)."""
    return plan.model_dump()
