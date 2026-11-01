"""Script engine: prompt -> structured, duration-aware script.

FreeAnimate AI ships with a deterministic *template* script engine that works
offline with zero API keys.  It expands a user prompt into a sectioned script
whose word budget matches the requested duration and speaking speed.

Honesty rules enforced here:

* The engine records ``script_source = "template"`` so the final report never
  pretends an LLM wrote the text.
* Factual claims are framed as narration, never as invented citations.  When a
  user supplies ``sources`` we attach them verbatim; we never invent sources.
* When a compatible free LLM (e.g. a local Ollama server) is detected, the
  pipeline may use it and record ``script_source = "ollama"`` instead.

An optional user-provided script file always wins over generation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from .constants import DEFAULT_WPM, TARGET_SCENE_SECONDS

# ---------------------------------------------------------------------------
# Content taxonomy — what kind of video is being made
# ---------------------------------------------------------------------------
CONTENT_KINDS = (
    "documentary",
    "science",
    "explainer",
    "story",
    "tutorial",
    "listicle",
    "opinion",
)

# Keyword cues used to guess the content kind from the prompt.
KIND_CUES: dict[str, tuple[str, ...]] = {
    "documentary": ("history of", "documentary", "rise and fall", "ancient", "century", "empire", "civilization", "war", "biography of"),
    "science": ("universe", "big bang", "galaxy", "galaxies", "cosmos", "cosmic", "planet", "star", "physics", "quantum", "evolution", "biology", "chemistry", "climate", "neuroscience", "astronomy", "geology", "science"),
    "explainer": ("explain", "how does", "what is", "what are", "why do", "why is", "science of", "understand", "overview of", "introduction to"),
    "story": ("story", "tale", "adventure", "journey of", "once upon", "mystery", "thriller", "fable", "legend"),
    "tutorial": ("tutorial", "how to", "step by step", "guide", "learn", "course", "setup", "install"),
    "listicle": ("top ", "best ", "list of", "ways to", "reasons why", "tips for"),
    "opinion": ("should", "is it worth", "against", "in defense of", "why you should", "why you shouldn't"),
}

# ---------------------------------------------------------------------------
# Section blueprints per content kind.  Each blueprint item is a "beat" the
# engine fills with topic-aware prose.  ``weight`` scales how many words the
# beat receives relative to other beats.
# ---------------------------------------------------------------------------
@dataclass
class Beat:
    role: str            # machine key, e.g. "hook"
    title_template: str  # on-screen / chapter title, may contain {Topic}
    weight: float = 1.0


BEATS: dict[str, tuple[Beat, ...]] = {
    "documentary": (
        Beat("hook", "{Topic}: The Story Everyone Should Know", 1.2),
        Beat("context", "Setting the Scene", 1.4),
        Beat("origin", "Where It All Began", 1.8),
        Beat("development_1", "The First Great Turning Point", 1.8),
        Beat("development_2", "Conflict, Change, and Momentum", 1.8),
        Beat("development_3", "A Society Transformed", 1.6),
        Beat("peak", "The Height of an Era", 1.6),
        Beat("decline", "Seeds of Decline", 1.4),
        Beat("legacy", "What It Left Behind", 1.4),
        Beat("reflection", "Why It Still Matters", 1.2),
        Beat("closing", "Closing Thoughts", 1.0),
    ),
    "science": (
        Beat("hook", "{Topic}: A Journey Through the Evidence", 1.2),
        Beat("context", "The Scale of the Question", 1.3),
        Beat("origin", "The First Moments", 1.8),
        Beat("development_1", "The First Structures Emerge", 1.8),
        Beat("development_2", "How the Pieces Fit Together", 1.8),
        Beat("development_3", "A Universe of Change", 1.5),
        Beat("evidence", "What the Evidence Shows", 1.6),
        Beat("open_questions", "The Questions That Remain", 1.4),
        Beat("reflection", "Why This Story Still Matters", 1.2),
        Beat("closing", "Closing Thoughts", 1.0),
    ),
    "explainer": (
        Beat("hook", "Understanding {Topic}", 1.2),
        Beat("question", "The Question at the Heart of It", 1.2),
        Beat("basics", "The Basic Idea", 1.6),
        Beat("mechanism_1", "How It Works: The First Piece", 1.8),
        Beat("mechanism_2", "How It Works: The Second Piece", 1.8),
        Beat("example", "A Concrete Example", 1.6),
        Beat("misconceptions", "Common Misconceptions", 1.4),
        Beat("implications", "Why This Matters", 1.4),
        Beat("limits", "Where the Explanation Reaches Its Limits", 1.2),
        Beat("closing", "Bringing It Together", 1.0),
    ),
    "story": (
        Beat("hook", "A Beginning", 1.2),
        Beat("world", "The World of Our Story", 1.4),
        Beat("inciting", "The Incident That Changed Everything", 1.6),
        Beat("rising_1", "First Steps and First Trials", 1.8),
        Beat("rising_2", "The Journey Deepens", 1.8),
        Beat("midpoint", "A Revelation", 1.6),
        Beat("rising_3", "The Darkest Hour", 1.8),
        Beat("climax", "The Climax", 1.8),
        Beat("resolution", "The Aftermath", 1.4),
        Beat("closing", "The Lesson That Remains", 1.0),
    ),
    "tutorial": (
        Beat("hook", "What You Will Build", 1.2),
        Beat("prereqs", "What You Need Before Starting", 1.2),
        Beat("concept", "The Core Concept", 1.5),
        Beat("step_1", "Step One", 1.7),
        Beat("step_2", "Step Two", 1.7),
        Beat("step_3", "Step Three", 1.7),
        Beat("step_4", "Step Four and Beyond", 1.5),
        Beat("pitfalls", "Common Mistakes to Avoid", 1.4),
        Beat("next", "Where to Go Next", 1.2),
        Beat("closing", "Recap", 1.0),
    ),
    "listicle": (
        Beat("hook", "{Topic}", 1.2),
        Beat("framing", "Why This List Matters", 1.2),
        Beat("item_1", "Number One", 1.6),
        Beat("item_2", "Number Two", 1.6),
        Beat("item_3", "Number Three", 1.6),
        Beat("item_4", "Number Four", 1.6),
        Beat("item_5", "Number Five", 1.6),
        Beat("summary", "Looking at the Pattern", 1.4),
        Beat("closing", "Final Words", 1.0),
    ),
    "opinion": (
        Beat("hook", "{Topic}: The Case For and Against", 1.2),
        Beat("stakes", "What Is Actually at Stake", 1.4),
        Beat("steelman", "The Strongest Argument on Each Side", 1.8),
        Beat("evidence_1", "The First Line of Evidence", 1.6),
        Beat("evidence_2", "The Second Line of Evidence", 1.6),
        Beat("counter", "Where the Counter-Argument Bites", 1.6),
        Beat("verdict", "A Careful Verdict", 1.6),
        Beat("closing", "What to Take Away", 1.0),
    ),
}

# ---------------------------------------------------------------------------
# Prose builders.  Each takes (topic, beat, prng-ish seed index, facts) and
# returns 1-3 narration sentences.  Deterministic: the same inputs always
# produce the same text, so projects resume reproducibly.
# ---------------------------------------------------------------------------
_OPENERS = (
    "Let's begin with {topic}.",
    "To understand {topic}, we need to start at the beginning.",
    "{Topic} is a subject worth examining carefully.",
    "Few topics reward a closer look quite like {topic}.",
)

_CONNECTORS = (
    "From here, the picture becomes more interesting.",
    "This is where {topic} starts to reveal its character.",
    "Keep this thread in mind; it returns later in a surprising way.",
    "Each layer we peel back adds context to the last.",
)

_CLOSERS = (
    "That is the shape of {topic} in broad strokes.",
    "We have traced {topic} from its roots to its lasting impact.",
    "And that is why {topic} continues to deserve our attention.",
    "Thank you for spending this time exploring {topic} with me.",
)

_FACT_TEMPLATE = "An important detail to note is that {fact}."


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


_ABSTRACT_STARTS = {
    "history", "story", "rise", "fall", "science", "art", "evolution",
    "origin", "origins", "future", "impact", "life", "world", "universe",
    "development", "idea", "concept", "theory", "physics", "biology",
    "chemistry", "economics", "psychology", "philosophy", "climate",
    "energy", "design", "music", "language", "literature", "religion",
    "war", "trade", "migration", "architecture", "mathematics", "math",
    "technology", "internet", "money", "gravity",
}


def _prose_topic(topic: str) -> str:
    """Topic phrased for use inside a sentence ("the history of X")."""
    t = topic.strip().rstrip(".?!")
    first = t.split(" ", 1)[0]
    if first and first.lower() == first and (first.lower() in _ABSTRACT_STARTS or " of " in t):
        return "the " + t
    return t


_SMALL_TITLE_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "at",
    "by", "in", "of", "on", "to", "up", "as", "via", "per", "from", "into",
    "with", "over", "under", "about", "after", "before", "than", "vs",
}


def _title_case(text: str) -> str:
    words = text.split(" ")
    out: list[str] = []
    after_colon = False
    for i, w in enumerate(words):
        if not w:
            out.append(w)
            continue
        low = w.lower()
        if i > 0 and low in _SMALL_TITLE_WORDS and not after_colon:
            out.append(low)
        elif len(w) > 1 and w.isupper():
            out.append(w)
        else:
            out.append(w[0].upper() + w[1:])
        after_colon = w.endswith(":")
    return " ".join(out)


def _topic_variants(topic: str) -> dict[str, str]:
    t = _prose_topic(topic)
    # "topic" keeps the user's natural casing so proper nouns ("Big Bang",
    # "ancient Egypt") are not flattened mid-sentence.
    return {"topic": t, "Topic": _sentence_case(t), "TOPIC": t.upper()}


def _fill(template: str, topic: str) -> str:
    return template.format(**_topic_variants(topic))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> int:
    # Count Unicode word-ish runs; CJK counts each character as a rough word.
    tokens = re.findall(r"[\w']+|[一-鿿぀-ヿ가-힯]", text, flags=re.UNICODE)
    return len(tokens)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------
@dataclass
class ScriptSection:
    id: str
    beat: str
    title: str
    narration: str
    word_count: int
    visual_hint: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Script:
    topic: str
    kind: str
    language: str          # language code the script is written in
    target_minutes: float
    target_wpm: int
    target_words: int
    actual_words: int
    estimated_seconds: float
    sections: list[ScriptSection]
    script_source: str     # "template" | "ollama" | "user"
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "kind": self.kind,
            "language": self.language,
            "target_minutes": self.target_minutes,
            "target_wpm": self.target_wpm,
            "target_words": self.target_words,
            "actual_words": self.actual_words,
            "estimated_seconds": self.estimated_seconds,
            "script_source": self.script_source,
            "sources": self.sources,
            "warnings": self.warnings,
            "sections": [s.to_dict() for s in self.sections],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Script":
        sections = [ScriptSection(**s) for s in data.get("sections", [])]
        return cls(
            topic=data["topic"],
            kind=data.get("kind", "documentary"),
            language=data.get("language", "en"),
            target_minutes=data.get("target_minutes", 10),
            target_wpm=data.get("target_wpm", DEFAULT_WPM),
            target_words=data.get("target_words", 0),
            actual_words=data.get("actual_words", 0),
            estimated_seconds=data.get("estimated_seconds", 0),
            sections=sections,
            script_source=data.get("script_source", "template"),
            sources=data.get("sources", []),
            warnings=data.get("warnings", []),
        )

    def full_text(self) -> str:
        return "\n\n".join(s.narration for s in self.sections)

    def stable_hash(self) -> str:
        """Hash of the narration used to skip unchanged work on resume."""
        joined = "\n".join(s.narration for s in self.sections)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def extract_topic(prompt: str) -> str:
    """Derive a concise topic label from a free-form prompt."""
    p = prompt.strip()
    if not p:
        return "Untitled Topic"
    # Strip leading instruction verbs.
    p = re.sub(
        r"^(please\s+)?(create|make|generate|write|produce|explain|tell me about|讲述|explain to me)\s+(a|an|the)?\s*"
        r"(cinematic\s+|documentary\s+|animated\s+|short\s+|long\s+|video\s+)*"
        r"(documentary|video|explainer|story|guide|tutorial)?\s*(about|on|of|explaining|covering|that explains|which explains)?\s*",
        "",
        p,
        flags=re.IGNORECASE,
    )
    # Cut at first instruction clause about style/formatting.
    p = re.split(r"[.!?]\s+(Use|Include|Add|Make sure|With engaging|Ensure)\b", p)[0]
    p = p.strip(" .!?\"'")
    # Drop a leftover leading article ("the Big Bang" -> "Big Bang").
    p = re.sub(r"^(the|a|an)\s+", "", p, flags=re.IGNORECASE)
    # Keep the leading clause when the prompt carries extra detail.
    if len(p) > 60:
        first = re.split(r",|;|\s+—\s+|\s+-\s+|\band\b", p)[0]
        if 5 <= len(first) <= 60:
            p = first.strip()
    if len(p) > 80:
        p = p[:77].rsplit(" ", 1)[0] + "…"
    return p or "Untitled Topic"


def guess_kind(prompt: str) -> str:
    """Pick the content kind whose keyword cues best match the prompt.

    Ranks by number of matched cues, then by total matched cue length, so a
    specific signal (``big bang``, ``galaxies``) outranks a generic one
    (``documentary``) instead of being lost to dictionary order.
    """
    low = prompt.lower()
    best = "documentary"
    best_key = (0, 0)
    for kind, cues in KIND_CUES.items():
        matched = [c for c in cues if c in low]
        key = (len(matched), sum(len(c) for c in matched))
        if key > best_key:
            best, best_key = kind, key
    return best


def compute_word_budget(minutes: float, wpm: int = DEFAULT_WPM, *, pause_ratio: float = 0.08) -> int:
    """Words needed to fill ``minutes`` of narration at ``wpm``.

    ``pause_ratio`` reserves a small fraction of runtime for breaths, pauses
    and transitions so the engine does not over-stuff the script.
    """
    gross = minutes * wpm
    return max(30, int(round(gross * (1.0 - pause_ratio))))


# Expansion beats: inserted when long durations need more distinct sections
# than the base blueprint provides (one section per ~240 words).  Each has a
# genuinely different scaffold so long scripts are not padded repetition.
_EXPANSION_TITLES = (
    "Deeper Dive: {Topic}",
    "A Closer Look",
    "The Turning Points in Detail",
    "Competing Interpretations",
    "Evidence and Observations",
    "Turning Points Continued",
    "Structures and Systems",
    "Forces and Turning Points",
    "Moments That Changed the Course",
    "Under the Surface",
    "Connections and Consequences",
    "The Wider View",
    "Beyond the Basics",
    "Patterns and Principles",
    "A Wider Frame",
)

_EXPANSION_BODIES = (
    "This deeper look at {topic} separates what is well established from what remains debated, so the picture stays honest.",
    "Here we slow down and examine a single facet of {topic} that is often rushed past in shorter treatments.",
    "To understand {topic} fully, the intermediate steps matter as much as the headline events, and this is where they belong.",
    "Different observers read {topic} differently; weighing those readings side by side sharpens our own view.",
    "What follows is a careful pass through the observations that matter most to the story of {topic}.",
    "Between the major milestones of {topic} sit smaller shifts that quietly set up everything that came after.",
    "Underlying structures shape how {topic} develops, and those structures deserve their own attention.",
    "A handful of pivotal forces drove {topic} forward; this section considers the ones that mattered most.",
    "Some moments in the story of {topic} look ordinary at the time and decisive in retrospect. This is one of them.",
    "Peeling back another layer of {topic} reveals mechanisms that the surface narrative leaves hidden.",
    "Every action around {topic} carries consequences; here we follow a few of them to where they lead.",
    "Behind the surface of {topic} lie the mechanisms that drive it, and understanding them changes how the facts read.",
    "With the foundations set, we can now handle the more demanding parts of {topic} without losing the thread.",
    "Recurring patterns in {topic} give us a working model — useful, imperfect, and worth testing against new cases.",
    "Stepping back to a wider frame shows how {topic} fits into a larger story than any single chapter can hold.",
)


def _select_beats(base: tuple[Beat, ...], needed: int) -> tuple[Beat, ...]:
    """Fit a beat blueprint to an exact section count.

    - needed > len(base): expand with detail beats (interleaved).
    - needed < len(base): keep hook + closing, evenly sample the middle.
    """
    if needed == len(base):
        return base
    if needed > len(base):
        return _expand_beats(base, needed)
    if needed <= 1:
        return (base[0],)
    if needed == 2:
        return (base[0], base[-1])
    mid_needed = needed - 2
    inner = list(base[1:-1])
    if mid_needed >= len(inner):
        picked = inner
    else:
        step = len(inner) / mid_needed
        picked = [inner[min(len(inner) - 1, int(i * step))] for i in range(mid_needed)]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        uniq: list[Beat] = []
        for b in picked:
            if b.role not in seen:
                seen.add(b.role)
                uniq.append(b)
        picked = uniq
    return tuple([base[0]] + picked + [base[-1]])


def _expand_beats(base: tuple[Beat, ...], needed: int) -> tuple[Beat, ...]:
    """Grow a beat blueprint to ``needed`` sections by interleaving expansion
    beats through the middle of the structure (never duplicating roles 1:1)."""
    if needed <= len(base):
        return base
    extra = needed - len(base)
    out: list[Beat] = []
    # Insert expansion beats evenly, skipping hook (first) and closing (last).
    inner = list(base[1:-1])
    per_slot = extra / max(1, len(inner))
    carried = 0.0
    exp_idx = 0
    for b in inner:
        out.append(b)
        carried += per_slot
        while carried >= 1 and exp_idx < extra:
            title = _EXPANSION_TITLES[exp_idx % len(_EXPANSION_TITLES)]
            out.append(Beat(f"detail_{exp_idx + 1}", title, 1.3))
            exp_idx += 1
            carried -= 1
    while exp_idx < extra:
        title = _EXPANSION_TITLES[exp_idx % len(_EXPANSION_TITLES)]
        out.append(Beat(f"detail_{exp_idx + 1}", title, 1.3))
        exp_idx += 1
    return tuple([base[0]] + out + [base[-1]])


# Visual hints used by the asset generator (kept textual, never fabricated).
_VISUAL_BY_BEAT_PREFIX: dict[str, str] = {
    "hook": "title card with topic, cinematic gradient",
    "context": "wide establishing scene, soft parallax layers",
    "origin": "timeline strip, earliest milestone highlighted",
    "development": "process diagram with progressive highlight",
    "peak": "highlight card, strongest single statement",
    "decline": "descending chart motif, muted palette",
    "legacy": "connection diagram, legacy nodes linked",
    "reflection": "quote card, contemplative composition",
    "closing": "closing card, thank-you line, fade",
    "question": "large question mark composition",
    "basics": "simple labeled diagram, three elements",
    "mechanism": "flow diagram, arrows advancing",
    "example": "concrete example card with bullet points",
    "misconceptions": "myth vs fact split layout",
    "implications": "branching implications tree",
    "limits": "boundary frame, cautious palette",
    "prereqs": "checklist card",
    "concept": "concept map, central node emphasized",
    "step": "numbered step card with progress bar",
    "pitfalls": "warning card, caution palette",
    "next": "roadmap strip, next milestones",
    "item": "numbered list card, one item highlighted",
    "framing": "framing statement over abstract background",
    "summary": "summary grid, all items visible",
    "stakes": "balance-scale composition",
    "steelman": "two-column comparison layout",
    "evidence": "evidence board with pinned notes",
    "counter": "counterweight composition, opposing arrows",
    "verdict": "verdict stamp composition",
    "world": "establishing world view, layered silhouettes",
    "inciting": "single dramatic turning-point card",
    "rising": "ascending journey strip",
    "midpoint": "revelation burst composition",
    "climax": "high-contrast climax card",
    "resolution": "settling scene, calm palette",
}


def _visual_hint_for(beat: str) -> str:
    for prefix, hint in _VISUAL_BY_BEAT_PREFIX.items():
        if beat.startswith(prefix):
            return hint
    return "topic illustration with slow camera push"


# ---------------------------------------------------------------------------
# Narration composer
# ---------------------------------------------------------------------------
class TemplateScriptEngine:
    """Offline, deterministic script composer.

    Composes topic-aware narration by combining:

    * user-provided ``facts`` / ``notes`` (used verbatim, never invented),
    * beat-specific prose scaffolds,
    * connective narration that varies by beat index (no copy-paste loops).

    The prose is honest general narration — it does not fabricate dates,
    citations, or statistics.  When the user supplies facts, they are woven in
    with attribution phrasing ("as provided in the project notes").
    """

    def __init__(self, wpm: int = DEFAULT_WPM):
        self.wpm = wpm

    def generate(
        self,
        prompt: str,
        *,
        target_minutes: float,
        language: str = "en",
        facts: Iterable[str] | None = None,
        kind: str | None = None,
    ) -> Script:
        topic = extract_topic(prompt)
        vkind = kind or guess_kind(prompt)
        self._kind = vkind
        base_beats = BEATS.get(vkind, BEATS["documentary"])
        target_words = compute_word_budget(target_minutes, self.wpm)
        # Pick a section count the engine can actually hit:
        #  - each beat has a natural floor of ~35 words (opening + body), so
        #    N <= target/35;
        #  - padding past ~260 words per beat reads as filler, so
        #    N >= ceil(target/260).
        # Within those bounds prefer the base blueprint size.
        max_beats_by_floor = max(1, target_words // 35)
        min_beats_by_pad = max(1, -(-target_words // 260))
        if len(base_beats) <= max_beats_by_floor:
            needed = max(len(base_beats), min_beats_by_pad)
        else:
            needed = max(1, min(max_beats_by_floor, len(base_beats)))
        beats = _select_beats(base_beats, needed)
        fact_list = [f.strip() for f in (facts or []) if f and f.strip()]

        warnings: list[str] = []
        if language != "en":
            warnings.append(
                "Script generated by the offline template engine is English narration text; "
                "set script language to 'en' or supply a translated script for non-English narration."
            )

        total_weight = sum(b.weight for b in beats)
        sections: list[ScriptSection] = []
        used_facts: list[str] = []

        for i, beat in enumerate(beats):
            share = beat.weight / total_weight
            budget = max(20, int(round(target_words * share)))
            narration = self._compose_beat(
                topic=topic,
                beat=beat,
                index=i,
                total=len(beats),
                budget=budget,
                prompt=prompt,
                facts=fact_list,
                used_facts=used_facts,
            )
            sections.append(
                ScriptSection(
                    id=f"sec_{i + 1:03d}",
                    beat=beat.role,
                    title=_title_case(_fill(beat.title_template, topic)),
                    narration=narration,
                    word_count=_words(narration),
                    visual_hint=_visual_hint_for(beat.role),
                )
            )

        actual = sum(s.word_count for s in sections)
        est_seconds = actual / self.wpm * 60.0

        # Trim very long scripts by dropping the least-important trailing
        # connective sentences (never drop user facts).
        if actual > target_words * 1.25 and len(sections) > 4:
            sections, actual, est_seconds = self._trim_sections(sections, target_words)

        unused = [f for f in fact_list if f not in used_facts]
        if unused:
            warnings.append(
                f"{len(unused)} provided fact(s) did not fit into the generated script; "
                "consider editing the script to include them."
            )

        return Script(
            topic=topic,
            kind=vkind,
            language=language,
            target_minutes=target_minutes,
            target_wpm=self.wpm,
            target_words=target_words,
            actual_words=actual,
            estimated_seconds=est_seconds,
            sections=sections,
            script_source="template",
            sources=[],
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    def _compose_beat(
        self,
        *,
        topic: str,
        beat: Beat,
        index: int,
        total: int,
        budget: int,
        prompt: str,
        facts: list[str],
        used_facts: list[str],
    ) -> str:
        parts: list[str] = []

        # Opening sentence varies by position.
        if index == 0:
            parts.append(_fill(_OPENERS[index % len(_OPENERS)], topic))
            parts.append(
                _fill(
                    "In this video we will walk through {topic} step by step — "
                    "what it is, how it developed, and why it matters today.",
                    topic,
                )
            )
        elif beat.role == "closing":
            if getattr(self, "_kind", "") != "science":
                parts.append(_fill(_CLOSERS[index % len(_CLOSERS)], topic))
        else:
            intro = _SECTION_INTROS[index % len(_SECTION_INTROS)]
            parts.append(_fill(intro, topic))

        # Core content: weave in a couple of user facts when available.
        facts_for_beat = [f for f in facts if f not in used_facts][:2]
        for fact in facts_for_beat:
            parts.append(_fill(_FACT_TEMPLATE, topic).replace("{fact}", fact))
            used_facts.append(fact)

        # Beat-specific scaffold prose (deterministic variety by index).
        parts.append(self._beat_body(beat, topic, index, total))
        if beat.role == "closing" and getattr(self, "_kind", "") == "science":
            parts.append(_fill("Thank you for watching.", topic))

        # Connective sentence (skip on last beat).
        if index < total - 1 and budget >= 40:
            parts.append(_fill(_CONNECTORS[index % len(_CONNECTORS)], topic))

        text = " ".join(p.strip() for p in parts if p and p.strip())
        return self._fit_to_budget(text, budget, topic, beat, index)

    def _beat_body(self, beat: Beat, topic: str, index: int, total: int) -> str:
        role = beat.role
        tl = _prose_topic(topic)
        if getattr(self, "_kind", "") == "science" and role in _SCIENCE_BODIES:
            return _sentence_case(_SCIENCE_BODIES[role].format(topic=tl))
        if getattr(self, "_kind", "") == "science" and role == "hook":
            return _sentence_case(
                "We will follow the evidence in order, from the earliest conditions "
                "to the structure we can observe today, and separate what is known "
                "from what is still being worked out."
            )
        # Expansion beats use the rotation of distinct deep-dive scaffolds.
        if role.startswith("detail_"):
            try:
                n = int(role.split("_", 1)[1])
            except ValueError:
                n = index
            body = _EXPANSION_BODIES[(n - 1) % len(_EXPANSION_BODIES)]
            return _sentence_case(body.format(topic=tl))
        templates = {
            "hook": f"Stay with us until the end, because the final part of the story about {tl} is the one most people never hear.",
            "context": f"To place {tl} in context, it helps to zoom out first and look at the bigger picture it belongs to, rather than examining it in isolation.",
            "origin": f"Every story about {tl} has a starting point — a moment when the essential ingredients first came together. Understanding that moment makes everything that follows easier to grasp.",
            "development_1": f"The first major turning point in the story of {tl} changed the rules of the game. From this point onward, nothing quite returned to the way it had been before.",
            "development_2": f"Momentum built through tension. In the case of {tl}, competing pressures pushed developments forward faster than anyone had expected, and the landscape shifted accordingly.",
            "development_3": f"As {tl} evolved, it began to reshape the world around it. New patterns emerged, and people found themselves adapting to a reality that had been redefined.",
            "peak": f"There is a high-water mark in every story — a moment when {tl} reached the height of its influence or visibility. That peak is worth examining closely, because it reveals what was strongest about it.",
            "decline": f"Nothing expands forever. The factors that once drove {tl} forward gradually became constraints, and the same strengths that built it eventually contributed to its transformation.",
            "legacy": f"What {tl} left behind matters more than when it ended. Its legacy survives in ideas, practices, and structures that continue to shape the present.",
            "reflection": f"So why does {tl} still deserve our attention today? Because the patterns it illustrates — growth, tension, adaptation, change — are the same patterns we continue to live through.",
            "closing": f"We have covered a great deal about {tl} in this video. If even one idea here gave you a new way of thinking about it, then this time was well spent.",
            "question": f"Behind the surface of {tl} sits a deceptively simple question. Once you see it clearly, the rest of the subject falls into place.",
            "basics": f"Let us start with the basic idea behind {tl}. Strip away the jargon, and you are left with a small number of core principles that everything else is built on.",
            "mechanism_1": f"The first working part of {tl} is the one that does most of the heavy lifting. Once you understand this piece, the rest of the system makes far more sense.",
            "mechanism_2": f"The second piece connects to the first. Together they form a loop: each one reinforces the other, and the behaviour of the whole becomes greater than its parts.",
            "example": f"To make this concrete, consider a practical example involving {tl}. Walking through a single real scenario clarifies what the abstract description only suggests.",
            "misconceptions": f"It is easy to misunderstand {tl}. Common assumptions — often repeated but rarely examined — can obscure how it actually works, so let us correct a few of them.",
            "implications": f"The implications of {tl} reach further than they first appear. Once you see the connections, you begin to notice them in places you would not have expected.",
            "limits": f"Every explanation has boundaries, and this one is no different. There are deeper layers to {tl} that a single video cannot fully cover, and intellectual honesty requires saying so.",
            "prereqs": f"Before we begin working with {tl}, make sure the foundations are in place. A few minutes of preparation here prevents hours of confusion later.",
            "concept": f"At the centre of {tl} sits one core concept. Hold onto it, and every step that follows becomes a variation on a theme you already understand.",
            "step_1": f"Our first concrete step takes the core idea of {tl} and puts it into practice. Follow along carefully; later steps build directly on this one.",
            "step_2": f"Now we build on that foundation. This second step extends what we have done and introduces the first real decision point in working with {tl}.",
            "step_3": f"The third step is where things start to come together. The pieces from earlier begin to connect, and the overall shape of working with {tl} becomes visible.",
            "step_4": f"From here, the pattern repeats with variations. Once the rhythm is familiar, applying it to new situations involving {tl} becomes second nature.",
            "pitfalls": f"There are mistakes nearly everyone makes when first working with {tl}. Knowing them in advance is the cheapest form of insurance.",
            "next": f"Where you go from here depends on your goals, but the path forward with {tl} is well trodden. The next milestones are clearly marked by those who have walked it before.",
            "item": f"This entry deserves its place on the list for a simple reason: its impact on the broader subject of {tl} is larger than its profile suggests.",
            "framing": f"Before the list begins, it is worth asking why a compilation like this about {tl} matters at all. Framing it correctly changes how every item on the list is understood.",
            "summary": f"Looking at everything together, a pattern emerges. The individual points about {tl} are not isolated facts; they describe a coherent whole.",
            "stakes": f"Questions like the ones surrounding {tl} are rarely abstract. Real decisions, real resources, and real trade-offs sit on the table, which is why careful thinking matters.",
            "steelman": f"Strong arguments exist on more than one side of {tl}. Presenting each side at its strongest — rather than at its easiest to dismiss — is the only honest way to examine it.",
            "evidence_1": f"The first line of evidence concerns outcomes. What has actually happened, as distinct from what was predicted, tells us the most about {tl}.",
            "evidence_2": f"A second line of evidence comes from comparison. Setting {tl} alongside similar cases highlights what is distinctive and what is merely typical.",
            "counter": f"No honest examination of {tl} avoids the strongest counter-argument. Addressed directly, it sharpens rather than destroys the case being made.",
            "verdict": f"Weighing the arguments and evidence together, a careful verdict about {tl} is possible — one that acknowledges uncertainty where it genuinely exists.",
            "world": f"Every story needs a world. The setting around {tl} shapes the choices available to everyone in it, long before the main events begin.",
            "inciting": f"Then comes the moment that sets everything in motion. The incident that launches {tl} looks, in retrospect, both inevitable and completely unexpected.",
            "rising_1": f"The first trials test the resolve at the heart of {tl}. What is learned here — and what is refused to be learned — shapes everything that follows.",
            "rising_2": f"As the journey around {tl} deepens, the stakes grow. Alliances form, assumptions break, and the path ahead becomes harder to predict.",
            "midpoint": f"At the midpoint, a revelation reframes everything. What {tl} appeared to be about turns out to be about something larger.",
            "rising_3": f"The darkest hour arrives when nothing seems to work. It is precisely here that the true character of {tl} is revealed.",
            "climax": f"All the threads converge at the climax. The central tension of {tl} reaches its breaking point, and the outcome is decided.",
            "resolution": f"In the aftermath, a new equilibrium settles in. The world of {tl} is recognisably different from where it began.",
            "reflection_story": f"Stories like {tl} endure because they mirror something true about human experience itself.",
        }
        base = templates.get(role)
        if not base:
            # Prefix-based fallback (development_1, step_2, item_3, ...)
            for prefix, text in templates.items():
                if role.startswith(prefix):
                    base = text
                    break
        if not base:
            base = f"As our examination of {tl} continues, each new angle adds depth to the picture."
        return _sentence_case(base)

    def _fit_to_budget(self, text: str, budget: int, topic: str, beat: Beat, index: int) -> str:
        """Pad or trim narration to approximately ``budget`` words."""
        wc = _words(text)
        if wc >= budget:
            # Trim whole sentences from the end (keep at least two).
            sentences = _split_sentences(text)
            while len(sentences) > 2 and _words(" ".join(sentences)) > budget + int(budget * 0.1):
                sentences.pop()
            return " ".join(sentences)
        # Pad with additional deterministic context sentences.
        pt = _prose_topic(topic)
        pad_sentences = [
            f"Consider for a moment what this means for anyone studying {pt} today.",
            f"It is a detail that repays attention, because small features of {pt} often carry the largest consequences.",
            f"Researchers, students, and curious observers alike return to this aspect of {pt} again and again.",
            f"Placed beside the wider narrative, this element of {pt} takes on a clearer significance.",
            f"Even casual observers of {pt} tend to notice this point first, and for good reason.",
            f"Understanding this part of {pt} makes the remaining chapters far easier to follow.",
            f"The subject offers few better illustrations of this principle than {pt} itself.",
            f"The more carefully one examines {pt}, the more its internal logic becomes visible.",
            f"Anyone following {pt} closely will recognise how this detail connects to the larger arc.",
            f"Set against everything we have seen so far, this dimension of {pt} completes an important part of the picture.",
            f"There is a reason this angle of {pt} keeps resurfacing in serious discussions of the subject.",
            f"Pause here and the pattern behind {pt} becomes considerably easier to see.",
            f"For viewers building their own understanding of {pt}, this is one of the load-bearing pieces.",
            f"Few aspects of {pt} reward the patience of a second look quite like this one.",
            f"It also explains why serious treatments of {pt} rarely skip over this stage.",
            f"Keep this in view; the concluding sections of our story of {pt} will lean on it directly.",
        ]
        sentences = _split_sentences(text)
        i = index
        # Allow padding up to the budget; cap iterations to avoid runaway loops.
        max_pads = max(4, (budget // 30) + 2)
        pads_used = 0
        while _words(" ".join(sentences)) < budget - 4 and pads_used < max_pads:
            pad = pad_sentences[i % len(pad_sentences)]
            sentences.append(pad)
            i += 1
            pads_used += 1
        return " ".join(sentences)

    def _trim_sections(self, sections: list[ScriptSection], target_words: int):
        total = sum(s.word_count for s in sections)
        # Drop trailing connective sentences across sections until within 110%.
        guard = 0
        while total > target_words * 1.1 and guard < 50:
            guard += 1
            changed = False
            for s in sections:
                if total <= target_words * 1.1:
                    break
                sentences = _split_sentences(s.narration)
                if len(sentences) > 3:
                    removed = sentences.pop()
                    s.narration = " ".join(sentences)
                    total -= s.word_count
                    s.word_count = _words(s.narration)
                    total += s.word_count
                    changed = True
            if not changed:
                break
        actual = sum(s.word_count for s in sections)
        est = actual / self.wpm * 60.0
        return sections, actual, est


# ---------------------------------------------------------------------------
# Post-measurement content extension
# ---------------------------------------------------------------------------
_EXTENSION_PADS = (
    "Zooming out, this adds another layer to the overall picture of {topic}.",
    "It is the kind of detail that quietly changes how the rest of the story reads.",
    "For anyone assembling a full understanding of {topic}, this belongs near the centre.",
    "Notice how it connects back to the earlier chapters without repeating them.",
    "The wider implications here are easy to underestimate on a first pass.",
    "Seen from this angle, the sequence of events around {topic} looks more deliberate than accidental.",
    "There are reasonable disagreements about emphasis here, and that nuance matters.",
    "What makes this worth dwelling on is how it echoes elsewhere in the story.",
    "It also gives us a useful checkpoint before moving to the final chapter.",
    "That detail will matter again when we draw the threads together.",
)


def compose_extension_narration(topic: str, target_words: int, seed_index: int = 0) -> str:
    """Compose extra narration that *extends* a script (used when a fast TTS
    voice would otherwise leave the video shorter than requested).

    The sentences come from the deep-dive scaffold rotation plus connective
    pads, so the extension reads as further exploration rather than filler.
    """
    tl = topic.lower()
    sentences: list[str] = []
    i = seed_index
    guard = 0
    while _words(" ".join(sentences)) < max(15, target_words) and guard < 60:
        if len(sentences) % 3 == 0:
            body = _EXPANSION_BODIES[i % len(_EXPANSION_BODIES)].format(topic=tl)
        else:
            body = _EXTENSION_PADS[i % len(_EXTENSION_PADS)].format(topic=tl)
        sentences.append(_sentence_case(body))
        i += 1
        guard += 1
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# User-provided script parsing
# ---------------------------------------------------------------------------
def parse_user_script(text: str, *, language: str = "en", topic: str = "User Script") -> Script:
    """Parse a plain-text script into sections.

    Supported conventions:

    * Lines starting with ``#`` become section titles.
    * Blank lines separate sections.
    * Text without titles is split into paragraphs.
    """
    lines = text.splitlines()
    sections: list[ScriptSection] = []
    current_title: str | None = None
    buffer: list[str] = []

    def flush():
        nonlocal current_title, buffer
        body = " ".join(x.strip() for x in buffer if x.strip()).strip()
        buffer = []
        if body or current_title:
            idx = len(sections) + 1
            title = current_title or f"Section {idx}"
            sections.append(
                ScriptSection(
                    id=f"sec_{idx:03d}",
                    beat="user",
                    title=title,
                    narration=body,
                    word_count=_words(body),
                    visual_hint="user script section illustration",
                )
            )
        current_title = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_title = stripped.lstrip("#").strip() or None
        elif not stripped:
            if buffer and any(x.strip() for x in buffer):
                flush()
        else:
            buffer.append(stripped)
    flush()

    # If there is only one giant section, split by paragraph-ish sentence count.
    if len(sections) == 1 and sections[0].word_count > 300:
        big = sections[0]
        sentences = _split_sentences(big.narration)
        per = 60  # sentences per sub-section
        new_sections: list[ScriptSection] = []
        for i in range(0, len(sentences), per):
            chunk = " ".join(sentences[i : i + per])
            n = len(new_sections) + 1
            new_sections.append(
                ScriptSection(
                    id=f"sec_{n:03d}",
                    beat="user",
                    title=f"{big.title} ({n})" if len(sentences) > per else big.title,
                    narration=chunk,
                    word_count=_words(chunk),
                    visual_hint="user script section illustration",
                )
            )
        sections = new_sections

    actual = sum(s.word_count for s in sections)
    minutes = actual / DEFAULT_WPM
    return Script(
        topic=topic,
        kind="user",
        language=language,
        target_minutes=minutes,
        target_wpm=DEFAULT_WPM,
        target_words=actual,
        actual_words=actual,
        estimated_seconds=actual / DEFAULT_WPM * 60.0,
        sections=sections,
        script_source="user",
    )


# Topic-specific narration scaffolds for the "science" kind.  Kept neutral and
# accurate: they describe structure and evidence, never invented facts.
_SCIENCE_BODIES: dict[str, str] = {
    "context": "To grasp the scale of {topic}, it helps to step back from the details and consider the whole system at once rather than examining one piece in isolation.",
    "origin": "Every account of {topic} starts at a point where the essential conditions first came together. Everything that follows builds on that single moment.",
    "development_1": "The first structures in {topic} formed through processes that repeat at every scale, from the smallest components to the largest patterns we can observe.",
    "development_2": "Understanding {topic} means seeing how its parts interact. Energy, matter, and time combine in ways that let complexity emerge from simple rules.",
    "development_3": "{topic} is not a finished picture but a changing one. As it evolved, new layers of structure appeared, each built on the one before.",
    "evidence": "The evidence for {topic} arrives from several independent directions. It is strongest where those separate lines of inquiry converge on the same conclusion.",
    "open_questions": "An honest account of {topic} ends with open questions as well as answers. The frontier is not a flaw in the story; it is part of what makes the subject alive.",
    "reflection": "Why does {topic} still matter? Because the same processes that shaped it continue to shape the world we can observe today.",
    "closing": "We have followed {topic} from its earliest conditions to the picture we can reconstruct now. The details will keep being refined, but the broad outline is already remarkable.",
}

_SECTION_INTROS = (
    "With that groundwork in place, the next part of the story comes into focus.",
    "The next layer of {topic} is where things get interesting.",
    "This part of {topic} deserves a closer look.",
    "To keep the thread clear, here is the next piece of the picture.",
)
