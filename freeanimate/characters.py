"""Reusable character library.

Characters persist across projects so the same design (and, for 2D, the same
rig/colour palette) can be reused in later scenes.  Each character stores its
description, appearance, clothing, palette, personality, voice assignment,
reference images, and available rig information.

Reference portraits are *procedurally drawn* 2D artwork (clearly labelled as
such) — not AI-generated portraits.  If a local AI image model is installed the
web UI can call it separately to make additional references, but this module
never fakes that.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .prompt_understanding import CharacterSpec

LIBRARY_FILENAME = "library.json"


def _seed(text: str) -> random.Random:
    return random.Random(int(hashlib.sha256(text.encode()).hexdigest()[:12], 16))


# Curated real-looking human palettes: skin, hair, clothing, highlight.
# Stylized hue-rotated palettes produced green/teal skin, which reads as a bug
# for human characters; robots and creatures keep the generated palette.
_NATURAL_PALETTES: tuple[tuple[str, str, str, str], ...] = (
    ("#f0c8a0", "#2b1d14", "#4f6f8f", "#f2f2f2"),
    ("#e8b98a", "#3b2a1f", "#7a4b3a", "#e8e2d6"),
    ("#c99b74", "#1f1712", "#4b5d67", "#dadfe3"),
    ("#a9714b", "#15110f", "#a35d3b", "#eeeeee"),
    ("#8d5a3b", "#241812", "#2f4858", "#f7f1e3"),
    ("#d8a87a", "#5a3a24", "#5d6b3f", "#e9e4da"),
    ("#b57b50", "#2b1d14", "#8441d6", "#eee6dc"),
    ("#f2d3b3", "#7a3f1d", "#276764", "#f0e9df"),
)


def default_palette(seed_text: str, species: str = "human") -> list[str]:
    rng = _seed(seed_text)
    if (species or "human").lower() in ("human", "person"):
        return list(_NATURAL_PALETTES[rng.randrange(len(_NATURAL_PALETTES))])
    base = rng.random()
    def hsl(h: float, s: float, l: float) -> str:
        r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return [hsl(base, 0.55, 0.72), hsl(base + 0.5, 0.45, 0.28),
            hsl(base + 0.75, 0.65, 0.55), hsl(base + 0.1, 0.35, 0.9)]


@dataclass
class Character:
    id: str
    name: str
    description: str = ""
    appearance: str = ""
    hair: str = ""
    clothing: str = ""
    color_palette: list[str] = field(default_factory=list)
    personality: str = ""
    species: str = "human"
    gender: str | None = None
    voice: str | None = None
    age: int | None = None
    age_group: str = "adult"
    build: str = "average"
    hair_style: str = "short"
    accessories: list[str] = field(default_factory=list)
    height_scale: float = 1.0
    reference_images: list[str] = field(default_factory=list)
    selected_reference: str | None = None
    rig: dict[str, Any] = field(default_factory=dict)
    projects: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Character":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_spec(cls, spec: CharacterSpec, *, project_slug: str | None = None) -> "Character":
        palette = list(spec.color_palette) or default_palette(spec.id + spec.name, spec.species)
        rig = {
            "type": "rig_2d",
            "body": spec.build if spec.build != "average" else "standard",
            "age_group": spec.age_group,
            "hair_style": spec.hair_style,
            "accessories": list(spec.accessories),
            "height_scale": spec.height_scale,
            "bones": ["root", "hips", "spine", "chest", "neck", "head",
                      "upper_arm_l", "forearm_l", "hand_l",
                      "upper_arm_r", "forearm_r", "hand_r",
                      "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r"],
            "mouth_shapes": ["rest", "aa", "ee", "oo", "mm", "fv", "th", "smile", "laugh"],
            "expressions": ["neutral", "happy", "curious", "surprised", "sad", "angry",
                            "laughing", "thoughtful"],
            "actions": ["idle", "walk", "run", "turn", "jump", "wave", "point", "laugh",
                        "clap", "dance", "greet", "approach", "leave", "sit", "stand",
                        "reach", "pick_up", "put_down", "throw", "catch", "play", "react",
                        "listen", "speak"],
        }
        if spec.species in ("robot", "creature"):
            rig["body"] = spec.species
        return cls(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            appearance=spec.appearance,
            clothing=spec.clothing,
            color_palette=palette,
            personality=spec.personality,
            species=spec.species,
            gender=spec.gender,
            voice=spec.voice,
            age=spec.age,
            age_group=spec.age_group,
            build=spec.build,
            hair_style=spec.hair_style,
            accessories=list(spec.accessories),
            height_scale=spec.height_scale,
            reference_images=list(spec.reference_images),
            rig=rig,
            projects=[project_slug] if project_slug else [],
        )

    def to_spec(self) -> CharacterSpec:
        return CharacterSpec(
            id=self.id,
            name=self.name,
            species=self.species,
            description=self.description,
            appearance=self.appearance,
            clothing=self.clothing,
            color_palette=self.color_palette,
            personality=self.personality,
            gender=self.gender,
            voice=self.voice,
            reference_images=self.reference_images,
            age=self.age,
            age_group=self.age_group,
            build=self.build,
            hair_style=self.hair_style,
            accessories=list(self.accessories),
            height_scale=self.height_scale,
        )


class CharacterLibrary:
    """JSON-backed character library.

    ``root`` holds ``library.json`` and a ``refs/`` directory for portraits.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.refs_dir = self.root / "refs"
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Character] = {}
        self.load()

    # -- persistence --------------------------------------------------------
    @property
    def library_path(self) -> Path:
        return self.root / LIBRARY_FILENAME

    def load(self) -> None:
        if not self.library_path.exists():
            self._items = {}
            return
        try:
            data = json.loads(self.library_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._items = {}
            return
        items = data.get("characters", data) if isinstance(data, dict) else data
        self._items = {}
        for entry in items if isinstance(items, list) else []:
            if isinstance(entry, dict) and entry.get("id"):
                c = Character.from_dict(entry)
                self._items[c.id] = c

    def save(self) -> Path:
        data = {"version": 1, "characters": [c.to_dict() for c in self._items.values()]}
        tmp = self.library_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.library_path)
        return self.library_path

    # -- CRUD ---------------------------------------------------------------
    def list(self) -> list[Character]:
        return sorted(self._items.values(), key=lambda c: c.created_at)

    def get(self, char_id: str) -> Character | None:
        return self._items.get(char_id)

    def require(self, char_id: str) -> Character:
        c = self.get(char_id)
        if c is None:
            raise KeyError(f"character not found: {char_id}")
        return c

    def add(self, character: Character) -> Character:
        # Unique id.
        base = character.id or "char"
        cid = base
        n = 2
        while cid in self._items:
            cid = f"{base}_{n}"
            n += 1
        character.id = cid
        if not character.color_palette:
            character.color_palette = default_palette(character.id + character.name, character.species)
        character.created_at = character.created_at or time.time()
        character.updated_at = time.time()
        self._items[cid] = character
        self.save()
        return character

    def update(self, character: Character) -> Character:
        if character.id not in self._items:
            raise KeyError(f"character not found: {character.id}")
        character.updated_at = time.time()
        self._items[character.id] = character
        self.save()
        return character

    def delete(self, char_id: str) -> bool:
        if char_id in self._items:
            del self._items[char_id]
            self.save()
            return True
        return False

    def associate_project(self, char_id: str, project_slug: str) -> None:
        c = self.get(char_id)
        if c and project_slug and project_slug not in c.projects:
            c.projects.append(project_slug)
            self.update(c)

    # -- reference portraits ------------------------------------------------
    def add_reference(self, char_id: str, image_path: Path, *, select: bool = False) -> Character:
        c = self.require(char_id)
        path = str(image_path)
        if path not in c.reference_images:
            c.reference_images.append(path)
        if select or c.selected_reference is None:
            c.selected_reference = path
        return self.update(c)

    def generate_reference(self, char_id: str, *, size: int = 512, variant: int = 0) -> Path:
        """Draw a procedural reference portrait and register it."""
        c = self.require(char_id)
        out = self.refs_dir / f"{c.id}_ref{variant + 1}.png"
        draw_character_portrait(c, out, size=size, variant=variant)
        self.add_reference(c.id, out, select=True)
        return out


# ---------------------------------------------------------------------------
# Procedural portrait drawing (Pillow)
# ---------------------------------------------------------------------------
def _hex_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    try:
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    except (ValueError, IndexError):
        return (200, 180, 160)


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(v * factor))) for v in rgb)  # type: ignore[return-value]


def draw_character_portrait(character: Character, out_path: Path, *, size: int = 512,
                            variant: int = 0) -> Path:
    """Draw a stylized head-and-shoulders portrait from the character's data."""
    from PIL import Image, ImageDraw

    rng = _seed(character.id + str(variant))
    palette = character.color_palette or default_palette(character.id + character.name, character.species)
    skin = _hex_rgb(palette[0])
    hair = _hex_rgb(palette[1])
    accent = _hex_rgb(palette[2])
    bg = _hex_rgb(palette[3])

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    # Soft backdrop circle.
    for i in range(6, 0, -1):
        r = int(size * (0.32 + i * 0.02))
        col = _shade(bg, 1.0 + i * 0.03)
        draw.ellipse([size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r], fill=col)

    cx = size // 2
    species = character.species
    head_r = int(size * 0.21)
    head_y = int(size * 0.42)

    # Shoulders / clothing.
    body_color = accent
    draw.ellipse([cx - int(size * 0.33), head_y + head_r, cx + int(size * 0.33), size + int(size * 0.1)],
                 fill=body_color)
    draw.polygon([(cx - int(size * 0.09), head_y + head_r - int(size * 0.02)),
                  (cx + int(size * 0.09), head_y + head_r - int(size * 0.02)),
                  (cx, head_y + head_r + int(size * 0.10))], fill=_shade(skin, 0.92))

    if species == "robot":
        draw.rounded_rectangle([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r],
                               radius=int(head_r * 0.25), fill=_shade(skin, 0.85),
                               outline=_shade(hair, 0.8), width=max(3, size // 120))
        draw.rectangle([cx - int(head_r * 1.4), head_y - int(head_r * 1.35),
                        cx + int(head_r * 1.4), head_y - int(head_r * 1.1)], fill=_shade(skin, 0.7))
        draw.ellipse([cx - int(head_r * 0.62), head_y - int(head_r * 0.35),
                      cx - int(head_r * 0.18), head_y + int(head_r * 0.08)], fill=accent)
        draw.ellipse([cx + int(head_r * 0.18), head_y - int(head_r * 0.35),
                      cx + int(head_r * 0.62), head_y + int(head_r * 0.08)], fill=accent)
        draw.rectangle([cx - int(head_r * 0.4), head_y + int(head_r * 0.45),
                        cx + int(head_r * 0.4), head_y + int(head_r * 0.6)], fill=_shade(hair, 0.6))
        draw.line([(cx, head_y - head_r), (cx, head_y - int(head_r * 1.5))], fill=accent, width=size // 90)
        draw.ellipse([cx - size // 40, head_y - int(head_r * 1.5) - size // 40,
                      cx + size // 40, head_y - int(head_r * 1.5) + size // 40], fill=accent)
    else:
        # Head.
        draw.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r], fill=skin)
        # Hair / ears depending on species.
        if species == "animal":
            ear_r = int(head_r * 0.45)
            for sign in (-1, 1):
                ex = cx + sign * int(head_r * 0.72)
                draw.ellipse([ex - ear_r, head_y - head_r - ear_r // 2,
                              ex + ear_r, head_y - head_r + ear_r], fill=hair)
            draw.ellipse([cx - int(head_r * 0.35), head_y + int(head_r * 0.15),
                          cx + int(head_r * 0.35), head_y + int(head_r * 0.75)], fill=_shade(skin, 1.08))
            draw.ellipse([cx - int(head_r * 0.12), head_y + int(head_r * 0.25),
                          cx + int(head_r * 0.12), head_y + int(head_r * 0.42)], fill=_shade(hair, 0.5))
        elif species == "creature":
            for sign in (-1, 1):
                bx = cx + sign * int(head_r * 0.55)
                draw.polygon([(bx, head_y - head_r), (bx + sign * int(head_r * 0.35), head_y - int(head_r * 1.5)),
                              (bx + sign * int(head_r * 0.1), head_y - head_r)], fill=hair)
            draw.arc([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r], 200, 340, fill=hair,
                     width=max(4, size // 90))
        else:
            # Human hair: cap + side locks.
            draw.chord([cx - head_r, head_y - int(head_r * 1.15), cx + head_r, head_y + int(head_r * 0.55)],
                       180, 360, fill=hair)
            for sign in (-1, 1):
                draw.rounded_rectangle([cx + sign * int(head_r * 0.85) - int(head_r * 0.16),
                                        head_y - int(head_r * 0.2),
                                        cx + sign * int(head_r * 0.85) + int(head_r * 0.16),
                                        head_y + int(head_r * 0.75)],
                                       radius=int(head_r * 0.16), fill=hair)
            # Simple hair highlight.
            draw.arc([cx - int(head_r * 0.7), head_y - int(head_r * 0.95),
                      cx + int(head_r * 0.7), head_y - int(head_r * 0.1)], 200, 340,
                     fill=_shade(hair, 1.5), width=max(2, size // 200))

        # Eyes.
        eye_y = head_y - int(head_r * 0.05)
        eye_dx = int(head_r * 0.38)
        eye_r = max(4, int(head_r * 0.14))
        for sign in (-1, 1):
            ex = cx + sign * eye_dx
            draw.ellipse([ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r], fill=(255, 255, 255))
            draw.ellipse([ex - eye_r // 2, eye_y - eye_r // 2, ex + eye_r // 2, eye_y + eye_r // 2],
                         fill=_shade(hair, 0.5))
            draw.ellipse([ex - 2, eye_y - 2, ex + 2, eye_y + 2], fill=(20, 20, 20))
        # Brows.
        for sign in (-1, 1):
            ex = cx + sign * eye_dx
            draw.arc([ex - eye_r * 2, eye_y - eye_r * 3, ex + eye_r * 2, eye_y - eye_r],
                     200, 340, fill=_shade(hair, 0.6), width=max(2, size // 180))
        # Nose + smile.
        draw.line([(cx, head_y + int(head_r * 0.12)), (cx - int(head_r * 0.08), head_y + int(head_r * 0.3))],
                  fill=_shade(skin, 0.8), width=max(2, size // 180))
        draw.arc([cx - int(head_r * 0.35), head_y + int(head_r * 0.3),
                  cx + int(head_r * 0.35), head_y + int(head_r * 0.72)], 20, 160,
                 fill=_shade(hair, 0.45), width=max(3, size // 100))

    # Name plate.
    try:
        from .fonts import find_font_for_script

        fr = find_font_for_script("latin")
        from PIL import ImageFont

        font = ImageFont.truetype(str(fr.path), max(14, size // 18)) if fr.ok and fr.path else ImageFont.load_default()
        text = character.name[:24]
        tw = draw.textlength(text, font=font)
        draw.rectangle([0, size - int(size * 0.11), size, size], fill=(0, 0, 0))
        draw.text(((size - tw) / 2, size - int(size * 0.095)), text, font=font, fill=(255, 255, 255))
    except Exception:  # noqa: BLE001 — label is optional
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def ensure_project_characters(
    library: CharacterLibrary,
    specs: list[CharacterSpec],
    project_slug: str,
) -> list[Character]:
    """Create/refresh library records for a project's characters."""
    out: list[Character] = []
    for spec in specs:
        existing = library.get(spec.id)
        if existing is None:
            record = Character.from_spec(spec, project_slug=project_slug)
            # Draw a first reference portrait so designs persist.
            library.add(record)
            try:
                library.generate_reference(record.id, size=384)
            except Exception:  # noqa: BLE001
                pass
            out.append(library.require(record.id))
        else:
            # Merge updated description/voice without clobbering references.
            existing.name = spec.name or existing.name
            existing.appearance = spec.appearance or existing.appearance
            existing.clothing = spec.clothing or existing.clothing
            existing.personality = spec.personality or existing.personality
            if spec.color_palette:
                existing.color_palette = spec.color_palette
            if spec.voice:
                existing.voice = spec.voice
            library.associate_project(existing.id, project_slug)
            library.update(existing)
            out.append(existing)
    return out
