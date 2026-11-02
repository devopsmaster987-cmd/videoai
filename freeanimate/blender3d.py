"""Optional genuine 3D rendering through a locally installed Blender.

Blender is *not bundled* and is not required.  When it is installed, this module
generates a complete ``bpy`` scene (characters, environment, lights, camera,
lip-synced mouth motion, head bobs, gestures, blinks), renders a PNG sequence and
encodes it with FFmpeg.

When Blender is absent the module says so and returns ``ok=False``; the studio
then falls back to the procedural 2D renderer, and the report states that 3D was
requested but unavailable.  It never pretends 3D output was produced.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .audio_mix import wav_duration_seconds
from .characters import Character, default_palette
from .ffmpeg_utils import FFmpegError, run_ffmpeg
from .lipsync import mouth_openness, viseme_at

METHOD_NAME = "blender_3d_procedural_scene"


@dataclass
class BlenderStatus:
    available: bool
    executable: str | None = None
    version: str | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def find_blender() -> BlenderStatus:
    candidates: list[Path] = []
    env = os.environ.get("BLENDER_EXE")
    if env:
        candidates.append(Path(env))
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    if sys.platform == "win32":
        for root in (Path("C:/Program Files/Blender Foundation"),
                     Path("C:/Program Files (x86)/Blender Foundation")):
            if root.is_dir():
                candidates.extend(sorted(root.glob("Blender */blender.exe"), reverse=True))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            portable = Path(local) / "FreeAnimate" / "Blender"
            if portable.is_dir():
                candidates.extend(sorted(portable.glob("blender-*/blender.exe"), reverse=True))
        repo_tools = Path(__file__).resolve().parent.parent / "tools"
        if repo_tools.is_dir():
            candidates.extend(sorted(repo_tools.glob("blender*/blender.exe"), reverse=True))
    candidates += [Path("/usr/bin/blender"), Path("/usr/local/bin/blender"),
                   Path("/Applications/Blender.app/Contents/MacOS/Blender")]
    for c in candidates:
        if c.exists() and c.is_file():
            version = None
            try:
                out = subprocess.run([str(c), "--version"], capture_output=True, text=True,
                                     timeout=30)
                first = (out.stdout or "").splitlines()
                version = first[0].strip() if first else None
            except (OSError, subprocess.SubprocessError):
                pass
            return BlenderStatus(True, str(c), version)
    return BlenderStatus(
        False, None, None,
        "Blender was not found. Install Blender (free, blender.org) or set BLENDER_EXE to its "
        "executable. 3D scenes are unavailable until then; 2D animation still works.",
    )


# ---------------------------------------------------------------------------
# Scene description helpers
# ---------------------------------------------------------------------------
_ENV_PALETTE: dict[str, dict[str, Any]] = {
    "garden": {"floor": "#6f9e4a", "sky": "#bfe3f5",
               "props": ["tree", "tree", "flowers", "fountain", "hedge", "bench", "flowers"]},
    "park": {"floor": "#7aa85a", "sky": "#c7e6f7",
             "props": ["tree", "tree", "bench", "flowers", "path", "hedge"]},
    "forest": {"floor": "#3f6b3a", "sky": "#9fc7b0",
               "props": ["tree", "tree", "tree", "tree", "tree", "rock", "fern"]},
    "beach": {"floor": "#e7d6a4", "sky": "#bfe9f7",
              "props": ["water", "palm", "palm", "shell", "sandcastle"]},
    "space": {"floor": "#1a1f33", "sky": "#05060f", "props": ["stars", "planet", "planet"]},
    "city": {"floor": "#7c7f86", "sky": "#dfe8ef",
             "props": ["building", "building", "building", "streetlamp", "car", "tree"]},
    "school": {"floor": "#c8b28a", "sky": "#d8ecef",
               "props": ["wall", "blackboard", "desk", "desk", "books", "clock"]},
    "castle": {"floor": "#7d7f74", "sky": "#cdd8e6",
               "props": ["tower", "tower", "tree", "candle"]},
    "mountains": {"floor": "#7e9b6a", "sky": "#cfe4f2",
                  "props": ["mountain", "mountain", "tree", "rock"]},
    "room": {"floor": "#b58b62", "sky": "#e9e2d6",
             "props": ["wall", "shelf", "sofa", "lamp", "window", "rug"]},
    "cafe": {"floor": "#a67c52", "sky": "#f0e2cc",
             "props": ["wall", "counter", "shelf", "cup", "lamp", "window"]},
    "kitchen": {"floor": "#c9a06a", "sky": "#f1e9da",
                "props": ["wall", "counter", "shelf", "window", "stove", "pot",
                          "steam", "lamp"]},
    "underwater": {"floor": "#2c5f6f", "sky": "#1d6f86",
                   "props": ["coral", "coral", "seaweed", "rock"]},
    "snow": {"floor": "#eef4f8", "sky": "#c9dceb",
             "props": ["tree", "tree", "rock", "tree"]},
    "desert": {"floor": "#dfc08a", "sky": "#f3e2c2",
               "props": ["cactus", "dune", "rock", "cactus"]},
    "horror_mansion": {"floor": "#4a4148", "sky": "#2a2330",
                       "props": ["wall", "candle", "tree", "candle"]},
    "night": {"floor": "#2b3a4a", "sky": "#0b1424", "props": ["tree", "stars", "tree"]},
    "generic": {"floor": "#8f9a86", "sky": "#cfe0ea", "props": ["tree", "flowers", "bench"]},
}


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _scene_props(environment: str) -> dict[str, Any]:
    """Deterministic 3D prop layout for one environment.

    Props are real meshes placed in front, middle and background layers so the
    scene has depth rather than a single flat row.
    """
    spec = _ENV_PALETTE.get(environment, _ENV_PALETTE["generic"])
    props: list[dict[str, Any]] = []
    count = 0
    for kind in spec["props"]:
        count += 1
        row = count % 3
        x = -4.4 + (count - 1) * 1.75
        y = 3.2 + row * 1.9
        props.append({"kind": kind, "x": x, "y": y, "z": 0.0,
                      "scale": 0.82 + (count % 4) * 0.14,
                      "color": spec["floor"]})
    return {"floor": spec["floor"], "sky": spec["sky"], "props": props}


def _character_3d(c: Character, index: int, total: int, *, style: str = "cartoon"
                   ) -> dict[str, Any]:
    palette = c.color_palette or default_palette(c.id + c.name)
    while len(palette) < 4:
        palette += default_palette(c.id + str(len(palette)))
    spread = 1.15 if total <= 2 else 1.55
    x = 0.0 if total == 1 else (-spread if index == 0 else spread if index == total - 1
                                else -spread + (2 * spread) * index / (total - 1))
    def color(value: str) -> list[float]:
        try:
            return list(_hex_rgb(value))
        except (ValueError, TypeError):
            return [0.7, 0.7, 0.7]

    skin, hair, shirt, pants = (color(palette[0]), color(palette[1]),
                                color(palette[2]), color(palette[3]))
    robot = (c.species or "human") == "robot"
    if style == "realistic":
        # Natural, plausible palettes instead of arbitrary saturated colors.
        natural = [
            ([0.86, 0.68, 0.51], [0.16, 0.10, 0.07], [0.25, 0.32, 0.42], [0.18, 0.20, 0.24]),
            ([0.66, 0.45, 0.30], [0.08, 0.06, 0.05], [0.42, 0.26, 0.20], [0.20, 0.22, 0.25]),
            ([0.92, 0.78, 0.64], [0.30, 0.18, 0.10], [0.30, 0.38, 0.36], [0.22, 0.24, 0.27]),
            ([0.52, 0.33, 0.22], [0.10, 0.08, 0.07], [0.55, 0.49, 0.38], [0.16, 0.18, 0.22]),
        ][index % 4]
        skin, hair, shirt, pants = [list(v) for v in natural]
    elif robot:
        skin = [0.62, 0.65, 0.70]
        hair = [0.10, 0.55, 0.72]
        shirt = [0.36, 0.39, 0.44]
        pants = [0.22, 0.24, 0.29]
    else:
        tones = [[0.90, 0.70, 0.55], [0.72, 0.51, 0.36],
                 [0.56, 0.37, 0.25], [0.96, 0.81, 0.69]]
        warm = tones[index % len(tones)]
        skin = [min(1.0, 0.30 * skin[i] + 0.70 * warm[i]) for i in range(3)]
        pants = [min(1.0, 0.72 * v) for v in pants]

    return {
        "id": c.id,
        "name": c.name,
        "species": c.species or "human",
        "x": x,
        "skin": skin,
        "hair": hair,
        "shirt": shirt,
        "pants": pants,
        "height": 1.0 + (0.04 if c.species == "robot" else 0.0),
        "gender": c.gender,
    }


def build_shot_plan(characters: list[Character], spans: list[Any],
                    duration: float) -> list[dict[str, Any]]:
    """Plan genuine camera cuts: a two-shot, per-speaker close-ups, reactions.

    The returned intervals are contiguous so every frame belongs to exactly one
    shot.  ``kind`` is ``two``, ``speaker`` or ``reaction`` and ``speaker`` is a
    character id where applicable.
    """
    duration = max(0.8, float(duration))
    ids = [c.id for c in characters]
    if not spans or not ids:
        return [{"start": 0.0, "end": duration, "kind": "two", "speaker": None}]

    ordered = sorted(spans, key=lambda s: (getattr(s, "start", 0.0), getattr(s, "end", 0.0)))
    shots: list[dict[str, Any]] = []
    cursor = 0.0
    first = ordered[0]
    open_end = min(duration, max(0.9, float(first.start) + 0.7))
    if open_end > 0:
        shots.append({"start": 0.0, "end": open_end, "kind": "two", "speaker": None})
        cursor = open_end
    for i, span in enumerate(ordered):
        start = max(cursor, float(getattr(span, "start", 0.0)) - 0.10)
        if start > cursor + 0.02:
            shots.append({"start": cursor, "end": start, "kind": "two", "speaker": None})
        end = min(duration, max(start + 0.5, float(getattr(span, "end", start + 0.5)) + 0.35))
        speaker = getattr(span, "character_id", None)
        shots.append({"start": start, "end": end, "kind": "speaker", "speaker": speaker})
        cursor = end
        # If there is a real pause before the next line, show the listener.
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            gap_start = cursor
            gap_end = max(cursor, float(getattr(nxt, "start", cursor)))
            if gap_end - gap_start >= 0.65:
                listener = next((cid for cid in ids if cid != speaker), None)
                if listener:
                    shots.append({"start": gap_start, "end": gap_end,
                                  "kind": "reaction", "speaker": listener})
                    cursor = gap_end
    if cursor < duration - 0.02:
        shots.append({"start": cursor, "end": duration, "kind": "two", "speaker": None})
    # Repair any accidental gaps/overlaps and clamp.
    cleaned: list[dict[str, Any]] = []
    t = 0.0
    for shot in shots:
        shot["start"] = max(t, min(duration, shot["start"]))
        shot["end"] = max(shot["start"] + 0.2, min(duration, shot["end"]))
        if shot["end"] <= shot["start"]:
            continue
        cleaned.append(shot)
        t = shot["end"]
    if cleaned:
        cleaned[-1]["end"] = duration
    return cleaned or [{"start": 0.0, "end": duration, "kind": "two", "speaker": None}]


def _mouth_curves(characters: list[Character], spans: list[Any], duration: float,
                  fps: int) -> dict[str, list[float]]:
    frames = max(1, int(round(duration * fps)))
    out: dict[str, list[float]] = {c.id: [0.0] * frames for c in characters}
    for span in spans:
        cid = getattr(span, "character_id", None)
        if cid not in out:
            continue
        for f in range(frames):
            t = f / fps
            if span.start <= t <= span.end:
                local = t - span.start
                vis = viseme_at(span.visemes, local)
                out[cid][f] = max(out[cid][f], mouth_openness(vis))
    return out


def _motion_curves(characters: list[Character], spans: list[Any], duration: float,
                   fps: int) -> dict[str, dict[str, list[float]]]:
    import math

    frames = max(1, int(round(duration * fps)))
    res: dict[str, dict[str, list[float]]] = {}
    for c in characters:
        nod: list[float] = []
        gesture: list[float] = []
        blink: list[float] = []
        for f in range(frames):
            t = f / fps
            speaking = any(getattr(s, "character_id", None) == c.id and s.start <= t <= s.end
                           for s in spans)
            nod.append((0.012 if speaking else 0.006) * math.sin(t * (6.0 if speaking else 2.0)))
            gesture.append((0.16 if speaking else 0.05) * math.sin(t * 4.3 + hash(c.id) % 7))
            phase = (t + (hash(c.id) % 100) / 100.0 * 3.0) % 3.2
            blink.append(0.12 if phase < 0.11 else 1.0)
        res[c.id] = {"nod": nod, "gesture": gesture, "blink": blink}
    return res


# ---------------------------------------------------------------------------
# bpy script template (only executed by Blender, never imported here)
# ---------------------------------------------------------------------------
_SCRIPT = r'''
import bpy, json, math, os, sys
from mathutils import Vector

DATA = json.loads(r"""__DATA_JSON__""")
out_dir = DATA["frames_dir"]
os.makedirs(out_dir, exist_ok=True)
STYLE = DATA.get("style", "cartoon")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Render settings -----------------------------------------------------------
scene.render.resolution_x = DATA["width"]
scene.render.resolution_y = DATA["height"]
scene.render.resolution_percentage = 100
scene.render.fps = DATA["fps"]
scene.frame_start = 1
scene.frame_end = DATA["frames"]
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
engine = DATA["engine"]
if engine == "EEVEE":
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0) else "BLENDER_EEVEE"
    except Exception:
        scene.render.engine = "BLENDER_WORKBENCH"
else:
    scene.render.engine = "BLENDER_WORKBENCH"
    try:
        sh = scene.display.shading
        sh.light = "STUDIO"
        sh.color_type = "MATERIAL"
        sh.show_shadows = True
        sh.show_cavity = True
        sh.background_type = "WORLD"
    except Exception:
        pass

# World ---------------------------------------------------------------------
sky = DATA["sky"]
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.color = (sky[0], sky[1], sky[2])
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (sky[0], sky[1], sky[2], 1.0)
    bg.inputs[1].default_value = 1.0


def material(name, color, rough=0.65, metal=0.0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (color[0], color[1], color[2], 1.0)
    m.use_nodes = True
    try:
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
            bsdf.inputs["Roughness"].default_value = rough
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = metal
            if STYLE == "realistic" and "skin" in name.lower():
                if "Subsurface Weight" in bsdf.inputs:
                    bsdf.inputs["Subsurface Weight"].default_value = 0.08
                if "Specular IOR Level" in bsdf.inputs:
                    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    except Exception:
        pass
    return m


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def cube(name, loc, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    return assign(o, mat)


def sphere(name, loc, radius, mat, scale=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=loc)
    o = bpy.context.active_object
    o.name = name
    if scale:
        o.scale = scale
    smooth(o)
    return assign(o, mat)


def cylinder(name, loc, radius, depth, mat, vertices=32, rot=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    o = bpy.context.active_object
    o.name = name
    if rot:
        o.rotation_euler = rot
    smooth(o)
    return assign(o, mat)


def cone(name, loc, r1, r2, depth, mat, rot=None):
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=r1, radius2=r2, depth=depth, location=loc)
    o = bpy.context.active_object
    o.name = name
    if rot:
        o.rotation_euler = rot
    smooth(o)
    return assign(o, mat)


def smooth(obj):
    try:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    except Exception:
        pass
    return obj


def cyl_between(name, p0, p1, radius, mat):
    a, b = Vector(p0), Vector(p1)
    delta = b - a
    length = max(0.001, delta.length)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=length,
                                        location=(a + b) * 0.5)
    o = bpy.context.active_object
    o.name = name
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = delta.to_track_quat("Z", "Y")
    smooth(o)
    return assign(o, mat)


def create_armature(prefix, x, seated):
    """Build a real Blender armature with spine, arms, legs and head bones."""
    bpy.ops.object.armature_add(location=(0, 0, 0))
    arm = bpy.context.active_object
    arm.name = "Rig_" + prefix
    arm.show_in_front = True
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for b in list(eb):
        eb.remove(b)

    def mk(name, head, tail, parent=None):
        b = eb.new(name)
        b.head = head
        b.tail = tail
        if parent:
            b.parent = eb[parent]
        b.use_connect = False
        return b

    if seated:
        hip_z, chest_z = 0.90, 1.32
        mk("root", (x, 0, 0.0), (x, 0, 0.30))
        mk("hips", (x, 0, hip_z), (x, 0, 1.02), "root")
        mk("spine", (x, 0, 1.02), (x, 0, 1.20), "hips")
        mk("chest", (x, 0, 1.20), (x, 0, chest_z), "spine")
        mk("neck", (x, 0, chest_z), (x, 0, 1.44), "chest")
        mk("head", (x, 0, 1.44), (x, 0, 1.70), "neck")
        for side, sx in (("L", -1), ("R", 1)):
            mk("upper_arm." + side, (x + sx * 0.21, 0, 1.28),
               (x + sx * 0.30, -0.18, 1.02), "chest")
            mk("forearm." + side, (x + sx * 0.30, -0.18, 1.02),
               (x + sx * 0.31, -0.40, 0.82), "upper_arm." + side)
            mk("hand." + side, (x + sx * 0.31, -0.40, 0.82),
               (x + sx * 0.31, -0.50, 0.77), "forearm." + side)
            mk("thigh." + side, (x + sx * 0.13, 0, hip_z),
               (x + sx * 0.13, -0.44, 0.54), "hips")
            mk("shin." + side, (x + sx * 0.13, -0.44, 0.54),
               (x + sx * 0.13, -0.30, 0.12), "thigh." + side)
            mk("foot." + side, (x + sx * 0.13, -0.30, 0.12),
               (x + sx * 0.13, -0.46, 0.07), "shin." + side)
    else:
        mk("root", (x, 0, 0.0), (x, 0, 0.30))
        mk("hips", (x, 0, 0.84), (x, 0, 0.99), "root")
        mk("spine", (x, 0, 0.99), (x, 0, 1.18), "hips")
        mk("chest", (x, 0, 1.18), (x, 0, 1.38), "spine")
        mk("neck", (x, 0, 1.38), (x, 0, 1.48), "chest")
        mk("head", (x, 0, 1.48), (x, 0, 1.72), "neck")
        for side, sx in (("L", -1), ("R", 1)):
            mk("upper_arm." + side, (x + sx * 0.22, 0, 1.34),
               (x + sx * 0.33, 0, 1.07), "chest")
            mk("forearm." + side, (x + sx * 0.33, 0, 1.07),
               (x + sx * 0.36, -0.04, 0.81), "upper_arm." + side)
            mk("hand." + side, (x + sx * 0.36, -0.04, 0.81),
               (x + sx * 0.37, -0.06, 0.72), "forearm." + side)
            mk("thigh." + side, (x + sx * 0.13, 0, 0.88),
               (x + sx * 0.13, 0, 0.49), "hips")
            mk("shin." + side, (x + sx * 0.13, 0, 0.49),
               (x + sx * 0.13, 0, 0.11), "thigh." + side)
            mk("foot." + side, (x + sx * 0.13, 0, 0.11),
               (x + sx * 0.13, -0.20, 0.06), "shin." + side)
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def bone_parent(obj, arm, bone):
    world = obj.matrix_world.copy()
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    obj.matrix_world = world


# Ground --------------------------------------------------------------------
floor = material("Ground", DATA["floor"], rough=0.9)
cube("Ground", (0, 0, -0.06), (24, 24, 0.12), floor)

# Lights --------------------------------------------------------------------
bpy.ops.object.light_add(type="SUN", location=(3, -4, 8))
sun = bpy.context.active_object
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(35))
bpy.ops.object.light_add(type="AREA", location=(-3.5, -3.0, 4.0))
fill = bpy.context.active_object
fill.data.energy = 180
fill.data.size = 5
fill.rotation_euler = (math.radians(60), 0, math.radians(-40))

# Camera --------------------------------------------------------------------
bpy.ops.object.camera_add(location=(0.15, -6.6, 2.25))
cam = bpy.context.active_object
cam.name = "Camera"
scene.camera = cam
target = bpy.data.objects.new("Target", None)
scene.collection.objects.link(target)
target.location = (0, 0.05, 1.06)
tc = cam.constraints.new(type="TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"
cam.data.lens = 42

# Props ---------------------------------------------------------------------
for i, p in enumerate(DATA["props"]):
    col = DATA["prop_color"]
    kind = p["kind"]
    x, y, z, s = p["x"], p["y"], p["z"], p["scale"]
    if kind in ("tree", "palm", "cactus"):
        trunk = material("Trunk%d" % i, (0.34, 0.22, 0.12))
        leaf = material("Leaf%d" % i, (0.18 + 0.1 * (i % 3), 0.42 + 0.08 * (i % 2), 0.2))
        cylinder("Trunk%d" % i, (x, y, z + 0.9 * s), 0.11 * s, 1.8 * s, trunk)
        if kind == "cactus":
            sphere("Cactus%d" % i, (x, y, z + 2.0 * s), 0.28 * s, leaf, scale=(1.0, 1.0, 2.2))
        else:
            for k in range(3):
                sphere("Leaf%d_%d" % (i, k), (x + (k - 1) * 0.22 * s, y, z + (2.0 + 0.2 * k) * s),
                       0.75 * s, leaf, scale=(1.0, 1.0, 0.85))
    elif kind == "flowers":
        for k in range(7):
            col2 = (0.9, 0.4 + 0.07 * k, 0.6)
            sphere("Flower%d_%d" % (i, k), (x + (k % 3 - 1) * 0.3, y + (k // 3) * 0.3, z + 0.18),
                   0.09, material("FlowerMat%d_%d" % (i, k), col2))
    elif kind == "fountain":
        stone = material("Stone%d" % i, (0.72, 0.72, 0.68))
        water = material("Water%d" % i, (0.35, 0.62, 0.82), rough=0.15)
        cylinder("Basin%d" % i, (x, y, z + 0.25 * s), 0.85 * s, 0.5 * s, stone)
        cylinder("Water%d" % i, (x, y, z + 0.51 * s), 0.72 * s, 0.03 * s, water)
        cylinder("Column%d" % i, (x, y, z + 0.9 * s), 0.14 * s, 1.3 * s, stone)
        sphere("Top%d" % i, (x, y, z + 1.6 * s), 0.22 * s, water)
    elif kind in ("building", "wall", "tower"):
        mat = material("Build%d" % i, (0.55 + 0.05 * (i % 3), 0.52, 0.5))
        w = 2.2 if kind != "tower" else 0.9
        h = 3.0 if kind != "tower" else 4.2
        cube("Build%d" % i, (x, y + 1.5, z + h / 2), (w, 1.2, h), mat)
        if kind == "tower":
            cone("Roof%d" % i, (x, y + 1.5, z + h + 0.7), 0.8, 0.0, 1.4,
                 material("RoofMat%d" % i, (0.45, 0.2, 0.2)))
    elif kind in ("mountain", "dune"):
        mat = material("Mount%d" % i, (0.5, 0.55, 0.48) if kind == "mountain" else (0.85, 0.7, 0.45))
        cone("Mount%d" % i, (x, y + 1.0, z + 1.6 * s), 2.4 * s, 0.0, 3.2 * s, mat)
    elif kind == "water":
        water = material("Sea", (0.22, 0.5, 0.72), rough=0.1)
        cube("Sea", (0, y + 6.0, z + 0.02), (26, 12, 0.04), water)
    elif kind == "stars":
        star = material("Star", (1.0, 1.0, 0.85))
        for k in range(40):
            sphere("Star%d" % k, ((k % 8 - 4) * 2.4, y + 4 + (k // 8) * 0.6,
                                  3.0 + (k % 5) * 0.7), 0.035, star)
    elif kind == "coral":
        mat = material("Coral%d" % i, (0.85, 0.4, 0.55))
        for k in range(4):
            cylinder("Coral%d_%d" % (i, k), (x + (k - 1.5) * 0.2, y, z + 0.35),
                     0.07, 0.7 + 0.1 * k, mat)
    elif kind == "bench":
        wood = material("BenchWood%d" % i, (0.48, 0.3, 0.16))
        cube("Bench%d" % i, (x, y, z + 0.45), (1.6, 0.45, 0.12), wood)
    elif kind == "shelf":
        wood = material("ShelfWood%d" % i, (0.52, 0.34, 0.2))
        cube("Shelf%d" % i, (x, y + 0.8, z + 1.1), (1.8, 0.4, 2.2), wood)
    elif kind == "candle":
        sphere("Candle%d" % i, (x, y, z + 0.6), 0.12, material("CandleMat%d" % i, (1.0, 0.85, 0.4)))
    elif kind == "stove":
        metal = material("Stove%d" % i, (0.20, 0.21, 0.22), metal=0.55)
        cube("Stove%d" % i, (x, y, z + 0.45), (1.2, 0.75, 0.90), metal)
        for bx in (-0.32, 0.32):
            cylinder("Burner%d_%d" % (i, int(bx * 10)), (x + bx, y, z + 0.92),
                     0.18, 0.04, material("BurnerMat%d_%d" % (i, int(bx * 10)),
                                         (0.08, 0.08, 0.09), metal=0.8))
    elif kind == "pot":
        steel = material("Pot%d" % i, (0.62, 0.64, 0.66), metal=0.75)
        cylinder("Pot%d" % i, (x, y, z + 1.05), 0.24, 0.26, steel)
        cylinder("PotLid%d" % i, (x, y, z + 1.19), 0.25, 0.03, steel)
    elif kind == "steam":
        steam = material("Steam%d" % i, (0.94, 0.95, 0.96), rough=0.95)
        for k in range(4):
            sphere("Steam%d_%d" % (i, k), (x + 0.12 * math.sin(k * 1.7),
                                           y, z + 1.34 + 0.18 * k),
                   0.12 + 0.03 * k, steam, scale=(1.0, 1.0, 1.3))
    elif kind in ("counter", "desk"):
        wood = material("FurnitureWood%d" % i, (0.44, 0.28, 0.17))
        h = 1.05 if kind == "counter" else 0.72
        cube("Furniture%d" % i, (x, y, z + h / 2), (2.0, 0.75, h), wood)
        for lx in (-0.55, 0.55):
            cube("FurnitureTop%d_%d" % (i, int(lx * 10)),
                 (x + lx, y, z + h + 0.04), (0.5, 0.5, 0.05), wood)
    elif kind == "blackboard":
        cube("Board%d" % i, (x, y + 2.5, z + 1.35), (3.0, 0.08, 1.7),
             material("BoardMat%d" % i, (0.10, 0.28, 0.20), rough=0.95))
    elif kind == "books":
        for k in range(6):
            cube("BookStack%d_%d" % (i, k), (x, y, z + 0.05 + k * 0.09),
                 (0.42, 0.30, 0.08),
                 material("BookStackMat%d_%d" % (i, k),
                          [(0.55, 0.20, 0.18), (0.20, 0.35, 0.55),
                           (0.70, 0.65, 0.50)][k % 3]))
    elif kind == "clock":
        cylinder("Clock%d" % i, (x, y + 2.4, z + 2.2), 0.38, 0.08,
                 material("ClockMat%d" % i, (0.92, 0.92, 0.88)),
                 rot=(math.radians(90), 0, 0))
    elif kind == "lamp":
        cylinder("LampPole%d" % i, (x, y, z + 0.95), 0.035, 1.9,
                 material("LampPoleMat%d" % i, (0.22, 0.22, 0.23), metal=0.6))
        cone("LampShade%d" % i, (x, y, z + 2.0), 0.30, 0.10, 0.35,
             material("LampShadeMat%d" % i, (0.88, 0.82, 0.62)))
    elif kind == "window":
        cube("WindowFrame%d" % i, (x, y + 2.45, z + 1.55), (1.5, 0.10, 1.4),
             material("WindowFrameMat%d" % i, (0.52, 0.38, 0.24)))
        cube("WindowGlass%d" % i, (x, y + 2.38, z + 1.55), (1.28, 0.04, 1.18),
             material("WindowGlassMat%d" % i, (0.55, 0.76, 0.90), rough=0.05))
    elif kind == "sofa":
        fabric = material("Sofa%d" % i, (0.34, 0.34, 0.46))
        cube("SofaBase%d" % i, (x, y, z + 0.34), (1.9, 0.85, 0.42), fabric)
        cube("SofaBack%d" % i, (x, y + 0.36, z + 0.76), (1.9, 0.20, 0.66), fabric)
        for ax in (-0.82, 0.82):
            cube("SofaArm%d_%d" % (i, int(ax * 10)), (x + ax, y, z + 0.55),
                 (0.22, 0.85, 0.42), fabric)
    elif kind == "rug":
        cube("Rug%d" % i, (x, y, z + 0.015), (3.2, 2.1, 0.03),
             material("RugMat%d" % i, (0.55, 0.28, 0.24)))
    elif kind in ("hedge", "fern", "seaweed"):
        shrub = material("Shrub%d" % i,
                         (0.12, 0.40, 0.24) if kind == "seaweed" else (0.15, 0.36, 0.17))
        for k in range(4):
            if kind == "fern":
                cone("Fern%d_%d" % (i, k), (x + (k - 1.5) * 0.16, y, z + 0.42),
                     0.09, 0.0, 0.85, shrub,
                     rot=(0, math.radians((k - 1.5) * 18), 0))
            else:
                sphere("Shrub%d_%d" % (i, k), (x - 0.45 + k * 0.30, y, z + 0.38),
                       0.42 * s, shrub, scale=(1.0, 0.9, 0.85))
    elif kind == "path":
        cube("Path%d" % i, (x, y, z + 0.01), (1.4, 4.2, 0.04),
             material("PathMat%d" % i, (0.66, 0.60, 0.50), rough=0.98))
    elif kind == "rock":
        sphere("Rock%d" % i, (x, y, z + 0.18 * s), 0.30 * s,
               material("RockMat%d" % i, (0.48, 0.49, 0.46)),
               scale=(1.25, 0.9, 0.75))
    elif kind == "shell":
        sphere("Shell%d" % i, (x, y, z + 0.10), 0.14 * s,
               material("ShellMat%d" % i, (0.92, 0.80, 0.68)),
               scale=(1.2, 1.0, 0.5))
    elif kind == "sandcastle":
        sand = material("Sand%d" % i, (0.82, 0.70, 0.45))
        cube("Castle%d" % i, (x, y, z + 0.22), (0.8, 0.8, 0.44), sand)
        for tx in (-0.32, 0.32):
            cube("CastleTower%d_%d" % (i, int(tx * 10)), (x + tx, y, z + 0.38),
                 (0.28, 0.28, 0.76), sand)
    elif kind == "streetlamp":
        metal = material("StreetMetal%d" % i, (0.20, 0.22, 0.24), metal=0.7)
        cylinder("StreetPole%d" % i, (x, y, z + 1.4), 0.06, 2.8, metal)
        sphere("StreetLight%d" % i, (x, y, z + 2.85), 0.18,
               material("StreetLightMat%d" % i, (1.0, 0.88, 0.55)))
    elif kind == "car":
        car = material("Car%d" % i, (0.22, 0.35, 0.55), metal=0.3)
        cube("CarBody%d" % i, (x, y, z + 0.42), (1.7, 0.85, 0.48), car)
        cube("CarTop%d" % i, (x, y, z + 0.78), (0.95, 0.72, 0.32),
             material("CarGlass%d" % i, (0.32, 0.45, 0.55), rough=0.1))
        for wx in (-0.55, 0.55):
            cube("Wheel%d_%d" % (i, int(wx * 10)), (x + wx, y, z + 0.25),
                 (0.30, 0.90, 0.50),
                 material("WheelMat%d_%d" % (i, int(wx * 10)), (0.06, 0.06, 0.07)))
    elif kind == "planet":
        pc = [(0.75, 0.35, 0.28), (0.30, 0.52, 0.68), (0.72, 0.62, 0.32)][i % 3]
        sphere("Planet%d" % i, (x, y + 3.0, 3.6 + 0.4 * i), 0.55 * s,
               material("PlanetMat%d" % i, pc), scale=(1.0, 1.0, 0.96))

# Table and chairs (conversation foreground) --------------------------------
if DATA["table"]:
    wood = material("TableWood", (0.46, 0.29, 0.16))
    cube("Table", (0, -0.75, 0.72), (3.6, 1.0, 0.10), wood)
    for lx in (-1.55, 1.55):
        for ly in (-1.08, -0.42):
            cube("TableLeg", (lx, ly, 0.34), (0.09, 0.09, 0.68), wood)
    chair_wood = material("ChairWood", (0.52, 0.34, 0.19))
    cup_mat = material("CupMat", (0.94, 0.92, 0.86), rough=0.35)
    coffee = material("CoffeeMat", (0.20, 0.11, 0.06), rough=0.30)
    for ci, cc in enumerate(DATA["characters"]):
        cx = cc["x"]
        cylinder("Cup%d" % ci, (cx + 0.35, -0.78, 0.80), 0.075, 0.11, cup_mat)
        cylinder("Coffee%d" % ci, (cx + 0.35, -0.78, 0.847), 0.060, 0.012, coffee)
        cube("ChairSeat_%d" % ci, (cx, 0.55, 0.44), (0.82, 0.72, 0.08), chair_wood)
        cube("ChairBack_%d" % ci, (cx, 0.90, 0.72), (0.72, 0.09, 0.58), chair_wood)
        for lx in (-0.33, 0.33):
            for ly in (0.25, 0.85):
                cube("ChairLeg_%d_%d_%d" % (ci, int(lx * 10), int(ly * 10)),
                     (cx + lx, ly, 0.21), (0.07, 0.07, 0.42), chair_wood)

# Characters ----------------------------------------------------------------
char_objs = {}
seated = bool(DATA["table"])
for c in DATA["characters"]:
    base_x = c["x"]
    base_y = 0.24 if seated else 0.0
    h = c["height"]
    robot = c["species"] == "robot"
    skin = material("Skin_" + c["id"], c["skin"], rough=0.48 if robot else 0.62,
                    metal=0.55 if robot else 0.0)
    hair = material("Hair_" + c["id"], c["hair"], rough=0.82)
    shirt = material("Shirt_" + c["id"], c["shirt"], rough=0.55 if robot else 0.78,
                     metal=0.35 if robot else 0.0)
    pants = material("Pants_" + c["id"], c["pants"], rough=0.60 if robot else 0.82,
                     metal=0.25 if robot else 0.0)
    dark = material("Dark_" + c["id"], (0.045, 0.045, 0.055))
    white = material("White_" + c["id"], (0.96, 0.96, 0.96), rough=0.25)
    arm = create_armature(c["id"], base_x, seated)
    parts = []

    def part(obj, bone):
        try:
            vg = obj.vertex_groups.new(name=bone)
            vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
        except Exception:
            pass
        parts.append(obj)
        return obj

    head_r = 0.185 if STYLE == "cartoon" else 0.165
    torso_r = 0.245 if STYLE == "cartoon" else 0.225
    if seated:
        part(cube("Hips_" + c["id"], (base_x, base_y, 0.92 * h),
                  (0.38, 0.30, 0.22), pants), "hips")
        part(cylinder("Abdomen_" + c["id"], (base_x, base_y, 1.05 * h),
                      torso_r * 0.92, 0.24, shirt), "spine")
        part(sphere("Chest_" + c["id"], (base_x, base_y, 1.22 * h), torso_r,
                    shirt, scale=(1.24, 0.82, 0.98)), "chest")
        part(cyl_between("Neck_" + c["id"], (base_x, base_y, 1.34 * h),
                         (base_x, base_y, 1.45 * h), 0.068, skin), "neck")
        part(sphere("Head_" + c["id"], (base_x, base_y - 0.03, 1.52 * h),
                    head_r, skin, scale=(1.0, 0.94, 1.06)), "head")
        part(sphere("Hair_" + c["id"], (base_x, base_y - 0.01, 1.59 * h),
                    head_r * 1.02, hair, scale=(1.03, 1.0, 0.66)), "head")
        for side, sx in (("L", -1), ("R", 1)):
            part(cyl_between("UpperArm.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.21, base_y, 1.28 * h),
                             (base_x + sx * 0.30, base_y - 0.18, 1.02 * h),
                             0.070, shirt), "upper_arm." + side)
            part(cyl_between("Forearm.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.30, base_y - 0.18, 1.02 * h),
                             (base_x + sx * 0.31, base_y - 0.40, 0.82 * h),
                             0.060, shirt), "forearm." + side)
            part(sphere("Hand.%s_%s" % (side, c["id"]),
                        (base_x + sx * 0.31, base_y - 0.45, 0.79 * h),
                        0.082, skin), "hand." + side)
            part(cyl_between("Thigh.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.13, base_y, 0.90 * h),
                             (base_x + sx * 0.13, base_y - 0.44, 0.54 * h),
                             0.105, pants), "thigh." + side)
            part(cyl_between("Shin.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.13, base_y - 0.44, 0.54 * h),
                             (base_x + sx * 0.13, base_y - 0.30, 0.12 * h),
                             0.085, pants), "shin." + side)
            part(cube("Shoe.%s_%s" % (side, c["id"]),
                      (base_x + sx * 0.13, base_y - 0.40, 0.07),
                      (0.17, 0.32, 0.11), dark), "foot." + side)
        head_z, head_y = 1.52 * h, base_y - 0.03
    else:
        part(cube("Hips_" + c["id"], (base_x, 0, 0.91 * h),
                  (0.34, 0.27, 0.20), pants), "hips")
        part(cylinder("Abdomen_" + c["id"], (base_x, 0, 1.06 * h),
                      torso_r * 0.92, 0.30, shirt), "spine")
        part(sphere("Chest_" + c["id"], (base_x, 0, 1.23 * h), torso_r,
                    shirt, scale=(1.20, 0.80, 0.96)), "chest")
        part(cyl_between("Neck_" + c["id"], (base_x, 0, 1.36 * h),
                         (base_x, 0, 1.47 * h), 0.068, skin), "neck")
        part(sphere("Head_" + c["id"], (base_x, 0, 1.56 * h), head_r, skin,
                    scale=(1.0, 0.94, 1.06)), "head")
        part(sphere("Hair_" + c["id"], (base_x, 0.015, 1.63 * h),
                    head_r * 1.02, hair, scale=(1.03, 1.0, 0.66)), "head")
        for side, sx in (("L", -1), ("R", 1)):
            part(cyl_between("UpperArm.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.22, 0, 1.34 * h),
                             (base_x + sx * 0.33, 0, 1.07 * h),
                             0.071, shirt), "upper_arm." + side)
            part(cyl_between("Forearm.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.33, 0, 1.07 * h),
                             (base_x + sx * 0.36, -0.04, 0.81 * h),
                             0.060, shirt), "forearm." + side)
            part(sphere("Hand.%s_%s" % (side, c["id"]),
                        (base_x + sx * 0.36, -0.05, 0.77 * h),
                        0.083, skin), "hand." + side)
            part(cyl_between("Thigh.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.13, 0, 0.88 * h),
                             (base_x + sx * 0.13, 0, 0.49 * h),
                             0.107, pants), "thigh." + side)
            part(cyl_between("Shin.%s_%s" % (side, c["id"]),
                             (base_x + sx * 0.13, 0, 0.49 * h),
                             (base_x + sx * 0.13, 0, 0.11 * h),
                             0.086, pants), "shin." + side)
            part(cube("Shoe.%s_%s" % (side, c["id"]),
                      (base_x + sx * 0.13, -0.08, 0.065),
                      (0.17, 0.30, 0.11), dark), "foot." + side)
        head_z, head_y = 1.56 * h, 0.0

    if c["species"] == "robot":
        for o in list(parts):
            if o.name.startswith("Head_") or o.name.startswith("Hair_"):
                parts.remove(o)
                bpy.data.objects.remove(o, do_unlink=True)
        part(cube("Head_" + c["id"], (base_x, head_y, head_z),
                  (0.30, 0.27, 0.30), skin), "head")
        part(cylinder("Antenna_" + c["id"], (base_x, head_y, head_z + 0.25),
                      0.018, 0.18, dark), "head")
        part(sphere("AntennaTip_" + c["id"], (base_x, head_y, head_z + 0.36),
                    0.04, hair), "head")

    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    body = bpy.context.active_object
    body.name = "Character_" + c["id"]
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    body.parent = arm

    face_y = head_y - (0.155 if STYLE == "cartoon" else 0.145)
    eye_l = sphere("EyeL_" + c["id"],
                   (base_x - 0.058, face_y, head_z + 0.035), 0.033, white,
                   scale=(1.0, 0.6, 1.0))
    eye_r = sphere("EyeR_" + c["id"],
                   (base_x + 0.058, face_y, head_z + 0.035), 0.033, white,
                   scale=(1.0, 0.6, 1.0))
    pupil_l = sphere("PupilL_" + c["id"],
                     (base_x - 0.058, face_y - 0.023, head_z + 0.035),
                     0.016, dark, scale=(1.0, 0.5, 1.0))
    pupil_r = sphere("PupilR_" + c["id"],
                     (base_x + 0.058, face_y - 0.023, head_z + 0.035),
                     0.016, dark, scale=(1.0, 0.5, 1.0))
    mouth = sphere("Mouth_" + c["id"],
                   (base_x, face_y - 0.020, head_z - 0.060),
                   0.045, dark, scale=(1.15, 0.35, 0.55))
    nose = sphere("Nose_" + c["id"],
                  (base_x, face_y - 0.030, head_z - 0.010), 0.024, skin)
    brow_l = sphere("BrowL_" + c["id"],
                    (base_x - 0.058, face_y - 0.006, head_z + 0.083),
                    0.028, hair, scale=(1.25, 0.35, 0.32))
    brow_r = sphere("BrowR_" + c["id"],
                    (base_x + 0.058, face_y - 0.006, head_z + 0.083),
                    0.028, hair, scale=(1.25, 0.35, 0.32))
    for obj in (eye_l, eye_r, pupil_l, pupil_r, mouth, nose, brow_l, brow_r):
        bone_parent(obj, arm, "head")
    if c.get("gender") in ("female", "woman") and not robot:
        for sx in (-1, 1):
            side = sphere("HairSide%d_%s" % (sx, c["id"]),
                          (base_x + sx * head_r * 0.92, head_y + 0.015, head_z - 0.035),
                          head_r * 0.34, hair, scale=(0.62, 0.62, 1.75))
            bone_parent(side, arm, "head")
    if c["species"] in ("animal", "creature"):
        for sx in (-1, 1):
            ear = cone("Ear%d_%s" % (sx, c["id"]),
                       (base_x + sx * 0.10, head_y, head_z + 0.18),
                       0.05, 0.0, 0.16, hair,
                       rot=(0, math.radians(sx * 18), 0))
            bone_parent(ear, arm, "head")

    char_objs[c["id"]] = {
        "arm": arm, "body": body, "head_z": head_z, "mouth": mouth,
        "eye_l": eye_l, "eye_r": eye_r,
        "mouth_base": (1.15, 0.35, 0.55),
        "eye_base": (1.0, 0.6, 1.0),
        "base_x": base_x, "base_y": base_y, "seated": seated,
        "gender": c.get("gender", ""),
    }


# Per-frame render ----------------------------------------------------------
def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def shot_at(t):
    for s in DATA["shots"]:
        if s["start"] <= t < s["end"]:
            return s
    return DATA["shots"][-1]


for f in range(1, DATA["frames"] + 1):
    t = (f - 1) / DATA["fps"]
    shot = shot_at(t)
    span = max(0.01, shot["end"] - shot["start"])
    p = smoothstep((t - shot["start"]) / span)
    speaker = shot.get("speaker")
    cx = next((c["x"] for c in DATA["characters"] if c["id"] == speaker), None)
    if shot["kind"] in ("speaker", "reaction") and cx is not None:
        side = 1.0 if cx >= 0 else -1.0
        cam.location.x = cx + side * 0.45 - side * 0.12 * p
        cam.location.y = -3.75 + 0.35 * p
        cam.location.z = 1.70 + 0.03 * p
        target.location.x = cx
        target.location.y = -0.02
        target.location.z = 1.30 if seated else 1.48
        cam.data.lens = 58 if shot["kind"] == "speaker" else 52
    else:
        cam.location.x = 0.15 - 0.25 * p
        cam.location.y = -6.8 + 0.55 * p
        cam.location.z = 2.30 - 0.12 * p
        target.location.x = 0.0
        target.location.y = 0.05
        target.location.z = 1.10 + 0.03 * p
        cam.data.lens = 40

    for c in DATA["characters"]:
        o = char_objs[c["id"]]
        m = DATA["mouth"][c["id"]][f - 1]
        mo = DATA["motion"][c["id"]]
        nod = mo["nod"][f - 1]
        gest = mo["gesture"][f - 1]
        blink = mo["blink"][f - 1]
        mb = o["mouth_base"]
        o["mouth"].scale = (mb[0] * (1.0 + 0.55 * m), mb[1],
                            mb[2] * (0.30 + 1.75 * m))
        eb = o["eye_base"]
        o["eye_l"].scale = (eb[0], eb[1], eb[2] * max(0.12, blink))
        o["eye_r"].scale = (eb[0], eb[1], eb[2] * max(0.12, blink))
        pb = o["arm"].pose.bones
        try:
            pb["root"].location = (0.008 * math.sin(t * 2.2),
                                   0.006 * math.sin(t * 2.1), 0.0)
            pb["head"].rotation_mode = "XYZ"
            pb["head"].rotation_euler = (nod * 0.9, 0.0,
                                         0.035 * math.sin(t * 1.3))
            pb["chest"].rotation_mode = "XYZ"
            pb["chest"].rotation_euler = (0.0, 0.0,
                                          0.022 * math.sin(t * 1.1))
            arm_angle = 0.08 if o["seated"] else 0.05
            pb["upper_arm.L"].rotation_mode = "XYZ"
            pb["upper_arm.R"].rotation_mode = "XYZ"
            pb["upper_arm.L"].rotation_euler = (-gest * 0.20, 0.0, arm_angle)
            pb["upper_arm.R"].rotation_euler = (gest * 0.20, 0.0, -arm_angle)
            pb["forearm.L"].rotation_mode = "XYZ"
            pb["forearm.R"].rotation_mode = "XYZ"
            pb["forearm.L"].rotation_euler = (-gest * 0.14, 0.0, 0.0)
            pb["forearm.R"].rotation_euler = (gest * 0.14, 0.0, 0.0)
            if not o["seated"]:
                swing = 0.10 * math.sin(t * 2.0)
                pb["thigh.L"].rotation_mode = "XYZ"
                pb["thigh.R"].rotation_mode = "XYZ"
                pb["thigh.L"].rotation_euler = (swing, 0.0, 0.0)
                pb["thigh.R"].rotation_euler = (-swing, 0.0, 0.0)
        except Exception:
            pass
    frame_path = os.path.join(out_dir, "frame_%04d.png" % f)
    complete = False
    if os.path.exists(frame_path):
        try:
            with open(frame_path, "rb") as fp:
                fp.seek(-16, os.SEEK_END)
                complete = b"IEND" in fp.read()
        except OSError:
            complete = False
    if complete:
        print("FREAME %d/%d" % (f, DATA["frames"]), flush=True)
        continue
    scene.frame_set(f)
    scene.render.filepath = os.path.join(out_dir, "frame_%04d" % f)
    bpy.ops.render.render(write_still=True)
    print("FREAME %d/%d" % (f, DATA["frames"]), flush=True)

print("DONE %d" % DATA["frames"], flush=True)
'''


# ---------------------------------------------------------------------------
@dataclass
class BlenderRenderResult:
    ok: bool
    video: Path | None = None
    frames_dir: Path | None = None
    frames: int = 0
    duration: float = 0.0
    fps: int = 24
    resolution: tuple[int, int] = (960, 540)
    engine: str = "WORKBENCH"
    style: str = "cartoon"
    blender_version: str | None = None
    method: str = METHOD_NAME
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["video"] = str(self.video) if self.video else None
        d["frames_dir"] = str(self.frames_dir) if self.frames_dir else None
        d["resolution"] = list(self.resolution)
        return d


def render_conversation_3d(
    characters: list[Character],
    spans: list[Any],
    out_dir: Path,
    *,
    environment: str = "garden",
    duration: float | None = None,
    fps: int = 24,
    resolution: tuple[int, int] = (960, 540),
    engine: str = "WORKBENCH",
    style: str = "cartoon",
    timeout: float = 1800.0,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> BlenderRenderResult:
    status = find_blender()
    if not status.available or not status.executable:
        return BlenderRenderResult(False, error=status.error, notes=["3D unavailable; no fake 3D was produced."])
    if not characters:
        return BlenderRenderResult(False, error="no characters for 3D scene")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "3d_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if duration is None:
        duration = max([getattr(s, "end", 0.0) for s in spans] or [3.0])
    duration = max(1.0, float(duration))
    frames = max(1, int(round(duration * fps)))
    scene = _scene_props(environment)
    chars = [_character_3d(c, i, len(characters), style=style)
             for i, c in enumerate(characters)]
    shots = build_shot_plan(characters, spans, duration)
    data = {
        "frames_dir": str(frames_dir.resolve()),
        "width": resolution[0], "height": resolution[1], "fps": fps, "frames": frames,
        "duration": duration, "engine": engine, "style": style, "shots": shots,
        "sky": list(_hex_rgb(scene["sky"])), "floor": list(_hex_rgb(scene["floor"])),
        "props": scene["props"], "prop_color": list(_hex_rgb(scene["floor"])),
        "table": len(characters) >= 2,
        "characters": chars,
        "mouth": _mouth_curves(characters, spans, duration, fps),
        "motion": _motion_curves(characters, spans, duration, fps),
    }
    script_path = out_dir / "scene_3d.py"
    script = _SCRIPT.replace("__DATA_JSON__", json.dumps(data))
    script_path.write_text(script, encoding="utf-8")
    cmd = [status.executable, "--background", "--factory-startup", "--python", str(script_path)]
    log_path = out_dir / "blender.log"
    deadline = time.time() + timeout
    done = 0
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                if line.startswith("FREAME "):
                    try:
                        done = int(line.split()[1].split("/")[0])
                    except (IndexError, ValueError):
                        pass
                    if progress:
                        progress(done, frames)
                if cancel and cancel():
                    proc.terminate()
                    return BlenderRenderResult(False, error="cancelled by user",
                                               frames_dir=frames_dir, frames=done)
                if time.time() > deadline:
                    proc.terminate()
                    return BlenderRenderResult(False, error="Blender render timed out",
                                               frames_dir=frames_dir, frames=done)
            code = proc.wait()
    except (OSError, subprocess.SubprocessError) as exc:
        return BlenderRenderResult(False, error=f"failed to run Blender: {exc}")
    if code != 0:
        return BlenderRenderResult(False, error=f"Blender exited with code {code} (see {log_path})",
                                   frames_dir=frames_dir, frames=done)
    rendered = sorted(frames_dir.glob("frame_*.png"))
    if not rendered:
        return BlenderRenderResult(False, error="Blender produced no frames (see blender.log)",
                                   frames_dir=frames_dir)
    video = out_dir / "conversation_3d.mp4"
    try:
        run_ffmpeg([
            "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
            "-y", str(video),
        ], timeout=900)
    except FFmpegError as exc:
        return BlenderRenderResult(False, error=f"FFmpeg encode failed: {exc}",
                                   frames_dir=frames_dir, frames=len(rendered))
    return BlenderRenderResult(
        True, video=video, frames_dir=frames_dir, frames=len(rendered),
        duration=len(rendered) / fps, fps=fps, resolution=resolution,
        engine=engine, style=style, blender_version=status.version,
        notes=[f"Rendered with {status.version or 'Blender'}".strip(),
               "Real 3D geometry and material colours using Blender Workbench studio lighting; "
               "the scene's Sun/Area lights are used by the optional EEVEE path."
               if engine == "WORKBENCH" else
               "Real 3D geometry, scene lighting and materials (Blender EEVEE).",
               "Characters are real armature-rigged meshes: spine/chest/neck/head, upper arm, "
               "forearm, hand, thigh, shin and foot bones drive the geometry.",
               f"Multi-shot camera plan contains {len(shots)} cuts (two-shot, speaker close-ups "
               "and listener reaction shots).",
               "Lip-sync is viseme-driven mouth opening timed to measured speech, not a full "
               "facial performance or phoneme-accurate solve.",
               "Realistic mode uses natural palettes, realistic proportions and skin/fabric "
               "roughness. It is genuine 3D animation, not photoreal neural video."],
    )
