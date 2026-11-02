# Real capabilities

This document states exactly what FreeAnimate Studio produces, what is
procedural, and what requires optional software or models. The application
labels every output the same way in its UI and reports.

## Always available (no model downloads, no network)

| Feature | How it is produced | Honest label |
|---|---|---|
| Prompt understanding | Rule-based planner in `prompt_understanding.py`; extracts characters, setting, dialogue, camera, duration and style | "rule-based plan" |
| Characters | Procedural portraits and persistent `CharacterLibrary` JSON | "procedurally drawn" |
| Backgrounds | 17 procedural environment renderers (garden, room, beach, space, city, …); scenes 2+ use a deterministic variant of the same environment | "procedural environment" |
| Scene editing | Per-scene silent previews and `Re-render scene N`; only that segment is encoded again, then reassembled with the existing voice track, subtitles and audio | partial re-render, stated in the report |
| Settings | Browser form writes validated defaults to `config/config.json`; invalid values are rejected | real config edit |
| 2D animation | Rigged characters drawn frame-by-frame with Pillow, mouth visemes, blinking, gestures, head turns, camera motion | `rigged_2d_character_animation` |
| Lip-sync | Text → viseme timeline in `lipsync.py`, timed to measured audio | "viseme lip-sync" |
| Voices | Piper / espeak-ng / Windows SAPI when installed; otherwise correctly timed silence | engine name is reported |
| Subtitles | Measured-timing SRT/VTT, optionally burned in with libass | "audio_measured" |
| Video encode | Frames piped to FFmpeg (H.264 + AAC) | real MP4, validated with ffprobe |
| Ambience | Procedurally synthesized pad (if enabled) | "procedural ambience, not AI music" |
| Browser studio | FastAPI + Jinja2 on `127.0.0.1`, background render jobs with progress/cancel | local only |
| Export bundle | ZIP containing MP4, SRT/VTT, mixed audio, manifests, plan and reports | derived from real project files |
| Long films (5–25 min) | Separate `film` pipeline: prompt → story plan → measured TTS → per-scene 2D render with resume → concat/validate/QA | `film_mode`, `prompt_story` / `user` script source reported |

Nothing above sends data over the network. No model is downloaded
automatically.

## Long films (5–25 minutes)

`film` is a second production path tuned for longer runtimes. It is still the
same local 2D rig renderer and installed offline voices — not neural video.

* Durations clamp to 5 min minimum, 15 min default, 25 min maximum; a
  45-second preview renders one scene before committing.
* Narrative prompts are expanded offline into a story script from the prompt's
  own beats (names, actions, props, setting). This is deterministic prose, not
  a language model; it paraphrases the prompt and invents no facts.
* Narration is voiced by an explicit off-screen `Narrator`; detected characters
  speak short dialogue lines, so lip-sync and per-speaker voices are exercised.
* Scene cuts and subtitle timings come from **measured** TTS audio. Speech pace
  is calibrated with a real synthesis probe.
* Scene renders and voice tracks are resumable (`film/state.json` fingerprint);
  an interrupted render continues with `film resume`.
* Every film run writes a validation report and an automated QA report with a
  contact sheet. QA covers scenes, cast/assets, action motion, props,
  continuity, stage bounds, camera coverage, export streams/duration, audio
  level, subtitle timing/speakers, and black/dark frames. Mechanical checks do
  not prove artistic quality.

## Optional: genuine Blender 3D

If Blender is installed (or `BLENDER_EXE` points to it), the studio generates a
real `bpy` scene: armature-rigged 3D character meshes, layered environment
props, a table/chairs, lights, a camera and per-frame mouth motion derived from
the same viseme timeline. Frames are rendered by Blender and encoded with FFmpeg.

* The default `WORKBENCH` path is fast on CPUs: real geometry and material
  colours with Blender studio lighting. The optional `EEVEE` path uses the
  scene's Sun/Area lights but is much slower on this machine.
* Characters are **segmented meshes skinned to a real armature** (root, hips,
  spine, chest, neck, head, upper arm, forearm, hand, thigh, shin and foot
  bones). There is no IK solver or production facial rig.
* Dialogue scenes use a **multi-shot camera plan**: opening two-shot, per-speaker
  close-ups and listener reaction shots, cut in Blender.
* Two visual modes are supported: `3d_cartoon` and `realistic_3d` (natural
  palettes/proportions and skin/fabric roughness). Realistic mode is genuine
  stylized 3D, not photoreal neural video.
* Lip-sync is viseme-driven mouth opening, not phoneme-accurate facial animation.
* The 3D path is used only when Blender is actually detected and the render
  succeeds. Otherwise the studio falls back to 2D and says so in the report.
* Blender is GPL software; output you render is yours.
* Install: https://www.blender.org/download/ then add `blender` to PATH, set
  `BLENDER_EXE`, or place a portable copy under
  `%LOCALAPPDATA%\FreeAnimate\Blender\blender-*\blender.exe` (the studio
  searches that location automatically).
* The generated scene is a stylized 3D scene, not a photoreal or neural video.

## Local open-weight AI images (installed on this machine)

The image backend is real local inference:

* Runtime: PyTorch 2.15.0+cpu, diffusers 0.42, transformers, safetensors,
  huggingface_hub (Python 3.14 wheels).
* Model: **Tiny SD** (`segmind/tiny-sd`), CreativeML OpenRAIL-M, ~1.06 GB,
  installed at `models/tiny_sd`.
* Device: CPU only (Intel HD 620 has no CUDA). Expect roughly 1-3 minutes for a
  512x512 image at 8 steps; the adapter defaults Tiny SD to 8 steps.

Ways to use it:

* Browser: **Image lab** (`/images`) - prompt, seed, size, steps, local
  preview/download, and "use as project background".
* CLI: `python -m freeanimate.cli image "prompt" --out out.png`.
* Video: `--method ai_image` generates a local diffusion background for the
  scene, then renders the normal 2D animation over it. `--method realistic`
  uses the same still-image route when Blender is unavailable, and the report
  labels it as a still-image story, not AI video.
* Cinematic: `--method cinematic`, `photoreal` or `still_image` generates a
  genuine local AI still image and uses procedural camera motion over it. This
  is a **still-image story**, not AI-generated video; the label and report say
  so explicitly.
* Reuse: an image saved from the Image lab as
  `projects/<slug>/assets/custom_background.png` is reused automatically by the
  next render of that project (unless `ai_image` regeneration is requested).

Nothing downloads automatically. `freeanimate models <id>` prints size,
storage, hardware and licence notes; `freeanimate models <id> --download --yes`
is the explicit opt-in used to install Tiny SD. Always check the model card for
current licence terms. AI video generation still requires a capable GPU and is
reported as unavailable on this machine.

### ComfyUI

Set `COMFYUI_URL` (default `http://127.0.0.1:8188`). Image generation uses a
standard checkpoint -> CLIP encode -> KSampler -> VAE decode -> SaveImage workflow.
Video generation requires your own exported workflow JSON with a `{{PROMPT}}`
placeholder, because video node graphs are model-specific. The studio does not
invent a universal video workflow.

## Hardware profiles

The profile is derived from measured RAM/VRAM/cores, never from a GPU name.

* **A — basic**: procedural 2D/3D; avoid large diffusion/video models.
* **B — capable**: small image models on CPU/GPU are often practical.
* **C — strong**: GPU-accelerated image/video models are practical if licensed
  and installed.

Python 3.14 wheels for PyTorch 2.15 are available and installed here; no
separate 3.11/3.12 environment is required on this machine.

## What is explicitly *not* claimed

* Procedural characters are not neural "AI actors".
* Procedural backgrounds are not diffusion images.
* 2D viseme lip-sync is not phoneme-perfect facial capture.
* A request for 3D or AI output that cannot run is reported and downgraded, not
  faked.
* No cloud TTS, cloud LLM, paid API or credit system is used.
* The planner writes placeholder dialogue marked `source=generated`; it is
  editable, not presented as a human script.
* Long durations are not promised to be fast. Measured 3D speed and practical
  duration guidance are in
  [HARDWARE_AND_RENDERING_LIMITS.md](HARDWARE_AND_RENDERING_LIMITS.md).

## Verification

Every exported video is probed with ffprobe and checked for:

* existence and non-zero size,
* readable container, video and audio streams,
* requested resolution and frame rate,
* measured duration within tolerance,
* decodability,
* subtitle sidecar files and scene-segment count.

The machine-readable result is `reports/validation_report.json`; the honest
method and limitation summary is `reports/studio_report.json`.
