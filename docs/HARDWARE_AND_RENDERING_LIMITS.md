# Hardware and rendering limits

This page records what was measured on the machine where this build was tested.
It is intentionally conservative: it does not promise fast 45-minute AI video.

## Measured machine

| Item | Measured value |
|---|---|
| OS | Windows 10 (10.0.19045) |
| CPU | Intel Core i7-7500U @ 2.70 GHz, 2 cores / 4 threads |
| RAM | 8.4 GB |
| GPU | Intel HD Graphics 620 (integrated) |
| CUDA / VRAM | No CUDA; no usable dedicated VRAM for diffusion/video models |
| Disk free | ~42.9 GB at the time of testing |
| Python | 3.14.6 |
| FFmpeg / FFprobe | 9.0.2, with libass |
| Blender | 4.5.15 LTS, portable build found by the app |
| Profile | A — basic computer |

The installed `.venv` on this machine is broken (it has no working `pytest` and
`pip` fails with `ModuleNotFoundError: No module named 'pip._vendor.rich.markup'`).
Commands in this report use the working system `python`. Recreate the virtual
environment before relying on it.

## What can run here

| Mode | Status | What is actually produced |
|---|---|---|
| Rigged 2D animation | Available | Pillow-drawn skeletal 2D characters, viseme mouth motion, blinking, gestures, camera motion, H.264 + AAC MP4 |
| Animated illustration / narrated story | Available | Procedural backgrounds, overlays and FFmpeg camera motion |
| Blender 3D cartoon | Available | Real `bpy` scene, armature-rigged meshes, 3D props, multi-shot camera, CPU Workbench render |
| Blender realistic 3D humans | Available | Same real armature/3D pipeline with natural palettes, realistic proportions and skin/fabric roughness. **Not photoreal neural video.** |
| Cinematic / photoreal still-image story | Available | Genuine local Tiny SD still image + procedural camera motion; labelled as a still-image story, not AI video |
| Local AI still image | Available | Tiny SD CPU inference. Measured: 128×128/1 step ~13s; 512×512/1 step ~68s after model load (~1.5–2 min total). Multi-step 512×512 images take several minutes. |
| Local AI video | Not available | No compatible local video runtime/model/GPU. Requests are reported and downgraded; no clip is faked. |
| Photoreal AI human video | Not available | Not claimed, not simulated |

## Measured film (2D long-video) speed

Film mode renders at 854×480, 24 fps with the `pc_friendly` preset. Measured on
this machine: **352 seconds of finished video (16 scenes, ~8,800 frames) in
about 16 minutes of wall time**, roughly 2.7× realtime, including TTS, per-scene
encoding, concat, audio mix, validation, QA and thumbnails. A 20-second preview
measured ~7.9–9.4 rendered frames/second.

Practical consequences:

* A 5-minute film is roughly a 15–20 minute job at 480p.
* 720p `balanced` is slower than 480p; 1080p `high` is slower again (the preset
  changes encode cost, not character detail).
* Long films are CPU-bound and use one core per scene render; closing other
  applications matters on a 2-core laptop.
* The 25-minute maximum is an honest planning limit, not a promise of fast
  output. No 40–45 minute film is generated under the default configuration.

## Measured render speed

A 4-second, 854×480, 24 fps Blender cartoon (96 frames, 2 rigged characters,
cafe environment, Workbench engine) rendered and encoded in **29.3 seconds** on
this machine. That is about **3.3 rendered frames/second**, or roughly
**7 seconds of wall time per 1 second of 480p 3D video**.

Approximate 3D wall-clock estimates on this hardware:

| Output length | 480p, 24 fps | Notes |
|---|---|---|
| 30 seconds | ~3.5 minutes | practical |
| 1 minute | ~7 minutes | practical |
| 5 minutes | ~36 minutes | long but possible |
| 10 minutes | ~1.2 hours | use 2D for long work |
| 15 minutes | ~1.8 hours | not recommended for 3D |
| 30 minutes | ~3.6 hours | not practical for 3D here |
| 45 minutes | ~5.5 hours | not practical for 3D here |

These are estimates from one measured render; scene complexity changes them.
1080p Workbench rendering and EEVEE are slower. The default the app chooses is
Workbench, not EEVEE.

2D animation is considerably faster than Blender on this CPU, but exact speed
depends on scene count, resolution and subtitle burn-in. Use
`python -m freeanimate.cli benchmark` for a local measurement.

## Duration behaviour

The UI offers 30 seconds through 45 minutes plus custom seconds. Duration is
handled honestly:

* **Conversation videos** follow the measured speech. They are not padded by
  repeating lines or inventing speech. If the requested duration is much longer
  than the dialogue, the plan preview shows a warning and the export is shorter
  unless more lines are added.
* **Narrated/documentary projects** can target longer durations, but rendering
  time grows roughly linearly and is dominated by frame rendering and encoding.
* **45-minute 3D video is not promised to be quick.** On this machine it is
  estimated at several hours. 2D or still-image storytelling is the practical
  route for long outputs.
* The app does not silently turn a short scene into a long narrated video.

## Quality limits (stated plainly)

* 3D characters are built from segmented procedural meshes driven by a real
  armature (root, hips, spine, chest, neck, head, arms, hands, legs, feet).
  They are not scanned humans, neural avatars or production character rigs.
* Facial animation is viseme-style mouth opening plus blinking and head motion,
  not phoneme-accurate facial capture.
* Camera work now uses a multi-shot plan (two-shot, speaker close-ups, listener
  reactions), but there is no full cinematic lighting/comp pipeline or physics.
* Realistic 3D uses natural palettes and material roughness; it is still a
  stylized 3D render, not photoreal footage.
* Local AI images use Tiny SD on CPU and are slow; large SDXL/video models are
  not practical with 8 GB RAM and no CUDA.
* No cloud model, paid API or automatic large download is used.

## Recommended choices on this machine

| Goal | Recommended mode |
|---|---|
| Fast conversation video | Rigged 2D animation |
| 3D cartoon short (≤1–2 minutes) | Blender 3D cartoon, 480p |
| Realistic 3D short | Realistic 3D at 480p, keep it short |
| Long story / documentary | Narrated 2D illustration |
| AI background still | Local AI image, but expect minutes per image |
| Cinematic still-image story | Local AI still image + camera motion, not AI video |
| True AI video | Not practical here; use a CUDA machine with a licensed video model |
