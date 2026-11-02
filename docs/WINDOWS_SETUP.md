# Windows setup (PowerShell)

Verified on Windows 11 with Python 3.14.6 and FFmpeg 9.0.2 (essentials build).

## 1. Python 3.10+

```powershell
python --version          # need 3.10+
pip install -r requirements.txt
```

Core deps are Pillow, numpy, pyttsx3 (Windows narration), FastAPI, Uvicorn and
Jinja2 for the local studio, plus pytest for tests. Everything else is stdlib
or FFmpeg. The optional local AI extras (`pip install -e ".[ai]"`) need Python
3.11/3.12 today; Python 3.14 may not have PyTorch wheels yet.

## 2. FFmpeg + FFprobe (with libass)

FFmpeg must include the `zoompan` and `subtitles` (libass) filters.

```powershell
# Chocolatey (recommended)
choco install ffmpeg -y
ffmpeg -version
ffprobe -version
ffmpeg -filters | findstr "zoompan subtitles"
```

If not on PATH after install, close and reopen PowerShell, or add the
Chocolatey bin directory (`C:\ProgramData\chocolatey\bin`) to PATH.
Portable builds (gyan.dev "essentials") also work — put `bin\` on PATH.

## 3. Check everything

```powershell
python -m freeanimate doctor
```

This verifies Python, FFmpeg version + required filters, Pillow, numpy, disk
space, TTS engines, and per-script fonts (latin, arabic, devanagari, cjk,
cyrillic).

## 4. Narration voices (optional but recommended)

Order of preference (auto-detected):

1. **Piper** — best offline neural TTS. Install: download a `piper` Windows
   binary from https://github.com/rhasspy/piper/releases, put it on PATH, and
   download a voice `.onnx` + `.json` into a folder; point FreeAnimate at it
   (see `python -m freeanimate providers list` for the searched location).
2. **espeak-ng** — robotic but supports many languages incl. Urdu/Arabic/Hindi:
   `choco install espeak-ng -y` (https://github.com/espeak-ng/espeak-ng).
3. **Windows SAPI** — built in; ships with two English voices (David, Zira).
   Extra voices: Settings > Time & Language > Speech, or the "Add language"
   optional speech packages.
4. **Silence** — if nothing matches the project's narration language, audio is
   silent but correctly timed, and the export report says so explicitly.

Verify what is available:

```powershell
python -m freeanimate providers list
```

## 5. Fonts

Windows ships fonts covering all bundled scripts: Arial (Latin/Arabic),
Nirmala UI (Devanagari/Urdu fallback), Microsoft YaHei (Chinese), Malgun
Gothic (Korean), MS Gothic (Japanese). For nicer Urdu, install
"Jameel Noori Nastaleeq" or "Noto Nastaliq Urdu" into
`C:\Windows\Fonts` — the font probe picks it up automatically
(`config/languages.json` lists per-language font hints).

## 6. Create your first video

```powershell
python -m freeanimate create --prompt "Explain how rainbows form" `
  --name "Rainbows" --duration 2 --language en --voice sapi `
  --resolution 720p --yes
```

Then open `projects\rainbows\renders\freeanimate_final.mp4`.

Troubleshooting: `python -m freeanimate status <project>` shows stage-by-stage
state; interrupted runs resume with `python -m freeanimate resume <project>`.

## 7. Browser studio (recommended)

One-time setup, then double-click the scripts:

| Script | What it does |
| --- | --- |
| `install.bat` | Creates `.venv`, upgrades pip, installs `requirements.txt` |
| `start.bat` | Starts the studio at http://127.0.0.1:8000 and opens the browser |
| `stop.bat` | Stops the process listening on the studio port |
| `check-health.bat` | Runs `doctor`, hardware report and TTS provider list |

Manual equivalent:

```powershell
python -m freeanimate serve --open
```

From the browser:

1. **New video** — describe the scene, choose resolution/method, preview the
   plan (real parsed characters, dialogue, capabilities), then render. Jobs run
   in background threads; closing the tab does not stop them.
2. **Projects** — watch the MP4, read measured dialogue timing, validation
   report and honest limitations.
3. **Scenes** — each scene segment is listed with a silent preview and a
   `Re-render scene N` button. Only that segment is encoded again; the voice
   track, subtitles, audio mix and other scenes are reused.
4. **Export bundle (ZIP)** — MP4, SRT/VTT, mixed audio, manifests and reports.
5. **Settings** — validated edits to the real `config/config.json` (resolution,
   frame rate, CRF, voice engine/speed, subtitle wrapping, ambience). Invalid
   values are rejected and the previous file is kept.
6. **Hardware / Models** — measured capabilities and model download *plans*
   only. Nothing is downloaded without an explicit `python -m freeanimate models
   download <id> --confirm` and a licence review.

CLI equivalents:

```powershell
python -m freeanimate make "A gardener and a robot discuss tomatoes" `
  --name "Garden Conversation" --resolution 720p --music procedural
python -m freeanimate hardware
python -m freeanimate models plan sd_turbo
python -m freeanimate understand "prompt text"
```

## 8. Optional genuine 3D (Blender)

Install Blender 3.x/4.x from https://www.blender.org/download/ and either add
it to PATH, set `BLENDER_EXE`, or use a portable copy under
`%LOCALAPPDATA%\FreeAnimate\Blender\blender-*\blender.exe`.
`python -m freeanimate hardware --json` reports whether it was found. With
Blender present, `make --method 3d` runs a real `blender --background --python`
render; otherwise the studio says so and uses the 2D rig � it never labels 2D
output as 3D.

The default Workbench path is CPU-friendly and uses real geometry/material
colours plus studio lighting. It does not use the scene's Sun/Area lights (the
optional EEVEE path does, but is much slower on this hardware). Characters are
procedural segmented meshes, not skeletal rigs; lip-sync is basic viseme-driven
mouth opening, and the camera is a single continuous move.

## 9. Local AI images (installed on this machine)

This machine now has a working local image backend:

* `torch 2.15.0+cpu`, `diffusers`, `transformers`, `accelerate`, `safetensors`,
  `huggingface_hub`.
* Tiny SD model (`segmind/tiny-sd`, CreativeML OpenRAIL-M, ~1.06 GB) at
  `models/tiny_sd`.

Try it:

```text
python -m freeanimate hardware --json
python -m freeanimate image "A warm sunlit garden with flowers and a fountain, stylized illustration" --out test.png
python -m freeanimate.cli serve --port 8000
```

Then open `http://127.0.0.1:8000/images` for the Image lab. You can generate,
preview and download images, and save one as a project's custom background for
the next render. `--method ai_image` uses the same local model to build scene
backgrounds. AI video generation is still unavailable without a CUDA GPU.

Nothing else is downloaded automatically; re-check the model card licence before
downloading other checkpoints (`python -m freeanimate models --task image`).

