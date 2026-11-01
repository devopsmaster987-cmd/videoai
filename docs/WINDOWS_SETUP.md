# Windows setup (PowerShell)

Verified on Windows 11 with Python 3.14.6 and FFmpeg 9.0.2 (essentials build).

## 1. Python 3.10+

```powershell
python --version          # need 3.10+
pip install -r requirements.txt
```

Core deps are only Pillow + numpy (+ pyttsx3 for Windows narration, pytest for
tests). Everything else is stdlib or FFmpeg.

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
