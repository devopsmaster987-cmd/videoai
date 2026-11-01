# FreeAnimate AI

Free, offline, multilingual AI animation and YouTube video creator — a
terminal-based tool that turns a prompt (or your own script) into a validated
MP4 with narration, burned-in subtitles, camera movement, and procedural
background ambience, plus three scored YouTube thumbnail options. No paid APIs,
no cloud accounts, no watermarks.

Implemented from the master prompt in `b.txt` as a modular package
(`freeanimate/`), improving on the earlier monolithic "pid" version.

## What it does — and what it does not

**It does:** generate honest template scripts, plan scenes, narrate via local
TTS, burn shaped subtitles, render procedural animated illustrations and motion
graphics with FFmpeg, mix audio, validate the MP4, and produce three distinct
YouTube thumbnail designs scored and ranked with documented heuristics.

**It does not:** produce photographic or AI-generated video footage. All visuals
are procedural art (layered gradients, silhouettes, diagrams, timelines) or
animated still images. It does not invent statistics, dates, or citations, and
it does not promise click-through rates — thumbnail scores are design
assessments, not CTR predictions.

### Optional local AI models

No shipped capability currently requires an optional AI model. The
`ai_video` animation mode is accepted by the configuration but **falls back to
hybrid procedural rendering** because no compatible free backend is bundled —
the output is never silently labeled as AI-generated footage. TTS quality
depends on locally installed voices (see below); everything else runs with
Pillow, numpy, and FFmpeg only.

## Installation (Windows)

```powershell
# 1. Install dependencies (see docs/WINDOWS_SETUP.md for full details)
pip install -r requirements.txt

# 2. FFmpeg must be on PATH (docs/WINDOWS_SETUP.md covers winget/scoop install)

# 3. Check your machine
python -m freeanimate doctor
```

**Required dependencies:** Python 3.10+, Pillow, numpy, FFmpeg (with `libass`
subtitles filter for shaped text).

**Optional dependencies:** `pyttsx3` (Windows SAPI narration), Piper or
espeak-ng (better/other-language voices). Without any TTS engine, narration is
**silently timed** and the limitation is reported — never gibberish.

## Quick start

```powershell
# Create a video end-to-end
python -m freeanimate create `
  --prompt "Create a cinematic documentary explaining the history of the universe, from the Big Bang to the formation of galaxies. Use engaging narration, consistent visuals, animated scenes, subtitles, and suitable background music." `
  --name "History of the Universe" --duration 5 --language en --voice sapi `
  --resolution 1080p --music procedural --yes
```

The final MP4 lands in `projects/<name>/renders/freeanimate_final.mp4` with a
validation report in `projects/<name>/reports/` and three thumbnails in
`projects/<name>/thumbnails/`. A launcher is also installed as
`freeanimate.bat`.

## CLI commands

| Command | Purpose |
| --- | --- |
| `create` | Interactive or flagged end-to-end creation |
| `plan` / `scenes` / `narration` | Inspect or run individual stages |
| `generate` / `resume` / `export` | Render a project (resumes completed stages) |
| `status <project>` | Stage list, narration engine, export path, thumbnail selection |
| `thumbnails <project> [show\|generate\|regenerate\|select\|open]` | Manage YouTube thumbnails without re-rendering |
| `projects list` | All projects |
| `providers list` | TTS engines probed at runtime + FFmpeg info |
| `doctor` | Environment check (FFmpeg filters, fonts, TTS, disk) |
| `benchmark` | Measured render throughput on this machine |
| `languages` | 19 supported languages (editable `config/languages.json`) |
| `clean-cache` | Remove cached frames/assets |

### Useful `create` flags

| Flag | Purpose |
| --- | --- |
| `--duration <minutes>` | Video length (e.g. `1`, `5`) |
| `--language <code>` | Project language (`en`, `ur`, `ar`, `hi`, …) applied to all roles |
| `--script-language` / `--narration-language` / `--subtitle-language` / `--onscreen-language` / `--thumbnail-language` | Per-role language overrides |
| `--voice <auto\|piper\|espeak\|sapi\|silence>` | Narration engine |
| `--resolution <720p\|1080p\|4k>` | Export size (1080p and 4K both supported) |
| `--animation-mode hybrid\|illustration\|motion_graphics` | Visual style |
| `--music none\|procedural` | Background ambience |
| `--no-burn-subtitles` | Keep subtitles as files only |
| `--script-file yourscript.txt` | Use your own script (sections with `# Title` lines) |
| `--no-thumbnails` | Skip thumbnail generation |
| `--auto-select-thumbnail` | Copy the recommended thumbnail to `youtube_thumbnail.jpg` automatically |
| `--yes` | Skip confirmation |

## How it works

1. **Script** — offline deterministic template engine (or your script file via
   `--script-file`). Kinds: documentary, science, explainer, story, tutorial,
   listicle, opinion. Scripts are honest general narration: no invented dates,
   citations, or statistics. User-provided `facts` are woven in verbatim.
2. **Scenes** — manifest with varied camera moves (zoom in/out, pan, drift)
   and transitions; durations resynced to measured narration.
3. **Narration** — TTS via Piper (best), espeak-ng, or Windows SAPI. If no
   engine has a voice for the language, the narration is **silently timed**
   and the limitation is reported (never gibberish).
4. **Subtitles** — SRT + WebVTT for upload, plus per-scene ASS files burned
   in with libass (HarfBuzz shaping + FriBidi bidi, so Urdu/Arabic/Indic
   render correctly). On-screen titles for shaping scripts are also rendered
   via libass, since Pillow cannot shape those scripts.
5. **Visuals** — procedural PIL illustrations (layered gradients, silhouettes,
   sun/moon, particles) and motion graphics (timelines, bar charts, lists,
   quote cards), animated with FFmpeg `zoompan` + fades.
6. **Audio** — narration mixed with optional procedural ambient pad (synthesized,
   labeled as ambience), ducked under speech.
7. **Assembly & validation** — concat, final mix, then FFprobe validation:
   size, streams, resolution, fps, duration tolerance (±15%, WARN only),
   decodability, subtitle files, scene coverage. JSON + TXT reports written.
8. **Thumbnails** — three distinct 1280×720 JPEG designs composed from the
   video's own scene assets, scored and recommended (see below).

## Thumbnails

After a successful export the pipeline generates **three distinct thumbnail
options** in `projects/<name>/thumbnails/`:

| # | Concept | Design |
| --- | --- | --- |
| 1 | High Click Appeal | Bold hero scene, dark scrim, large high-contrast title, content-kind tag, accent edge bar |
| 2 | Cinematic & Professional | Blurred darkened backdrop with a sharp framed inset, thin light border, understated caption |
| 3 | Creative Alternative | Diagonal split-panel collage of two scene frames with geometric accents |

Each is a valid 1280×720 RGB JPEG. The report
`thumbnails/thumbnail_recommendations.json` records per-option scores,
strengths/weaknesses, the recommendation with reasoning, and any warnings.

### How the recommendation works

Eight documented criteria (max 100 total) are measured **from the actual image
pixels and project metadata** — never random, never fabricated:

| Criterion | Max |
| --- | --- |
| Topic relevance (text/title keyword overlap) | 25 |
| Visual clarity (edge-density measurement) | 20 |
| Text readability (local contrast behind glyphs) | 15 |
| Contrast (Michelson ratio) | 10 |
| Composition (rule-of-thirds focal energy) | 10 |
| Audience relevance (mood↔content-kind affinity) | 10 |
| Distinctiveness (pixel difference vs the other options) | 5 |
| Honesty & accuracy (no invented numbers/hype words) | 5 |

Options failing critical quality checks (blank, wrong dimensions, not JPEG,
black bars, unreadable) are disqualified. The highest-scoring option **that
passed validation** is recommended, with reasoning written into the report.
Scores are heuristic design assessments — **not predictions of YouTube
click-through rate, views, or growth.**

### Generating, selecting, and regenerating thumbnails

```powershell
# View the report + recommendation
python -m freeanimate thumbnails <project>

# Accept the recommendation (copies to thumbnails/youtube_thumbnail.jpg)
python -m freeanimate thumbnails <project> select

# Pick option 2 explicitly; --yes confirms replacing a previous selection
python -m freeanimate thumbnails <project> select --index 2 --yes

# Rebuild one option (or all three) from existing scene assets —
# the rendered video is never touched or re-rendered
python -m freeanimate thumbnails <project> regenerate --index 1

# Open the thumbnails folder in Explorer
python -m freeanimate thumbnails <project> open
```

A previously selected `youtube_thumbnail.jpg` is **never overwritten without
confirmation** (`--yes`). Regeneration only rewrites the option files and the
report; if the selected option itself was regenerated, a warning tells you to
re-select to adopt the new version.

Thumbnail failures are reported honestly as warnings and **never fail the
video project** — the export stays valid, and the report tells you how to
retry. For RTL scripts (Urdu/Arabic/Hebrew/Persian), thumbnail text is shaped
via FFmpeg+libass; when correct shaping cannot be verified the thumbnail is
delivered **text-free** rather than shipping broken disconnected glyphs.

Thumbnail language follows the project language by default; override with
`--thumbnail-language <code>` at creation or `--language` on `regenerate`.

## Using an existing script

```powershell
python -m freeanimate create --prompt "My topic" --script-file myscript.txt --duration 5 --yes
```

The script file uses `# Section Title` lines followed by narration paragraphs.
Scene durations are planned against your text; nothing is rewritten.

## Languages

```powershell
python -m freeanimate languages          # list all 19 languages
python -m freeanimate create --language ur --narration-language en ...
```

`--language` sets the project language for every role; per-role flags
(`--script-language`, `--narration-language`, `--subtitle-language`,
`--onscreen-language`, `--thumbnail-language`) override individual roles.
The language table is data-driven: edit `config/languages.json` to add or
adjust entries without touching code. Subtitles and on-screen titles for
shaping scripts (Urdu, Arabic, Hindi, …) are rendered through libass.

## Resuming a failed or interrupted project

```powershell
python -m freeanimate status <project>     # shows which stage needs work
python -m freeanimate resume <project>     # continues from the first incomplete stage
```

Completed stages (script, scenes, narration, subtitles, assets, render,
export) are reused; only the unfinished work runs. `--fresh` on `generate`
forces a full rebuild.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `doctor` reports FFmpeg missing | Install FFmpeg and add it to PATH (docs/WINDOWS_SETUP.md) |
| Subtitles missing / not shaped | Ensure FFmpeg has the `libass` `subtitles` filter (`doctor` checks this) |
| Narration is silent | Install a TTS voice: Piper, espeak-ng, or a Windows SAPI voice for the language (`providers list` shows what was probed) |
| Duration WARN (±15%) | Normal for fast TTS voices; content is extended with additional scenes, never silence-padded |
| `thumbnail generation failed` warning | Video is still valid; run `freeanimate thumbnails generate <project>` after checking scene assets exist |
| Thumbnail text missing on RTL languages | Intentional: shaping could not be verified, so a text-free thumbnail is shipped (b.txt 12.5) |
| Project stuck / crashed mid-run | `freeanimate resume <project>` — completed stages are skipped |
| Low disk / slow machine | `--low-resource` flag, `clean-cache`, and 720p resolution |

## Honest limitations

- **Template scripts are general, not factual reporting.** They describe
  structure and evidence without fabricating specifics. For factual content,
  pass your own script (`--script-file`) or facts.
- **TTS quality depends on installed voices.** Windows SAPI ships two English
  voices; other languages need Piper voices or espeak-ng (or a Windows voice).
  Without one, narration is silent but correctly timed.
- **Visuals are procedural art**, not photographic/AI-generated imagery. The
  tool does not provide unlimited AI-generated video footage.
- **Thumbnail scores are design heuristics**, not CTR or performance
  predictions; the disclaimer is embedded in every report.
- **Duration tolerance is ±15%** (`WARN`, not failure). Shortfall is fixed by
  composing *additional* content scenes — never by silence-padding or repeating
  scenes.
- Rendering is CPU FFmpeg (`zoompan`); 5 min @1080p is roughly 1–2 min of
  render time on a typical machine (see `benchmark`).

## Development

```powershell
python -m pytest tests/ -q     # 158 tests: unit + end-to-end render + thumbnails
```

Project layout: `freeanimate/` (package), `tests/`, `config/` (editable
defaults + languages), `projects/` (output), `cache/` (scratch frames).

MIT licensed. All dependencies are free/open source.
