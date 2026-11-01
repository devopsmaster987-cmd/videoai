"""Font detection for multilingual on-screen text and subtitles.

We never assume a font is installed (section 4.4).  For each writing-system
family we keep an ordered preference list of Windows font files; detection
checks ``C:\\Windows\\Fonts`` plus the user's fonts folder.  When nothing
suitable is found we return actionable installation guidance instead of
pretending rendering will succeed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Ordered preference per writing-system family.  First hit wins.
SCRIPT_FONT_PREFERENCE: dict[str, tuple[str, ...]] = {
    "latin": ("arial.ttf", "segoeui.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf"),
    "cyrillic": ("arial.ttf", "segoeui.ttf", "times.ttf"),
    "arabic": (
        "arial.ttf",
        "Jameel Noori Nastaleeq.ttf",
        "NotoNaskhArabic-Regular.ttf",
        "NotoNastaliqUrdu-Regular.ttf",
        "Nirmala.ttf",
    ),
    "devanagari": ("Nirmala.ttf", "mangal.ttf", "NotoSansDevanagari-Regular.ttf", "kalinga.ttf"),
    "bengali": ("Nirmala.ttf", "Vrinda.ttf", "NotoSansBengali-Regular.ttf"),
    "gurmukhi": ("Nirmala.ttf", "NotoSansGurmukhi-Regular.ttf"),
    "hebrew": ("arial.ttf", "Nirmala.ttf"),
    "cjk": ("msyh.ttc", "simsun.ttc", "msgothic.ttc", "malgun.ttf", "mingliub.ttc"),
    "thai": ("LeelawUI.ttf", "Nirmala.ttf"),
}

INSTALL_GUIDANCE = {
    "arabic": "Install a font with Arabic glyphs (Arial ships with Windows; for Urdu Nastaliq install 'Jameel Noori Nastaleeq' or Noto Nastaliq Urdu) and place the .ttf in C:\\Windows\\Fonts.",
    "devanagari": "Install 'Nirmala UI' (ships with Windows 10+) or Noto Sans Devanagari; place the .ttf in C:\\Windows\\Fonts.",
    "bengali": "Install 'Nirmala UI' or Noto Sans Bengali; place the .ttf in C:\\Windows\\Fonts.",
    "gurmukhi": "Install 'Nirmala UI' or Noto Sans Gurmukhi; place the .ttf in C:\\Windows\\Fonts.",
    "cjk": "Install 'Microsoft YaHei' (Chinese), 'Yu Gothic'/'MS Gothic' (Japanese), or 'Malgun Gothic' (Korean); place the file in C:\\Windows\\Fonts.",
    "hebrew": "Install a font with Hebrew glyphs (Arial ships with Windows).",
    "cyrillic": "Install Arial or another Unicode font covering Cyrillic.",
    "latin": "Install Arial or Segoe UI.",
    "thai": "Install Leelawadee UI or Noto Sans Thai.",
}


@dataclass(frozen=True)
class FontMatch:
    family: str
    filename: str
    path: Path
    covers_rtl: bool
    notes: str = ""


@dataclass(frozen=True)
class FontResult:
    """Outcome of a font search: either a match or honest guidance."""

    ok: bool
    family: str
    font: FontMatch | None = None
    guidance: str = ""
    searched_dirs: tuple[Path, ...] = ()

    @property
    def path(self) -> Path | None:
        return self.font.path if self.font else None


def font_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    windir = os.environ.get("WINDIR", r"C:\Windows")
    dirs.append(Path(windir) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        dirs.append(Path(userprofile) / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts")
    return [d for d in dirs if d.is_dir()]


def find_font_for_script(family: str, extra_dirs: list[Path] | None = None) -> FontResult:
    """Find a usable font file for a writing-system family.

    Never assumes a font is installed; returns :class:`FontResult` with
    ``ok=False`` and human-readable guidance when nothing is found.
    """
    prefs = SCRIPT_FONT_PREFERENCE.get(family)
    if not prefs:
        prefs = SCRIPT_FONT_PREFERENCE["latin"]
    dirs = font_search_dirs()
    if extra_dirs:
        dirs = list(extra_dirs) + dirs

    for d in dirs:
        for name in prefs:
            candidate = d / name
            if candidate.is_file():
                return FontResult(
                    ok=True,
                    family=family,
                    font=FontMatch(
                        family=family,
                        filename=name,
                        path=candidate,
                        covers_rtl=family in ("arabic", "hebrew"),
                    ),
                    searched_dirs=tuple(dirs),
                )
    guidance = INSTALL_GUIDANCE.get(family, INSTALL_GUIDANCE["latin"])
    return FontResult(ok=False, family=family, guidance=guidance, searched_dirs=tuple(dirs))


def find_fonts_for_scripts(families: list[str], extra_dirs: list[Path] | None = None) -> dict[str, FontResult]:
    return {fam: find_font_for_script(fam, extra_dirs) for fam in dict.fromkeys(families)}


# Map Windows font filenames to the family names libass/fontconfig expect.
FAMILY_NAMES: dict[str, str] = {
    "arial.ttf": "Arial",
    "arialbd.ttf": "Arial",
    "segoeui.ttf": "Segoe UI",
    "tahoma.ttf": "Tahoma",
    "verdana.ttf": "Verdana",
    "calibri.ttf": "Calibri",
    "times.ttf": "Times New Roman",
    "nirmala.ttf": "Nirmala UI",
    "nirmalab.ttf": "Nirmala UI",
    "mangal.ttf": "Mangal",
    "vrinda.ttf": "Vrinda",
    "kalinga.ttf": "Kalinga",
    "msyh.ttc": "Microsoft YaHei",
    "msyhbd.ttc": "Microsoft YaHei",
    "simsun.ttc": "SimSun",
    "simsunb.ttf": "SimSun-ExtB",
    "msyh.ttf": "Microsoft YaHei",
    "msgothic.ttc": "MS Gothic",
    "malgun.ttf": "Malgun Gothic",
    "malgunbd.ttf": "Malgun Gothic",
    "mingliub.ttc": "MingLiU-ExtB",
    "leelawui.ttf": "Leelawadee UI",
}


def family_name_for(filename: str) -> str:
    """Return the font family name for a Windows font filename."""
    return FAMILY_NAMES.get(filename.lower(), "Arial")


# Families whose glyphs require complex shaping (joining, ligatures,
# conjuncts).  Pillow without libraqm cannot shape these correctly, so
# on-screen text for these scripts is rendered by libass instead.
SHAPING_FAMILIES = frozenset({
    "arabic", "hebrew", "devanagari", "bengali", "gurmukhi",
})


def script_needs_shaping(family: str) -> bool:
    """True when a script family needs shaping Pillow cannot provide."""
    return family.lower() in SHAPING_FAMILIES
