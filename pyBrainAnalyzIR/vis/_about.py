"""Package metadata for the About dialog, parsed from the project README.

The authoritative source is ``pyNIRS_toolbox/README.md``.  Fallback values
are hard-coded so the GUI always has something to display even if the file
cannot be located.
"""

from __future__ import annotations

import os
import re
from typing import Dict

# ── hard-coded defaults (kept in sync with README.md) ───────────────────────
_DEFAULTS: Dict[str, str] = {
    "Name": "pyBrainAnalyzIR",
    "Version": "0.1.0",
    "Author": "T. Huppert",
    "Affiliation": "University of Pittsburgh, Departments of Radiology and Bioengineering",
    "Build Date": "2026-08-20",
    "License": "BSD 3-Clause",
    "Repository": "https://github.com/huppertt/pyNIRS_toolbox",
}


def _find_readme() -> str | None:
    """Walk up from this file's directory to locate the project README.md."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):  # look up to 6 levels up
        candidate = os.path.join(here, "README.md")
        if os.path.isfile(candidate):
            return candidate
        here = os.path.dirname(here)
    return None


def _parse_readme(path: str) -> Dict[str, str]:
    """Extract the metadata table rows from the README.md file."""
    result: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        # Match markdown table rows: | **Key** | Value |
        for m in re.finditer(
            r"\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|\n]+)\|", content
        ):
            key = m.group(1).strip()
            value = m.group(2).strip()
            result[key] = value
    except Exception:
        pass
    return result


def get_about_info() -> Dict[str, str]:
    """Return a dict of metadata fields for the About dialog."""
    info = dict(_DEFAULTS)
    readme = _find_readme()
    if readme:
        parsed = _parse_readme(readme)
        info.update(parsed)
    return info


def get_about_text() -> str:
    """Return a multi-line string suitable for display in the About dialog."""
    info = get_about_info()
    lines = [
        info.get("Name", "pyBrainAnalyzIR"),
        "",
        f"Version:      {info.get('Version', 'N/A')}",
        f"Build Date:   {info.get('Build Date', 'N/A')}",
        "",
        f"Author:       {info.get('Author', 'N/A')}",
        f"Affiliation:  {info.get('Affiliation', 'N/A')}",
        "",
        f"License:      {info.get('License', 'N/A')}",
        f"Repository:   {info.get('Repository', 'N/A')}",
    ]
    return "\n".join(lines)
