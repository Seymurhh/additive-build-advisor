"""Build a LaTeX-typeset PDF of the technical report (docs/report.pdf).

Converts REPORT.md to PDF with pandoc + xelatex: strips the Mermaid code block
(the system-diagram PNG already conveys the architecture) and rewrites a few
Unicode symbols into robust LaTeX math so xelatex always renders them.

Requires pandoc and a LaTeX engine (xelatex). Run:  python examples/make_report_pdf.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    md = (ROOT / "REPORT.md").read_text()
    md = re.sub(r"```mermaid.*?```", "", md, flags=re.DOTALL)   # drop mermaid source
    # robust LaTeX for the few risky glyphs
    md = md.replace("ε*", r"$\varepsilon^{*}$").replace("ε", r"$\varepsilon$")
    md = md.replace("→", r" $\rightarrow$ ")

    build = ROOT / "_report_build.md"
    build.write_text(md)
    out = ROOT / "docs" / "report.pdf"
    cmd = [
        "pandoc", str(build), "-o", str(out),
        "--pdf-engine=xelatex",
        "--resource-path", str(ROOT),
        "--toc", "--number-sections",
        "--metadata", "title=Additive Build Advisor — Technical Report",
        "--metadata", "author=Seymur Hasanov",
        "-V", "geometry:margin=1in",
        "-V", "colorlinks=true", "-V", "linkcolor=RoyalBlue", "-V", "urlcolor=RoyalBlue",
        "-V", "fontsize=11pt",
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    finally:
        build.unlink(missing_ok=True)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
