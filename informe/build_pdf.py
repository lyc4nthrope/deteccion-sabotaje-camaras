"""Build informe/Informe_Auditoria_Tecnica.pdf from the section files (pandoc to HTML, Edge to PDF).

    python informe/build_pdf.py

Needs pandoc and Microsoft Edge (or Chrome). The images are read from results/ and embedded in the HTML.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFORME = ROOT / "informe"
OUT = INFORME / "Informe_Auditoria_Tecnica.pdf"
BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]
TITLE = """---
title: "Auditoría híbrida de modelos de visión para la detección de sabotaje en cámaras de vigilancia"
subtitle: "Informe de Auditoría Técnica"
author: "Grupo 7 · Seguridad. Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga"
date: "Docente: Ing. Ana Maria Tamayo Ocampo, Mg. Noviembre de 2026"
lang: es
---
"""
CSS = """
@page { size: A4; margin: 2cm 1.8cm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; max-width: none; color: #1a1a1a; }
header#title-block-header { text-align: center; margin: 5cm 0 3cm; page-break-after: always; }
h1.title { font-size: 22pt; } p.subtitle { font-size: 15pt; color: #444; } p.author, p.date { font-size: 11pt; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; page-break-before: always; margin-top: 0; }
nav#TOC { page-break-after: always; } nav#TOC ul { list-style: none; padding-left: 1em; }
h2 { font-size: 13pt; margin-top: 1.4em; } h3 { font-size: 11.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 0.8em 0; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3px 6px; vertical-align: top; } th { background: #eee; }
img { max-width: 100%; display: block; margin: 0.6em auto; } code { font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f3f3f3; padding: 6px; font-size: 8.5pt; white-space: pre-wrap; }
"""


def main() -> None:
    parts = sorted((INFORME / "secciones").glob("[0-1][0-9]_*.md"))
    parts += [INFORME / "referencias.md", INFORME / "secciones" / "A_reproducir.md"]
    missing = [p for p in parts if not p.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    body = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts)
    browser = next((b for b in BROWSERS if b.exists()), None)
    if browser is None:
        sys.exit("Edge or Chrome not found")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "informe.md").write_text(TITLE + "\n" + body, encoding="utf-8")
        (tmp / "estilo.css").write_text(CSS, encoding="utf-8")
        html = tmp / "informe.html"
        subprocess.run(
            ["pandoc", str(tmp / "informe.md"), "-f", "gfm+pipe_tables+yaml_metadata_block", "-t", "html5", "-s",
             "--toc", "--toc-depth=2", "--embed-resources", "--css", str(tmp / "estilo.css"),
             f"--resource-path={INFORME / 'secciones'}", "-o", str(html)],
            check=True,
        )
        subprocess.run(
            [str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT}", html.as_uri()],
            check=True, capture_output=True,
        )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(parts)} parts)")


if __name__ == "__main__":
    main()
