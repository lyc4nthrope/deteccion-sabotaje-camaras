"""Build informe/Informe_Auditoria_Tecnica.docx inside the Universidad del Quindío template.

    python informe/build_docx.py

The template (informe/plantilla_uniquindio.docx) keeps its cover art, headers, footers, page numbers, fonts and back
cover. Only four things change: the title block of the cover, the running header text, the back-cover name and the
body, which is the Markdown of informe/secciones/ turned into Word paragraphs, lists, tables and figures. The text of
the sections is not edited.
"""

from __future__ import annotations

import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
INFORME = ROOT / "informe"
TEMPLATE = INFORME / "plantilla_uniquindio.docx"
OUT = INFORME / "Informe_Auditoria_Tecnica.docx"

TITLE = "Auditoría híbrida de modelos de visión para la detección de sabotaje en cámaras de vigilancia"
SUBTITLE_LINES = [
    ("Informe de Auditoría Técnica", True, 28),
    ("Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga", False, 24),
    ("Grupo 7 · Seguridad", False, 24),
    ("Docente: Ing. Ana Maria Tamayo Ocampo, Mg.", False, 24),
    ("Noviembre de 2026", False, 24),
]
HEADER_TEXT = "Auditoría de sabotaje en cámaras"
BACK_COVER_NAME = "Grupo 7 · Seguridad"

FONT = '<w:rFonts w:ascii="Helvetica Neue" w:cs="Helvetica Neue" w:eastAsia="Helvetica Neue" w:hAnsi="Helvetica Neue"/>'
MONO = '<w:rFonts w:ascii="Consolas" w:cs="Consolas" w:eastAsia="Consolas" w:hAnsi="Consolas"/>'
TEXT_WIDTH_DXA = 12240 - 2 * 1701  # page width minus the template margins
TEXT_WIDTH_EMU = TEXT_WIDTH_DXA * 635

# ---------------------------------------------------------------- styles and numbering added to the template

STYLES = f"""
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Cuerpo"><w:name w:val="Cuerpo"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>
<w:rPr>{FONT}<w:color w:val="333333"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Lista"><w:name w:val="Lista"/><w:basedOn w:val="Cuerpo"/>
<w:pPr><w:spacing w:after="60"/><w:jc w:val="left"/></w:pPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Leyenda"><w:name w:val="Leyenda"/><w:basedOn w:val="Cuerpo"/>
<w:pPr><w:spacing w:before="60" w:after="200"/><w:jc w:val="left"/></w:pPr><w:rPr><w:color w:val="595959"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="TextoTabla"><w:name w:val="Texto de tabla"/><w:basedOn w:val="Cuerpo"/>
<w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr><w:rPr><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Codigo"><w:name w:val="Código"/><w:basedOn w:val="Cuerpo"/>
<w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:left="170"/><w:jc w:val="left"/></w:pPr>
<w:rPr>{MONO}<w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Cita"><w:name w:val="Cita"/><w:basedOn w:val="Cuerpo"/>
<w:pPr><w:pBdr><w:left w:val="single" w:sz="18" w:space="8" w:color="BFBFBF"/></w:pBdr><w:ind w:left="340"/></w:pPr><w:rPr><w:i/><w:iCs/></w:rPr></w:style>
<w:style w:type="character" w:customStyle="1" w:styleId="CodigoCar"><w:name w:val="Código Car"/><w:rPr>{MONO}<w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>
"""

HEADINGS = {  # style id -> (size in half points, color, space before, space after, page break before)
    "Heading1": (40, "595959", 0, 240, True),
    "Heading2": (28, "595959", 360, 120, False),
    "Heading3": (24, "333333", 280, 100, False),
    "Heading4": (22, "333333", 220, 80, False),
}

NUM_BULLET, NUM_ORDERED = 1, 2


def _abstract_num(abstract_id: int, bullet: bool) -> str:
    levels = []
    for level in range(3):
        left, hanging = 567 + 425 * level, 283
        fmt = "bullet" if bullet else ("decimal" if level == 0 else "lowerLetter")
        text = ("•" if level == 0 else "–" if level == 1 else "·") if bullet else f"%{level + 1}."
        font = '<w:rPr><w:rFonts w:ascii="Helvetica Neue" w:hAnsi="Helvetica Neue" w:hint="default"/></w:rPr>' if bullet else ""
        levels.append(
            f'<w:lvl w:ilvl="{level}"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/><w:lvlText w:val="{text}"/>'
            f'<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="{left}" w:hanging="{hanging}"/></w:pPr>{font}</w:lvl>'
        )
    return f'<w:abstractNum w:abstractNumId="{abstract_id}"><w:multiLevelType w:val="hybridMultilevel"/>{"".join(levels)}</w:abstractNum>'


# ---------------------------------------------------------------- Markdown to WordprocessingML


def _run(text: str, bold=False, italic=False, code=False) -> str:
    props = ""
    if code:
        props += '<w:rStyle w:val="CodigoCar"/>'
    if bold:
        props += "<w:b/><w:bCs/>"
    if italic:
        props += "<w:i/><w:iCs/>"
    props = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{props}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def inline_runs(children) -> str:
    out, bold, italic = [], 0, 0
    for tok in children:
        if tok.type == "text":
            out.append(_run(tok.content, bold > 0, italic > 0))
        elif tok.type == "code_inline":
            out.append(_run(tok.content, bold > 0, italic > 0, code=True))
        elif tok.type in ("softbreak",):
            out.append(_run(" ", bold > 0, italic > 0))
        elif tok.type == "hardbreak":
            out.append("<w:r><w:br/></w:r>")
        elif tok.type == "strong_open":
            bold += 1
        elif tok.type == "strong_close":
            bold -= 1
        elif tok.type == "em_open":
            italic += 1
        elif tok.type == "em_close":
            italic -= 1
        elif tok.type in ("link_open", "link_close"):
            continue
        elif tok.type == "html_inline":
            continue
        else:
            out.append(_run(tok.content or "", bold > 0, italic > 0))
    return "".join(out)


def plain_text(children) -> str:
    return "".join(t.content for t in children if t.type in ("text", "code_inline"))


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


class Builder:
    def __init__(self, base: Path):
        self.base = base
        self.body: list[str] = []
        self.media: list[tuple[str, Path]] = []  # (target name, source)
        self.relationships: list[str] = []
        self.nums: list[str] = []
        self.next_num = 10
        self.drawing_id = 100

    # -- blocks
    def paragraph(self, style: str, runs: str, extra_ppr: str = "") -> None:
        self.body.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{extra_ppr}</w:pPr>{runs}</w:p>')

    def picture(self, src: str, alt: str) -> None:
        path = (self.base / src).resolve()
        if not path.exists():
            sys.exit(f"missing figure: {path}")
        width_px, height_px = png_size(path)
        width = min(TEXT_WIDTH_EMU, width_px * 9525)
        height = int(width * height_px / width_px)
        index = len(self.media) + 1
        target, rid = f"media/informe_{index}.png", f"rIdInforme{index}"
        self.media.append((f"word/{target}", path))
        self.relationships.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
        )
        self.drawing_id += 1
        drawing = (
            f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{width}" cy="{height}"/>'
            f'<wp:docPr id="{self.drawing_id}" name="Figura {index}" descr="{escape(alt, {chr(34): "&quot;"})}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{self.drawing_id}" name="informe_{index}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>"
        )
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="Cuerpo"/><w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/></w:pPr>'
            f"<w:r>{drawing}</w:r></w:p>"
        )

    def table(self, rows: list[list[list]], header_rows: int) -> None:
        columns = max(len(r) for r in rows)
        weights = [4] * columns
        for row in rows:
            for c, cell in enumerate(row):
                text = plain_text(cell)
                longest_word = max((len(w) for w in text.split()), default=0)
                weights[c] = max(weights[c], min(len(text), 38), longest_word + 3)
        total = sum(weights)
        widths = [max(700, round(TEXT_WIDTH_DXA * w / total)) for w in weights]
        widths[-1] += TEXT_WIDTH_DXA - sum(widths)
        border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        borders = "".join(border.format(side) for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
        xml = [
            f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_WIDTH_DXA}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr>'
            f'<w:tblGrid>{"".join(f"<w:gridCol w:w=\"{w}\"/>" for w in widths)}</w:tblGrid>'
        ]
        for r, row in enumerate(rows):
            head = r < header_rows
            props = "<w:cantSplit/>" + ("<w:tblHeader/>" if head else "")
            cells = []
            for c in range(columns):
                cell = row[c] if c < len(row) else []
                shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if head else ""
                runs = inline_runs(cell)
                if head:
                    runs = runs.replace("<w:rPr>", "<w:rPr><w:b/><w:bCs/>").replace("<w:r><w:t ", "<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t ")
                cells.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{widths[c]}" w:type="dxa"/>{shade}</w:tcPr>'
                    f'<w:p><w:pPr><w:pStyle w:val="TextoTabla"/></w:pPr>{runs}</w:p></w:tc>'
                )
            xml.append(f"<w:tr><w:trPr>{props}</w:trPr>{''.join(cells)}</w:tr>")
        xml.append("</w:tbl>")
        self.body.append("".join(xml))
        self.body.append('<w:p><w:pPr><w:pStyle w:val="Cuerpo"/><w:spacing w:after="60"/></w:pPr></w:p>')

    # -- Markdown tokens
    def render(self, text: str) -> None:
        tokens = MarkdownIt("commonmark").enable("table").parse(text)
        i, list_stack = 0, []
        while i < len(tokens):
            tok = tokens[i]
            kind = tok.type
            if kind == "heading_open":
                inline = tokens[i + 1]
                level = int(tok.tag[1])
                self.paragraph(f"Heading{min(level, 4)}", inline_runs(inline.children))
                i += 3
            elif kind == "paragraph_open":
                inline = tokens[i + 1]
                kids = inline.children
                images = [k for k in kids if k.type == "image"]
                if images and not plain_text(kids).strip():
                    for image in images:
                        self.picture(image.attrs["src"], image.content)
                elif list_stack:
                    ordered, num_id, level = list_stack[-1]
                    numpr = f'<w:numPr><w:ilvl w:val="{min(level, 2)}"/><w:numId w:val="{num_id}"/></w:numPr>'
                    self.paragraph("Lista", inline_runs(kids), numpr)
                else:
                    plain = plain_text(kids).lstrip()
                    caption = plain.startswith(("Figura ", "Tabla "))
                    keep = "<w:keepNext/>" if plain.startswith("Tabla ") else ""
                    self.paragraph("Leyenda" if caption else "Cuerpo", inline_runs(kids), keep)
                i += 3
            elif kind in ("bullet_list_open", "ordered_list_open"):
                ordered = kind == "ordered_list_open"
                if ordered:
                    num_id = self.next_num
                    self.next_num += 1
                    start = int(tok.attrs.get("start", 1)) if tok.attrs else 1
                    self.nums.append(
                        f'<w:num w:numId="{num_id}"><w:abstractNumId w:val="{NUM_ORDERED}"/>'
                        f'<w:lvlOverride w:ilvl="0"><w:startOverride w:val="{start}"/></w:lvlOverride></w:num>'
                    )
                else:
                    num_id = NUM_BULLET
                list_stack.append((ordered, num_id, len(list_stack)))
                i += 1
            elif kind in ("bullet_list_close", "ordered_list_close"):
                list_stack.pop()
                if not list_stack:
                    self.body.append('<w:p><w:pPr><w:pStyle w:val="Cuerpo"/><w:spacing w:after="0" w:line="120" w:lineRule="exact"/></w:pPr></w:p>')
                i += 1
            elif kind in ("fence", "code_block"):
                lines = tok.content.rstrip("\n").split("\n")
                for line in lines:
                    self.paragraph("Codigo", _run(line or " "))
                self.body.append('<w:p><w:pPr><w:pStyle w:val="Cuerpo"/><w:spacing w:after="60"/></w:pPr></w:p>')
                i += 1
            elif kind == "table_open":
                rows, header_rows, row = [], 0, None
                i += 1
                while tokens[i].type != "table_close":
                    t = tokens[i]
                    if t.type == "tr_open":
                        row = []
                    elif t.type in ("th_open", "td_open"):
                        row.append(tokens[i + 1].children)
                        i += 2
                    elif t.type == "tr_close":
                        rows.append(row)
                    elif t.type == "thead_close":
                        header_rows = len(rows)
                    i += 1
                self.table(rows, header_rows)
                i += 1
            elif kind == "blockquote_open":
                i += 1
                while tokens[i].type != "blockquote_close":
                    if tokens[i].type == "inline":
                        self.paragraph("Cita", inline_runs(tokens[i].children))
                    i += 1
                i += 1
            elif kind == "hr":
                self.body.append('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="BFBFBF"/></w:pBdr></w:pPr></w:p>')
                i += 1
            else:
                i += 1


# ---------------------------------------------------------------- template surgery


def textbox_anchor(box_id: int, y_offset: int, width: int, height: int, paragraphs: str) -> str:
    return (
        '<w:r><mc:AlternateContent><mc:Choice Requires="wps"><w:drawing>'
        '<wp:anchor allowOverlap="1" behindDoc="0" distB="0" distT="0" distL="114300" distR="114300" hidden="0" layoutInCell="1" locked="0" relativeHeight="0" simplePos="0">'
        '<wp:simplePos x="0" y="0"/><wp:positionH relativeFrom="column"><wp:posOffset>-114299</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="paragraph"><wp:posOffset>{y_offset}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{width}" cy="{height}"/><wp:effectExtent b="0" l="0" r="0" t="0"/>'
        '<wp:wrapSquare wrapText="bothSides" distB="0" distT="0" distL="114300" distR="114300"/>'
        f'<wp:docPr id="{box_id}" name="Cuadro {box_id}"/><a:graphic><a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        f'<wps:wsp><wps:cNvSpPr txBox="1"/><wps:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></wps:spPr>'
        f"<wps:txbx><w:txbxContent>{paragraphs}</w:txbxContent></wps:txbx>"
        '<wps:bodyPr anchorCtr="0" anchor="t" bIns="45700" lIns="91425" rIns="91425" tIns="45700" wrap="square"><a:noAutofit/></wps:bodyPr>'
        "</wps:wsp></a:graphicData></a:graphic></wp:anchor></w:drawing></mc:Choice></mc:AlternateContent></w:r>"
    )


def cover_paragraph() -> str:
    def line(text: str, bold: bool, size: int, color: str, after: int = 0) -> str:
        return (
            f'<w:p><w:pPr><w:spacing w:after="{after}" w:before="0" w:line="240"/><w:jc w:val="left"/></w:pPr><w:r><w:rPr>{FONT}'
            f'{"<w:b/>" if bold else ""}<w:color w:val="{color}"/><w:sz w:val="{size}"/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
        )

    title = textbox_anchor(1, 3600000, 4200000, 2400000, line(TITLE, True, 36, "595959"))
    subtitle = textbox_anchor(
        2, 6300000, 4200000, 2200000,
        "".join(line(t, bold, size, "595959" if bold else "767171", 80) for t, bold, size in SUBTITLE_LINES),
    )
    return f'<w:p><w:pPr><w:spacing w:after="0" w:before="0"/></w:pPr>{title}{subtitle}</w:p>'


def build() -> None:
    sections = sorted((INFORME / "secciones").glob("[0-1][0-9]_*.md"))
    sections += [INFORME / "referencias.md", INFORME / "secciones" / "A_reproducir.md"]
    missing = [p for p in sections if not p.exists()]
    if missing:
        sys.exit(f"missing: {missing}")

    builder = Builder(INFORME / "secciones")
    for path in sections:
        builder.render(path.read_text(encoding="utf-8"))

    zin = zipfile.ZipFile(TEMPLATE)
    files = {name: zin.read(name) for name in zin.namelist()}
    document = files["word/document.xml"].decode("utf-8")

    body_start = document.index("<w:body>") + len("<w:body>")
    # The cover paragraph holds text boxes with nested paragraphs, so locate the paragraph that closes section 1 by
    # its sample text instead of splitting the body.
    marker = document.index("DESPUES DE ESTA P")
    start = document.rfind("<w:p ", 0, marker)
    end = document.index("</w:p>", marker) + len("</w:p>")
    section_properties = re.search(r"<w:sectPr>.*?</w:sectPr>", document[start:end], re.S).group(0)
    rest = document[end:].replace("NOMBRE DE LA DEPENDENCIA", escape(BACK_COVER_NAME))
    new_body = (
        cover_paragraph()
        + "".join(builder.body)
        + f"<w:p><w:pPr>{section_properties}</w:pPr></w:p>"
    )
    files["word/document.xml"] = (document[:body_start] + new_body + rest).encode("utf-8")

    header = files["word/header1.xml"].decode("utf-8").replace("Nombre del documento", escape(HEADER_TEXT))
    files["word/header1.xml"] = header.replace('<w:textDirection w:val="btLr"/>', "").encode("utf-8")

    styles = files["word/styles.xml"].decode("utf-8")
    styles = styles.replace('<w:rFonts w:ascii="Calibri" w:cs="Calibri" w:eastAsia="Calibri" w:hAnsi="Calibri"/>', FONT, 1)
    for style_id, (size, color, before, after, page_break) in HEADINGS.items():
        pattern = re.compile(rf'(<w:style w:type="paragraph" w:styleId="{style_id}">.*?)(</w:style>)', re.S)
        replacement = (
            f'<w:style w:type="paragraph" w:styleId="{style_id}"><w:name w:val="heading {style_id[-1]}"/><w:basedOn w:val="Normal"/>'
            f'<w:next w:val="Cuerpo"/><w:qFormat/><w:pPr><w:keepNext w:val="1"/><w:keepLines w:val="1"/>'
            f'<w:pageBreakBefore w:val="{1 if page_break else 0}"/><w:spacing w:before="{before}" w:after="{after}" w:lineRule="auto"/>'
            f'<w:outlineLvl w:val="{int(style_id[-1]) - 1}"/></w:pPr><w:rPr>{FONT}<w:b w:val="1"/><w:bCs w:val="1"/>'
            f'<w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr></w:style>'
        )
        styles, count = pattern.subn(lambda m: replacement, styles, count=1)
        if count != 1:
            sys.exit(f"style {style_id} not found in the template")
    files["word/styles.xml"] = styles.replace("</w:styles>", STYLES + "</w:styles>").encode("utf-8")

    numbering = files["word/numbering.xml"].decode("utf-8")
    abstracts = _abstract_num(NUM_BULLET, True) + _abstract_num(NUM_ORDERED, False)
    nums = f'<w:num w:numId="{NUM_BULLET}"><w:abstractNumId w:val="{NUM_BULLET}"/></w:num>' + "".join(builder.nums)
    numbering = numbering.replace("</w:numbering>", "") if "</w:numbering>" in numbering else numbering
    numbering = re.sub(r"/>\s*$", ">", numbering, count=1) if numbering.rstrip().endswith("/>") else numbering
    files["word/numbering.xml"] = (numbering + abstracts + nums + "</w:numbering>").encode("utf-8")

    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    files["word/_rels/document.xml.rels"] = rels.replace("</Relationships>", "".join(builder.relationships) + "</Relationships>").encode("utf-8")
    for name, source in builder.media:
        files[name] = source.read_bytes()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("[Content_Types].xml", files.pop("[Content_Types].xml"))
        for name, data in files.items():
            zout.writestr(name, data)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(builder.media)} figures, {len(sections)} parts)")


if __name__ == "__main__":
    build()
