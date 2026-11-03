#!/usr/bin/env python3
"""Build review copies of annotation guidelines without replacing public artifacts.

Usage:
    python3 docs/annotation/curator/build_guidelines.py \
        docs/annotation/chemistry_annotation_guidelines.md \
        docs/annotation/legal_annotation_guidelines.md

Requires XeLaTeX and Carlito for PDFs, and pypdf for the page-count report.
DOCX generation uses only the Python standard library and retains the adjacent
existing DOCX's styles, header, footer, relationships, and section settings.
The supported Markdown subset is headings, paragraphs, bold/italic/code spans,
simple bullet lists, tables, and explicit PAGE BREAK comments.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W)
ET.register_namespace("r", R)
BREAK = "<!-- PAGE BREAK -->"
INLINE = re.compile(r"(\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*)|`[^`]+`)")


def tag(name: str) -> str:
    return f"{{{W}}}{name}"


def el(name: str, **attrs: object) -> ET.Element:
    return ET.Element(tag(name), {tag(k): str(v) for k, v in attrs.items()})


def inline_spans(text: str):
    """Yield literal text and its simple Markdown emphasis, without changing it."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            yield part[2:-2], "bold"
        elif part.startswith("*") and part.endswith("*"):
            yield part[1:-1], "italic"
        elif part.startswith("`") and part.endswith("`"):
            yield part[1:-1], "code"
        else:
            yield part, "plain"


def blocks(markdown: str):
    lines = markdown.splitlines()
    pos = 0
    while pos < len(lines):
        line = lines[pos].strip()
        if not line:
            pos += 1
            continue
        if line == BREAK:
            yield "break", None
            pos += 1
            continue
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if match:
            yield "heading", (len(match[1]), match[2])
            pos += 1
            continue
        if line.startswith("|"):
            rows = []
            while pos < len(lines) and lines[pos].strip().startswith("|"):
                cells = [x.strip() for x in lines[pos].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    if len(cells) < 2 or (rows and len(cells) != len(rows[0])):
                        raise ValueError("Markdown tables must have at least two consistent columns")
                    rows.append(cells)
                pos += 1
            yield "table", rows
            continue
        if line.startswith("- "):
            yield "bullet", line[2:]
            pos += 1
            continue
        paragraph = [line]
        pos += 1
        while pos < len(lines):
            next_line = lines[pos].strip()
            if (not next_line or next_line == BREAK or next_line.startswith("|")
                    or next_line.startswith("- ") or re.match(r"^#{1,3}\s", next_line)):
                break
            paragraph.append(next_line)
            pos += 1
        yield "paragraph", " ".join(paragraph)


def word_paragraph(text: str, style: str = "Normal", bold: bool = False):
    paragraph = el("p")
    properties = ET.SubElement(paragraph, tag("pPr"))
    properties.append(el("pStyle", val=style))
    for part, kind in inline_spans(text):
        run = ET.SubElement(paragraph, tag("r"))
        properties = ET.SubElement(run, tag("rPr"))
        if bold or kind == "bold":
            properties.append(el("b"))
        if kind == "italic":
            properties.append(el("i"))
        if kind == "code":
            properties.append(el("rFonts", ascii="Carlito", hAnsi="Carlito"))
        value = ET.SubElement(run, tag("t"), {f"{{{XML}}}space": "preserve"})
        value.text = part
    return paragraph


def word_table(rows):
    column_count = len(rows[0])
    if column_count == 2:
        widths = [2300, 7566]
    else:
        first_width = 3500
        other_width = (9866 - first_width) // (column_count - 1)
        widths = [first_width] + [other_width] * (column_count - 1)
        widths[-1] += 9866 - sum(widths)
    table = el("tbl")
    properties = ET.SubElement(table, tag("tblPr"))
    properties.append(el("tblW", w=9866, type="dxa"))
    properties.append(el("tblLayout", type="fixed"))
    borders = ET.SubElement(properties, tag("tblBorders"))
    for edge in ("bottom", "insideH"):
        borders.append(el(edge, val="single", sz=4, color="D8E0E6"))
    margins = ET.SubElement(properties, tag("tblCellMar"))
    for edge, width in (("top", 75), ("left", 95), ("bottom", 75), ("right", 95)):
        margins.append(el(edge, w=width, type="dxa"))
    grid = ET.SubElement(table, tag("tblGrid"))
    for width in widths:
        grid.append(el("gridCol", w=width))
    for index, row in enumerate(rows):
        tr = ET.SubElement(table, tag("tr"))
        tr_properties = ET.SubElement(tr, tag("trPr"))
        tr_properties.append(el("cantSplit"))
        if index == 0:
            tr_properties.append(el("tblHeader"))
        for column, (text, width) in enumerate(zip(row, widths)):
            cell = ET.SubElement(tr, tag("tc"))
            cell_properties = ET.SubElement(cell, tag("tcPr"))
            cell_properties.append(el("tcW", w=width, type="dxa"))
            if index == 0:
                cell_properties.append(el("shd", fill="EAF0F5"))
            paragraph = word_paragraph(text, "TableText", bold=index == 0)
            if column_count > 2 and column > 0:
                paragraph.find(tag("pPr")).append(el("jc", val="center"))
            cell.append(paragraph)
    return table


def build_docx(template: Path, output: Path, content):
    if template.resolve() == output.resolve():
        raise ValueError("Review output must not replace the public DOCX template")
    with ZipFile(template) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        body = document.find(tag("body"))
        section = body.find(tag("sectPr"))
        if section is None:
            raise ValueError(f"No section properties found in {template}")
        section = copy.deepcopy(section)
        body.clear()
        for kind, data in content:
            if kind == "break":
                paragraph = ET.SubElement(body, tag("p"))
                run = ET.SubElement(paragraph, tag("r"))
                run.append(el("br", type="page"))
            elif kind == "heading":
                level, text = data
                body.append(word_paragraph(text, {1: "Title", 2: "Heading1", 3: "Heading2"}[level]))
            elif kind == "table":
                body.append(word_table(data))
            elif kind == "bullet":
                body.append(word_paragraph("• " + data, "ListText"))
            else:
                body.append(word_paragraph(data))
        body.append(section)
        replacement = ET.tostring(document, encoding="utf-8", xml_declaration=True)
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as destination:
            for member in archive.infolist():
                destination.writestr(member, replacement if member.filename == "word/document.xml"
                                     else archive.read(member.filename))


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
    "%": r"\%", "&": r"\&", "$": r"\$", "#": r"\#", "_": r"\_",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}",
}


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(char, char) for char in text)


def latex_inline(text: str) -> str:
    result = []
    for part, kind in inline_spans(text):
        escaped = latex_escape(part)
        command = {"bold": "textbf", "italic": "textit", "code": "texttt"}.get(kind)
        result.append("\\" + command + "{" + escaped + "}" if command else escaped)
    return "".join(result)


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Carlito}
\usepackage[a4paper,left=18mm,right=18mm,top=17mm,bottom=17mm,headheight=12pt,headsep=5mm,footskip=8mm]{geometry}
\usepackage[table]{xcolor}
\usepackage{array,tabularx,fancyhdr,needspace,lastpage}
\definecolor{ink}{HTML}{193A50}
\definecolor{muted}{HTML}{627785}
\definecolor{rulegray}{HTML}{D8E0E6}
\definecolor{tablehead}{HTML}{EAF0F5}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\fontsize{8}{9}\selectfont\color{muted}QUESTION ANNOTATION GUIDELINES}
\fancyfoot[R]{\fontsize{8}{9}\selectfont\color{muted}DOMAIN guidelines\quad •\quad \thepage\ / \pageref{LastPage}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.14}
\setlength{\emergencystretch}{2em}
\clubpenalty=10000
\widowpenalty=10000
\raggedbottom
\begin{document}
\fontsize{11}{12.7}\selectfont
"""


def latex_document(content, domain: str) -> str:
    result = [PREAMBLE.replace("DOMAIN", latex_escape(domain))]
    for kind, data in content:
        if kind == "break":
            result.append(r"\newpage")
        elif kind == "heading":
            level, text = data
            size, leading = {1: (17, 20), 2: (13, 15), 3: (11.5, 13.5)}[level]
            result.append(r"\Needspace{3\baselineskip}")
            if level != 1:
                result.append(r"\vspace{4pt}")
            result.append(r"{\color{ink}\bfseries\fontsize{%s}{%s}\selectfont %s\par}" %
                          (size, leading, latex_inline(text)))
        elif kind == "table":
            column_count = len(data[0])
            column_spec = (
                r">{\raggedright\arraybackslash}p{39mm}>{\raggedright\arraybackslash}X"
                if column_count == 2 else
                r">{\raggedright\arraybackslash}p{60mm}" +
                r">{\centering\arraybackslash}X" * (column_count - 1)
            )
            result.extend([
                r"{\fontsize{10.5}{12}\selectfont",
                r"\arrayrulecolor{rulegray}",
                r"\begin{tabularx}{\textwidth}{" + column_spec + "}",
            ])
            for row_number, row in enumerate(data):
                if row_number == 0:
                    result.append(r"\rowcolor{tablehead}")
                cells = [latex_inline(cell) for cell in row]
                if row_number == 0:
                    cells = [r"\textbf{" + cell + "}" for cell in cells]
                result.append(" & ".join(cells) + r" \\ \hline")
            result.extend([r"\end{tabularx}", "}"])
        elif kind == "bullet":
            result.append(r"\hangindent=1em\hangafter=1 • " + latex_inline(data) + r"\par")
        else:
            result.append(latex_inline(data) + r"\par")
    result.append(r"\end{document}")
    return "\n".join(result) + "\n"


def build(markdown: Path, output_dir: Path, expected_pages: int):
    markdown = markdown.resolve()
    template = markdown.with_suffix(".docx")
    if not template.exists():
        raise FileNotFoundError(f"Adjacent DOCX style template is missing: {template}")
    content = list(blocks(markdown.read_text(encoding="utf-8")))
    title = next((data[1] for kind, data in content if kind == "heading"), markdown.stem)
    domain = next((name for name in ("Chemistry", "Legal") if name.lower() in markdown.stem.lower()), title)
    output_dir.mkdir(parents=True, exist_ok=True)
    docx = output_dir / template.name
    tex = output_dir / (markdown.stem + ".tex")
    pdf = output_dir / (markdown.stem + ".pdf")
    if pdf.resolve() == markdown.with_suffix(".pdf").resolve():
        raise ValueError("Build to a review directory; do not replace public PDFs before review")
    build_docx(template, docx, content)
    tex.write_text(latex_document(content, domain), encoding="utf-8")
    command = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
               "-output-directory", str(output_dir), str(tex)]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("XeLaTeX failed:\n" + result.stdout[-6000:] + result.stderr[-2000:])
    from pypdf import PdfReader
    page_count = len(PdfReader(pdf).pages)
    log = tex.with_suffix(".log").read_text(errors="replace")
    warnings = [line.strip() for line in log.splitlines()
                if "Overfull" in line or "Missing character:" in line]
    report = {
        "source": str(markdown), "pdf": str(pdf), "docx": str(docx),
        "pages": page_count, "expected_pages": expected_pages,
        "page_count_ok": page_count == expected_pages,
        "explicit_page_breaks": sum(kind == "break" for kind, _ in content),
        "layout_warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("markdown", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/clir_guideline_build"))
    parser.add_argument("--expected-pages", type=int, default=4)
    args = parser.parse_args()
    if not shutil.which("xelatex"):
        parser.error("xelatex is required")
    reports = [build(path, args.output_dir.resolve(), args.expected_pages) for path in args.markdown]
    return 0 if all(report["page_count_ok"] and not report["layout_warnings"] for report in reports) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
