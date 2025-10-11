#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
md_to_book_legacy_like.py (with auto ToC rebuild)
=================================================
Stable Markdown → EPUB/DOCX/PDF converter that also **rebuilds the Table of Contents**
from your current H1 headings **by default**, so edits you make to the manuscript are
kept in sync automatically.

What’s new
----------
• Auto ToC rebuild (on by default). Use --no-rebuild-toc to disable.
• ToC bullets are regenerated from H1 headings found after the first H1.
  - If a heading looks like "# Book Title: Chapter Title", the ToC shows "Chapter N: Chapter Title".
  - Otherwise it shows "Chapter N: <full H1 text>".
• Everything else stays the same: robust, dependency-light exporters.

Usage
-----
python md_to_book_legacy_like.py \
  --input ./book_out/book.md \
  --outdir ./book_out \
  --title "Your Book Title" \
  --author "Your Name" \
  --all

Options
-------
--strip-citations    Remove [@cite_key] inline citations.
--strip-figures      Remove Markdown image lines and figure placeholders.
--no-rebuild-toc     Do NOT modify the Table of Contents block (default: rebuild).

Install
-------
pip install "markdown==3.*" "beautifulsoup4==4.*" "ebooklib==0.*" "python-docx==1.*" "reportlab==4.*"
"""

from __future__ import annotations
import argparse, re
from pathlib import Path

# Optional deps
try:
    import markdown  # type: ignore
except Exception:
    markdown = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

try:
    from ebooklib import epub  # type: ignore
except Exception:
    epub = None

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None

# ReportLab for robust PDF
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Preformatted
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
except Exception:
    SimpleDocTemplate = None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def clean_markdown(md: str, strip_citations: bool, strip_figures: bool) -> str:
    # Remove hyphenation line-breaks like "Introduc-\ntion" → "Introduction"
    md = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", md)

    # Normalize em-dash from spaced hyphens
    md = re.sub(r"\s+-\s+(?=\w)", " — ", md)

    if strip_citations:
        # remove [@key] or [@a; @b] blocks
        md = re.sub(r"\[@[^\]]+\]", "", md)

    if strip_figures:
        # drop pure image lines and simple [Figure: ...] placeholders
        md = re.sub(r'^\s*!\[[^\]]*\]\([^)]+\)\s*$', '', md, flags=re.M)
        md = re.sub(r'\*\[Figure:[^\]]+\]\*', '', md)

    # Tidy excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def parse_chapter_headings(md: str):
    """Return a list of H1 texts (excluding the very first H1, assumed book title)."""
    headings = re.findall(r"(?m)^#\s+(.+)$", md)
    if not headings:
        return []
    return headings[1:] if len(headings) > 1 else []


def build_toc_bullets_from_h1(md: str):
    """Create a list of '- Chapter N: Title' from H1 headings after the first one.
       If an H1 contains a ':', use the part after the first colon as the chapter title.
    """
    chapters = parse_chapter_headings(md)
    bullets = []
    for i, full in enumerate(chapters, 1):
        # If looks like "Book Title: Chapter Title", prefer the RHS
        title = full.split(":", 1)[1].strip() if ":" in full else full.strip()
        bullets.append(f"- Chapter {i}: {title}")
    return "\n".join(bullets) if bullets else "- (no chapters found)"


def rebuild_toc_block(md: str) -> str:
    """Replace the '## Table of Contents' section with regenerated bullets.
       We look for:
         '## Table of Contents' line,
         then replace until the next '---' horizontal rule OR next '## ' header.
    """
    toc_header = re.search(r"(?m)^##\s+Table of Contents\s*$", md)
    if not toc_header:
        # Insert a ToC after an About section if present, else after the first H1 block.
        bullets = build_toc_bullets_from_h1(md)
        about = re.search(r"(?m)^##\s+About this book\s*$", md)
        if about:
            insert_pos = about.end()
            # Find the end of the about section (up to next HR or header)
            tail = md[insert_pos:]
            m_end = re.search(r"(?m)^(?:---|\#\#\s+)", tail)
            end_pos = insert_pos + (m_end.start() if m_end else 0)
            if end_pos == insert_pos:
                end_pos = insert_pos
            toc_block = "\n\n## Table of Contents\n" + bullets + "\n\n"
            return md[:insert_pos] + md[insert_pos:end_pos] + toc_block + md[end_pos:]
        else:
            # Append at top after first H1
            h1 = re.search(r"(?m)^#\s+.+$", md)
            if not h1:
                return md  # nothing to do
            pos = h1.end()
            toc_block = "\n\n## Table of Contents\n" + bullets + "\n\n"
            return md[:pos] + toc_block + md[pos:]

    # Found a ToC header; replace its block
    start = toc_header.end()
    tail = md[start:]
    m_end = re.search(r"(?m)^(?:---|\#\#\s+)", tail)
    end = start + (m_end.start() if m_end else 0)

    bullets = build_toc_bullets_from_h1(md)
    new_block = "\n" + bullets + "\n"
    return md[:start] + new_block + md[end:]


def split_by_h1(md: str):
    """Return list of (title, chunk_md) split on H1. Keep first block as 'Front Matter' if no leading H1."""
    parts = re.split(r"(?m)^(# .+)\n", md)
    out = []
    if not parts or len(parts) == 1:
        return [("Manuscript", md)]
    # parts like: ["<before>", "# Title1\n", "<text1>", "# Title2\n", "<text2>", ...]
    preface = parts[0].strip()
    i = 1
    if preface:
        out.append(("Front Matter", preface))
    while i < len(parts):
        h = parts[i].strip()
        body = parts[i+1] if i+1 < len(parts) else ""
        title = h.lstrip("# ").strip()
        out.append((title, body))
        i += 2
    return out


def to_html(md: str) -> str:
    if markdown is None:
        # very light fallback: wrap pre tags
        safe = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{safe}</pre>"
    return markdown.markdown(md, extensions=["fenced_code", "tables"]).strip()


def make_epub(md: str, outdir: Path, title: str, author: str) -> Path:
    if epub is None:
        return outdir / "book.epub"  # skip silently
    bk = epub.EpubBook()
    bk.set_identifier("md-book-" + re.sub(r"\W+", "-", (title or 'book').lower()))
    bk.set_title(title or "Book")
    bk.set_language("en")
    if author:
        bk.add_author(author)

    chapters = []
    for idx, (sec_title, chunk) in enumerate(split_by_h1(md), 1):
        html = to_html(f"# {sec_title}\n\n{chunk}")
        item = epub.EpubHtml(title=sec_title, file_name=f"sec_{idx:02d}.xhtml", content=html)
        bk.add_item(item)
        chapters.append(item)

    bk.toc = tuple(chapters)
    bk.spine = ["nav"] + chapters
    bk.add_item(epub.EpubNcx())
    bk.add_item(epub.EpubNav())

    out = outdir / "book.epub"
    epub.write_epub(str(out), bk)
    return out


def make_docx(md: str, outdir: Path, title: str, author: str) -> Path:
    if Document is None or BeautifulSoup is None or markdown is None:
        return outdir / "book.docx"  # skip silently
    html = to_html(md)
    soup = BeautifulSoup(html, "html.parser")
    doc = Document()
    if title:
        doc.add_heading(title, level=0)
    if author:
        doc.add_paragraph(f"Author: {author}")
    doc.add_paragraph("")

    for node in soup.children:
        if getattr(node, "name", None) is None:
            continue
        txt = node.get_text(strip=False)
        if node.name == "h1":
            doc.add_heading(txt, level=1)
        elif node.name == "h2":
            doc.add_heading(txt, level=2)
        elif node.name == "h3":
            doc.add_heading(txt, level=3)
        elif node.name == "p":
            doc.add_paragraph(txt)
        elif node.name == "ul":
            for li in node.find_all("li", recursive=False):
                doc.add_paragraph(li.get_text(), style="List Bullet")
        elif node.name == "ol":
            for li in node.find_all("li", recursive=False):
                doc.add_paragraph(li.get_text(), style="List Number")
        else:
            if txt.strip():
                doc.add_paragraph(txt)

    out = outdir / "book.docx"
    doc.save(str(out))
    return out


def make_pdf(md: str, outdir: Path, title: str, author: str) -> Path:
    out = outdir / "book.pdf"
    if SimpleDocTemplate is None:
        return out  # silently skip if reportlab missing

    doc = SimpleDocTemplate(str(out), pagesize=LETTER,
                            leftMargin=0.85*inch, rightMargin=0.85*inch,
                            topMargin=0.85*inch, bottomMargin=0.85*inch)
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle('H1', parent=styles['Heading1'], alignment=TA_LEFT, spaceAfter=8)
    H2 = ParagraphStyle('H2', parent=styles['Heading2'], alignment=TA_LEFT, spaceAfter=6)
    H3 = ParagraphStyle('H3', parent=styles['Heading3'], alignment=TA_LEFT, spaceAfter=4)
    P  = ParagraphStyle('P',  parent=styles['BodyText'],  alignment=TA_LEFT, leading=14, spaceAfter=6)
    I  = ParagraphStyle('I',  parent=P, italic=True)

    story = []
    if title:
        story.append(Paragraph(title, H1))
    if author:
        story.append(Paragraph(f"Author: {author}", P))
    story.append(Spacer(1, 12))

    # Convert Markdown to HTML and walk
    html = to_html(md)
    if BeautifulSoup is None:
        story.append(Preformatted(md, P))
        doc.build(story)
        return out

    soup = BeautifulSoup(html, "html.parser")
    for node in soup.children:
        name = getattr(node, "name", None)
        if not name:
            continue
        txt = node.get_text(strip=False).replace("\n", "<br/>")

        if name == "h1":
            story.append(Paragraph(txt, H1))
        elif name == "h2":
            story.append(Paragraph(txt, H2))
        elif name == "h3":
            story.append(Paragraph(txt, H3))
        elif name == "p":
            story.append(Paragraph(txt, I if node.find('em') else P))
        elif name in ("ul", "ol"):
            for li in node.find_all("li", recursive=False):
                story.append(Paragraph("• " + li.get_text(), P))
        else:
            story.append(Paragraph(txt, P))

        story.append(Spacer(1, 6))

    doc.build(story)
    return out


def maybe_rebuild_toc(md: str, enabled: bool) -> str:
    if not enabled:
        return md
    try:
        return rebuild_toc_block(md)
    except Exception:
        # If anything goes sideways, return original md untouched.
        return md


def main():
    ap = argparse.ArgumentParser(description="Convert Markdown to EPUB/DOCX/PDF and auto-rebuild ToC from headings.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--author", type=str, default="")
    ap.add_argument("--strip-citations", action="store_true")
    ap.add_argument("--strip-figures", action="store_true")
    ap.add_argument("--no-rebuild-toc", action="store_true", help="Do not rebuild the Table of Contents block.")
    ap.add_argument("--all", action="store_true", help="Write EPUB, DOCX, and PDF")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    md = read_text(args.input)

    # 1) Clean / normalize
    md = clean_markdown(md, strip_citations=args.strip_citations, strip_figures=args.strip_figures)

    # 2) Rebuild ToC (default on)
    md = maybe_rebuild_toc(md, enabled=not args.no_rebuild_toc)

    # Always write back a cleaned, ToC-synced MD
    cleaned_path = args.outdir / "book.cleaned.md"
    write_text(cleaned_path, md)

    # 3) Outputs
    epub_path = make_epub(md, args.outdir, args.title or "Book", args.author or "Anonymous")
    docx_path = make_docx(md, args.outdir, args.title, args.author)
    pdf_path  = make_pdf(md, args.outdir, args.title, args.author)

    print(f"Wrote: {epub_path}")
    print(f"Wrote: {docx_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
