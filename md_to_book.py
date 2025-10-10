#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
md_to_book.py — Convert Markdown to EPUB / DOCX / PDF

Priority:
1) Pandoc (best), 2) WeasyPrint (HTML->PDF), 3) ReportLab (basic)

Install (choose what you need):
  python -m pip install "pypandoc-binary==1.*"
  python -m pip install "weasyprint==61.*" "tinycss2==1.*" "cssselect2==0.*"
  python -m pip install "markdown==3.*" "beautifulsoup4==4.*"
  python -m pip install "reportlab==4.*"
"""
import argparse, os, subprocess, sys
from pathlib import Path

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}")
    return p.stdout

def to_epub(md_path: Path, outdir: Path, title: str, author: str):
    out = outdir / "book.epub"
    run(["pandoc", str(md_path), "-o", str(out), "--metadata", f"title={title}", "--metadata", f"author={author}"])
    return out

def to_docx(md_path: Path, outdir: Path):
    out = outdir / "book.docx"
    run(["pandoc", str(md_path), "-o", str(out)])
    return out

def to_pdf_pandoc(md_path: Path, outdir: Path, title: str):
    out = outdir / "book.pdf"
    run(["pandoc", str(md_path), "-o", str(out), "-V", f"title={title}"])
    return out

def to_pdf_weasy(md_path: Path, outdir: Path):
    from markdown import markdown
    from bs4 import BeautifulSoup
    from weasyprint import HTML
    html = markdown(md_path.read_text(encoding="utf-8"))
    soup = BeautifulSoup(html, "html.parser")
    out = outdir / "book.pdf"
    HTML(string=str(soup)).write_pdf(str(out))
    return out

def to_pdf_reportlab(md_path: Path, outdir: Path):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.units import inch
    out = outdir / "book.pdf"
    c = pdf_canvas.Canvas(str(out), pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    y = height - margin
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            y -= 14
            if y < margin: c.showPage(); y = height - margin
            continue
        words = line.split(" ")
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if len(test) > 95:
                c.drawString(margin, y, cur)
                y -= 12
                if y < margin: c.showPage(); y = height - margin
                cur = w
            else:
                cur = test
        if cur:
            c.drawString(margin, y, cur)
            y -= 12
            if y < margin: c.showPage(); y = height - margin
    c.showPage(); c.save()
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert Markdown to EPUB/DOCX/PDF")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./dist"))
    ap.add_argument("--title", type=str, default="Untitled Book")
    ap.add_argument("--author", type=str, default="Anonymous")
    ap.add_argument("--epub", action="store_true")
    ap.add_argument("--docx", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--all", action="store_true", help="Produce EPUB, DOCX, and PDF (Pandoc).")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    md_path = args.input

    produced = []
    if args.all or args.epub:
        try:
            produced.append(to_epub(md_path, args.outdir, args.title, args.author))
        except Exception as e:
            print("[warn] EPUB via Pandoc failed:", e)
    if args.all or args.docx:
        try:
            produced.append(to_docx(md_path, args.outdir))
        except Exception as e:
            print("[warn] DOCX via Pandoc failed:", e)
    if args.all or args.pdf:
        # try pandoc, then weasyprint, then reportlab
        try:
            produced.append(to_pdf_pandoc(md_path, args.outdir, args.title))
        except Exception as e:
            print("[warn] PDF via Pandoc failed:", e)
            try:
                produced.append(to_pdf_weasy(md_path, args.outdir))
            except Exception as e2:
                print("[warn] PDF via WeasyPrint failed:", e2)
                try:
                    produced.append(to_pdf_reportlab(md_path, args.outdir))
                except Exception as e3:
                    print("[warn] PDF via ReportLab failed:", e3)

    if not produced:
        print("No files produced. Consider installing Pandoc or additional PDF backends.")
    else:
        for p in produced:
            print("Wrote:", p)

if __name__ == "__main__":
    main()
