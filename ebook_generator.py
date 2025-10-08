#!/usr/bin/env python3
"""
AI e‑Book Generator (≈400 pages)
--------------------------------

Given a single BOOK_IDEA string, this program plans, drafts, and compiles a full-length
non‑fiction or fiction e‑book (~100k–120k words, ~350–450 pages at 250–300 wpp).

Highlights
- Outline-first workflow: metadata → synopsis → parts → chapters → sections → scenes.
- Resumable + checkpointed generation (JSONL). Safe to stop/restart any time.
- Pluggable LLM backends (OpenAI, Anthropic, or Ollama/local) via a common interface.
- Style controls (tone, audience, reading level). Content policy guardrails.
- Structured prompts enforce cohesion and continuity across chapters.
- Emits Markdown sources, plus optional EPUB/PDF/DOCX if toolchain available.

Quick start
-----------
1) Install deps (choose your backend):
   # Core
   pip install pydantic==2.* rich==13.* tqdm==4.* tenacity==8.* python-dateutil==2.*
   
   # At least one backend (uncomment one):
   pip install openai==1.*
   # pip install anthropic==0.*
   # For local models via Ollama's HTTP API, no extra pip needed (uses 'requests').
   pip install requests==2.*

   # Optional outputs
   pip install ebooklib==0.* markdown==3.* python-docx==1.* reportlab==4.*
   # (Or ensure 'pandoc' is installed for high‑quality EPUB/PDF conversion.)

2) Export credentials for your chosen backend (one of):
   export OPENAI_API_KEY=...
   export ANTHROPIC_API_KEY=...
   # or run an Ollama server locally (http://localhost:11434) with a model pulled
   # e.g., `ollama pull llama3.1`.

3) Run:
   python ebook_generator.py \
     --idea "A microbiologist uses AI to decode the hidden ecology of dairy farm microbes" \
     --genre "popular science" \
     --audience "curious adults" \
     --reading-level "grade 10" \
     --style "engaging, clear, lightly humorous" \
     --min-words 105000 --max-words 125000 \
     --backend openai --model gpt-4.1 \
     --outdir ./book_out

   You can resume safely:
   python ebook_generator.py --resume --outdir ./book_out

Outputs
-------
- book_out/
  ├─ plan/               # JSON & MD planning artifacts
  ├─ chapters/           # Per‑chapter Markdown drafts
  ├─ book.md             # Single concatenated Markdown manuscript
  ├─ book.epub           # If ebooklib or pandoc available
  ├─ book.pdf            # If reportlab or pandoc available
  └─ checkpoints.jsonl   # Resumable generation log

Notes
-----
- 400 pages is a *lot* of text. This program generates in small, controlled chunks
  with continuity prompts. Expect long runtimes and API costs for cloud LLMs. For a
  cheaper test, try --min-words 20_000 first.
- Prompts include safety and factuality guidance but you are responsible for final edits.

"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from tqdm import tqdm

# Optional imports guarded at runtime
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None

try:
    import requests  # for Ollama or custom HTTP LLMs
except Exception:  # pragma: no cover
    requests = None

# Optional for outputs
try:
    from ebooklib import epub  # type: ignore
except Exception:
    epub = None

try:
    import markdown  # type: ignore
except Exception:
    markdown = None

try:
    import docx  # python-docx
    from docx import Document  # type: ignore
except Exception:
    Document = None

console = Console()

# -----------------------------
# LLM Backend Abstraction
# -----------------------------

class LLMError(Exception):
    pass

class LLMClient:
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1500, model: str = "") -> str:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def __init__(self, model: str):
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai")
        self.model = model
        # Using new OpenAI client style
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1500, model: str = "") -> str:
        m = model or self.model
        try:
            resp = self.client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(str(e))

class AnthropicClient(LLMClient):
    def __init__(self, model: str):
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. pip install anthropic")
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1500, model: str = "") -> str:
        m = model or self.model
        try:
            msg = self.client.messages.create(
                model=m,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            # anthropic SDK returns a list of content blocks
            parts = []
            for block in msg.content:
                if getattr(block, "type", "") == "text":
                    parts.append(block.text)
            return "\n".join(parts)
        except Exception as e:
            raise LLMError(str(e))

class OllamaClient(LLMClient):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        if requests is None:
            raise RuntimeError("requests not installed. pip install requests")
        self.model = model
        self.host = host.rstrip('/')

    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1500, model: str = "") -> str:
        m = model or self.model
        try:
            # Simple non‑stream call
            payload = {
                "model": m,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise LLMError(str(e))

# -----------------------------
# Data Models
# -----------------------------

class BookMeta(BaseModel):
    title: str
    subtitle: str
    author: str
    genre: str
    audience: str
    reading_level: str
    style: str
    promise: str = Field(description="Unique value proposition / hook.")

class SectionPlan(BaseModel):
    title: str
    summary: str

class ChapterPlan(BaseModel):
    number: int
    title: str
    purpose: str
    key_points: List[str]
    target_words: int
    sections: List[SectionPlan]

class BookPlan(BaseModel):
    synopsis: str
    parts: List[str]
    chapters: List[ChapterPlan]
    total_target_words: int

# -----------------------------
# Prompt Templates
# -----------------------------

SYSTEM_POLICY = (
    "You are a seasoned book author and editor. Write original, coherent, factual, and well‑structured prose. "
    "Avoid harmful, illegal, or hateful content. Do not fabricate real citations or misrepresent facts. If the user idea requires facts, "
    "present them responsibly and add plain‑language context. Maintain consistency in names, terminology, and narrative voice."
)

META_PROMPT = """
Given the book idea below, propose professional metadata and a strong market hook.
Return JSON with keys: title, subtitle, author, genre, audience, reading_level, style, promise.

Book idea: {idea}
Preferred genre: {genre}
Target audience: {audience}
Reading level: {reading_level}
Style notes: {style}
""".strip()

PLAN_PROMPT = """
You are designing a complete long‑form book based on the metadata and idea below.
Produce:
1) A 2–3 paragraph synopsis.
2) A list of up to 4 PART labels (optional for non‑fiction).
3) A detailed chapter plan with 24–36 chapters. Each chapter must include:
   - number (int), title, purpose (1–2 sentences), key_points (4–8 bullets),
   - target_words (3000–4000 words),
   - 4–8 sections with title + 1–2 sentence summary each.
Aim total target words between {min_words} and {max_words}.
Return JSON with keys: synopsis, parts, chapters[], total_target_words.

Metadata: {meta_json}
Book idea: {idea}
""".strip()

CHAPTER_DRAFT_PROMPT = """
Draft Chapter {ch_num}: "{ch_title}" for the book described below. Write in the established voice.
Use the chapter plan faithfully, covering each section in order with smooth transitions.
Target ~{target_words} words (±10%).
Include:
- A short italicized chapter opener (1–2 sentences) that teases the theme.
- Clear section headings using Markdown H3 (### Section Title).
- Figures/tables placeholders in Markdown when helpful (e.g., *[Figure: ...]*), but do not invent data.
- End with 3–5 reflective questions or key takeaways as a bulleted list.

Constraints:
- Preserve terminology continuity with previous chapters (names, terms, acronyms).
- Avoid repetition and filler; prefer concrete examples and explanations.
- For claims of fact, write cautiously and avoid specific statistics unless widely accepted or necessary.

Book metadata: {meta_json}
Book synopsis: {synopsis}
Chapter plan JSON: {chapter_json}

Previously drafted chapter titles:
{prev_titles}
""".strip()

# -----------------------------
# Orchestration
# -----------------------------

def estimate_chapter_count(total_words: int, per_chapter: int = 3500) -> int:
    return max(16, min(40, round(total_words / per_chapter)))


def sanitize_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\- _]+", "", name).strip().replace(" ", "_")
    return s[:120] if s else "chapter"


class Checkpointer:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def log(self, event: Dict[str, Any]):
        event["ts"] = datetime.utcnow().isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class BookBuilder:
    def __init__(self, llm: LLMClient, outdir: Path, idea: str, genre: str, audience: str,
                 reading_level: str, style: str, min_words: int, max_words: int, resume: bool = False,
                 temperature: float = 0.7, model: str = ""):
        self.llm = llm
        self.outdir = outdir
        self.plan_dir = outdir / "plan"
        self.chapters_dir = outdir / "chapters"
        self.idea = idea
        self.genre = genre
        self.audience = audience
        self.reading_level = reading_level
        self.style = style
        self.min_words = min_words
        self.max_words = max_words
        self.temperature = temperature
        self.model = model
        self.checkpoint = Checkpointer(outdir / "checkpoints.jsonl")
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.meta: Optional[BookMeta] = None
        self.plan: Optional[BookPlan] = None

        if resume:
            self._try_load_state()

    def _try_load_state(self):
        meta_path = self.plan_dir / "meta.json"
        plan_path = self.plan_dir / "plan.json"
        if meta_path.exists():
            self.meta = BookMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if plan_path.exists():
            self.plan = BookPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))

    def _save_json(self, path: Path, data: BaseModel | Dict[str, Any]):
        if isinstance(data, BaseModel):
            text = data.model_dump_json(indent=2, ensure_ascii=False)
        else:
            text = json.dumps(data, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")

    # ----- Steps -----
    def generate_meta(self) -> BookMeta:
        if self.meta:
            return self.meta
        prompt = META_PROMPT.format(
            idea=self.idea, genre=self.genre, audience=self.audience,
            reading_level=self.reading_level, style=self.style
        )
        text = self.llm.complete(SYSTEM_POLICY, prompt, temperature=self.temperature, max_tokens=1200, model=self.model)
        # Try parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # heuristic extraction (very basic)
            data = {
                "title": "Untitled Book",
                "subtitle": "A Project Generated by AI",
                "author": "Anonymous",
                "genre": self.genre,
                "audience": self.audience,
                "reading_level": self.reading_level,
                "style": self.style,
                "promise": "A fresh perspective created from a single idea.",
            }
        self.meta = BookMeta(**data)
        self._save_json(self.plan_dir / "meta.json", self.meta)
        self.checkpoint.log({"stage": "meta", "meta": self.meta.model_dump()})
        return self.meta

    def generate_plan(self) -> BookPlan:
        if self.plan:
            return self.plan
        meta = self.generate_meta()
        prompt = PLAN_PROMPT.format(
            meta_json=meta.model_dump_json(indent=2, ensure_ascii=False),
            idea=self.idea,
            min_words=self.min_words,
            max_words=self.max_words,
        )
        text = self.llm.complete(SYSTEM_POLICY, prompt, temperature=0.6, max_tokens=4000, model=self.model)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse plan JSON from the model. Inspect plan/ folder.")
        plan = BookPlan(**data)
        # Persist
        self.plan = plan
        self._save_json(self.plan_dir / "plan.json", plan)
        (self.plan_dir / "synopsis.md").write_text(plan.synopsis, encoding="utf-8")
        self.checkpoint.log({"stage": "plan", "total_target_words": plan.total_target_words})
        return plan

    def draft_chapter(self, ch: ChapterPlan, prev_titles: List[str]) -> str:
        meta = self.generate_meta()
        plan = self.generate_plan()
        prompt = CHAPTER_DRAFT_PROMPT.format(
            ch_num=ch.number,
            ch_title=ch.title,
            target_words=ch.target_words,
            meta_json=meta.model_dump_json(indent=2, ensure_ascii=False),
            synopsis=plan.synopsis,
            chapter_json=ch.model_dump_json(indent=2, ensure_ascii=False),
            prev_titles="\n".join(f"- {t}" for t in prev_titles) if prev_titles else "(none)",
        )
        text = self.llm.complete(SYSTEM_POLICY, prompt, temperature=self.temperature, max_tokens=5000, model=self.model)
        return text

    def run(self):
        meta = self.generate_meta()
        plan = self.generate_plan()

        chapter_files: List[Path] = []
        prev_titles: List[str] = []

        for ch in tqdm(plan.chapters, desc="Drafting chapters"):
            ch_name = f"{ch.number:02d}_{sanitize_filename(ch.title)}.md"
            ch_path = self.chapters_dir / ch_name
            if ch_path.exists() and ch_path.stat().st_size > 200:
                prev_titles.append(ch.title)
                chapter_files.append(ch_path)
                continue

            text = self.draft_chapter(ch, prev_titles)
            ch_doc = f"# {meta.title}: {ch.title}\n\n{text.strip()}\n"
            ch_path.write_text(ch_doc, encoding="utf-8")
            self.checkpoint.log({"stage": "chapter", "chapter": ch.number, "title": ch.title, "path": str(ch_path)})
            prev_titles.append(ch.title)
            chapter_files.append(ch_path)

        # Assemble book
        book_md = self._assemble_book_md(meta, plan, chapter_files)
        (self.outdir / "book.md").write_text(book_md, encoding="utf-8")
        self.checkpoint.log({"stage": "assemble", "path": str(self.outdir / "book.md")})

        # Optional exports
        try:
            self._export_epub(meta, book_md)
        except Exception as e:
            console.print(f"[yellow]EPUB export skipped/failed: {e}")
        try:
            self._export_docx(book_md)
        except Exception as e:
            console.print(f"[yellow]DOCX export skipped/failed: {e}")
        try:
            self._export_pdf(book_md)
        except Exception as e:
            console.print(f"[yellow]PDF export skipped/failed: {e}")

        console.print("\n[bold green]Done![/] Manuscript ready at:", self.outdir / "book.md")

    def _assemble_book_md(self, meta: BookMeta, plan: BookPlan, chapter_files: List[Path]) -> str:
        front = [
            f"# {meta.title}",
            f"## {meta.subtitle}",
            f"**Author:** {meta.author}",
            "\n---\n",
            "## About this book",
            plan.synopsis.strip(),
            "\n---\n",
            "## Table of Contents",
        ]
        for ch in plan.chapters:
            front.append(f"- Chapter {ch.number}: {ch.title}")
        front.append("\n---\n")

        parts = [p.read_text(encoding="utf-8") for p in chapter_files]
        return "\n\n".join(front + parts) + "\n"

    # ----- Exports -----
    def _export_epub(self, meta: BookMeta, book_md: str):
        if epub is None:
            raise RuntimeError("ebooklib not installed")
        bk = epub.EpubBook()
        bk.set_identifier("ai-ebook-" + re.sub(r"\W+", "-", meta.title.lower()))
        bk.set_title(meta.title)
        bk.set_language("en")
        bk.add_author(meta.author)

        # Split markdown into chapters by H1 markers (simple heuristic)
        chunks = re.split(r"\n(?=# )", book_md)
        spine = []
        toc = []
        for i, chunk in enumerate(chunks, 1):
            html = markdown.markdown(chunk) if markdown else f"<pre>{chunk}</pre>"
            c = epub.EpubHtml(title=f"Section {i}", file_name=f"sec_{i}.xhtml", content=html)
            bk.add_item(c)
            spine.append(c)
            toc.append(c)

        # required items
        bk.toc = tuple(toc)
        bk.spine = ['nav'] + spine
        bk.add_item(epub.EpubNcx())
        bk.add_item(epub.EpubNav())

        out = self.outdir / "book.epub"
        epub.write_epub(str(out), bk)
        console.print(f"[green]EPUB written:[/] {out}")

    def _export_docx(self, book_md: str):
        if Document is None:
            raise RuntimeError("python-docx not installed")
        from markdown import markdown as md_to_html  # requires markdown package
        from bs4 import BeautifulSoup  # optional but nice to have
        try:
            from bs4 import BeautifulSoup
        except Exception:
            raise RuntimeError("BeautifulSoup4 required for docx export: pip install beautifulsoup4")

        html = md_to_html(book_md)
        soup = BeautifulSoup(html, "html.parser")
        doc = Document()
        for node in soup.children:
            txt = node.get_text(strip=False)
            if node.name == "h1":
                doc.add_heading(txt, level=0)
            elif node.name == "h2":
                doc.add_heading(txt, level=1)
            elif node.name == "h3":
                doc.add_heading(txt, level=2)
            elif node.name == "p":
                doc.add_paragraph(txt)
            elif node.name == "ul":
                for li in node.find_all("li", recursive=False):
                    doc.add_paragraph(li.get_text(), style="List Bullet")
            else:
                doc.add_paragraph(txt)
        out = self.outdir / "book.docx"
        doc.save(str(out))
        console.print(f"[green]DOCX written:[/] {out}")

    def _export_pdf(self, book_md: str):
        # Lightweight fallback PDF using reportlab (no images / basic formatting)
        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.pdfgen import canvas as pdf_canvas
            from reportlab.lib.units import inch
        except Exception:
            raise RuntimeError("reportlab not installed; for better PDFs, install pandoc")

        out_path = self.outdir / "book.pdf"
        c = pdf_canvas.Canvas(str(out_path), pagesize=LETTER)
        width, height = LETTER
        margin = 0.75 * inch
        max_width = width - 2 * margin
        y = height - margin
        for line in book_md.splitlines():
            if not line.strip():
                y -= 14
                if y < margin:
                    c.showPage(); y = height - margin
                continue
            # naive wrap
            words = line.split(" ")
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if len(test) > 95:
                    c.drawString(margin, y, cur)
                    y -= 12
                    if y < margin:
                        c.showPage(); y = height - margin
                    cur = w
                else:
                    cur = test
            if cur:
                c.drawString(margin, y, cur)
                y -= 12
                if y < margin:
                    c.showPage(); y = height - margin
        c.showPage(); c.save()
        console.print(f"[green]PDF written:[/] {out_path}
")

# -----------------------------
# CLI
# -----------------------------

def build_llm(backend: str, model: str) -> LLMClient:
    backend = backend.lower()
    if backend == "openai":
        return OpenAIClient(model=model)
    if backend == "anthropic":
        return AnthropicClient(model=model)
    if backend in {"ollama", "local"}:
        return OllamaClient(model=model)
    raise ValueError(f"Unsupported backend: {backend}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a full e‑book from a single idea using an LLM.")
    p.add_argument("--idea", type=str, required=False, default="An untitled book idea",
                   help="Single‑sentence core idea for the book.")
    p.add_argument("--genre", type=str, default="non‑fiction")
    p.add_argument("--audience", type=str, default="general audience")
    p.add_argument("--reading-level", type=str, default="grade 10")
    p.add_argument("--style", type=str, default="clear, engaging, practical")
    p.add_argument("--min-words", type=int, default=105_000)
    p.add_argument("--max-words", type=int, default=125_000)
    p.add_argument("--backend", type=str, choices=["openai", "anthropic", "ollama", "local"], default="openai")
    p.add_argument("--model", type=str, default="gpt-4.1")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--outdir", type=Path, default=Path("./book_out"))
    p.add_argument("--resume", action="store_true", help="Resume from existing plan/chapters if present.")
    p.add_argument("--no-exports", action="store_true", help="Skip EPUB/DOCX/PDF exports; keep Markdown only.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    llm = build_llm(args.backend, args.model)

    builder = BookBuilder(
        llm=llm,
        outdir=args.outdir,
        idea=args.idea,
        genre=args.genre,
        audience=args.audience,
        reading_level=args.reading_level,
        style=args.style,
        min_words=args.min_words,
        max_words=args.max_words,
        resume=args.resume,
        temperature=args.temperature,
        model=args.model,
    )

    builder.run()

if __name__ == "__main__":
    main()
