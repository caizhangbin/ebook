#!/usr/bin/env python3
"""
AI e-Book Generator (Self-Citing Edition)
----------------------------------------
- Outline-first planning → chapter drafting → assembly to Markdown (+ optional EPUB/DOCX/PDF)
- **Self-cite mode**: per-chapter literature search (Crossref), strict in-text citations [@key],
  auto-compiled References chapter using only fetched/known keys (prevents hallucinated refs).
- Works with OpenAI, Anthropic, or local Ollama models.

Quick start (OpenAI example):
    python -m pip install 'pydantic==2.*' 'rich==13.*' 'tqdm==4.*' 'tenacity==8.*' 'python-dateutil==2.*' 'requests==2.*'
    python -m pip install 'openai==1.*'  # or anthropic==0.*; Ollama needs no extra package
    # optional outputs
    python -m pip install 'ebooklib==0.*' 'markdown==3.*' 'python-docx==1.*' 'reportlab==4.*' 'beautifulsoup4==4.*'
    # optional: user-provided BibTeX
    python -m pip install 'bibtexparser==1.*'

    export OPENAI_API_KEY="sk-..."   # if using OpenAI
    python ebook_generator.py \
      --idea "A microbiologist uses AI to decode the hidden ecology of dairy farm microbes" \
      --genre "textbook" \
      --audience "upper-undergrad and graduate students" \
      --reading-level "university" \
      --style "scholarly, rigorous, precise" \
      --min-words 50000 --max-words 60000 \
      --backend openai --model gpt-5-thinking \
      --auto-cite \
      --outdir ./book_out
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
from rich.console import Console
from tqdm import tqdm

# ----- Optional imports guarded at runtime -----
try:
    import openai  # type: ignore
except Exception:
    openai = None

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

try:
    import requests  # HTTP (Ollama + Crossref queries)
except Exception:
    requests = None

try:
    from ebooklib import epub  # type: ignore
except Exception:
    epub = None

try:
    import markdown  # type: ignore
except Exception:
    markdown = None

try:
    import docx  # type: ignore
    from docx import Document  # type: ignore
except Exception:
    Document = None

try:
    import bibtexparser  # type: ignore
except Exception:
    bibtexparser = None

console = Console()

# =========================
# LLM Backends
# =========================
class LLMError(Exception):
    pass

class LLMClient:
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1500, model: str = "") -> str:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def __init__(self, model: str):
        if openai is None:
            raise RuntimeError("openai package not installed. Run: pip install openai==1.*")
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7,
                 max_tokens: int = 1500, model: str = "") -> str:
        m = model or self.model
        base = {
            "model": m,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }

        def chat_call(extra):
            return self.client.chat.completions.create(**{**base, **extra})

        # Try legacy-style params first
        try:
            resp = chat_call({"max_tokens": max_tokens, "temperature": temperature})
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)

            # Retry if model requires max_completion_tokens instead of max_tokens
            if "max_completion_tokens" in msg:
                try:
                    resp = chat_call({"extra_body": {"max_completion_tokens": max_tokens},
                                      "temperature": temperature})
                    return resp.choices[0].message.content or ""
                except Exception as e2:
                    msg = str(e2)

            # Retry if model disallows temperature (only default supported)
            if "temperature" in msg and "unsupported" in msg.lower():
                try:
                    # omit temperature entirely
                    resp = chat_call({"max_tokens": max_tokens})
                    return resp.choices[0].message.content or ""
                except Exception as e3:
                    msg = str(e3)

            # Final fallback: Responses API (max_output_tokens)
            try:
                resp2 = self.client.responses.create(
                    model=m,
                    input=[{"role": "system", "content": system},
                           {"role": "user", "content": prompt}],
                    max_output_tokens=max_tokens,
                    # omit temperature to satisfy strict models
                )
                return getattr(resp2, "output_text", None) or ""
            except Exception:
                pass

            raise LLMError(msg)


    def __init__(self, model: str):
        if openai is None:
            raise RuntimeError("openai package not installed. Run: pip install openai==1.*")
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1500, model: str = "") -> str:
        m = model or self.model
        base = {
            "model": m,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        try:
            # First try legacy-style param (works on many chat models)
            resp = self.client.chat.completions.create(**base, max_tokens=max_tokens)
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)
            # Some newer models require max_completion_tokens instead of max_tokens
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                resp = self.client.chat.completions.create(
                    **base,
                    extra_body={"max_completion_tokens": max_tokens},
                )
                return resp.choices[0].message.content or ""
            # (Optional) final fallback: Responses API with max_output_tokens
            try:
                resp2 = self.client.responses.create(
                    model=m,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                # openai==1.* returns a convenience helper:
                return getattr(resp2, "output_text", None) or ""
            except Exception:
                pass
            raise LLMError(msg)


class AnthropicClient(LLMClient):
    def __init__(self, model: str):
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic==0.*")
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
            raise RuntimeError("requests not installed. Run: pip install requests==2.*")
        self.model = model
        self.host = host.rstrip("/")

    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1500, model: str = "") -> str:
        m = model or self.model
        try:
            payload = {
                "model": m,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise LLMError(str(e))

# =========================
# Data Models
# =========================
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

# =========================
# Prompts
# =========================
SYSTEM_POLICY = (
    "You are a seasoned book author and editor. Write original, coherent, factual prose. "
    "Avoid harmful or hateful content. Do not fabricate real citations. Maintain consistency."
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
Design a complete long-form book based on the metadata and idea below.
Produce:
1) A 2–3 paragraph synopsis.
2) A list of up to 4 PART labels (optional for non-fiction).
3) A detailed chapter plan with 16–22 chapters. Each chapter must include:
   - number (int), title, purpose (1–2 sentences), key_points (4–8 bullets),
   - target_words (2500–3000 words),
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

Citations (when writing a textbook):
- Cite only sources that are present in the provided bibliography list below, using the Markdown form `[@key]`.
- If no appropriate source exists in the bibliography, write without a citation rather than fabricating one.

Book metadata: {meta_json}
Book synopsis: {synopsis}
Chapter plan JSON: {chapter_json}

Available bibliography keys (title → key):
{bib_keys}

Previously drafted chapter titles:
{prev_titles}
""".strip()

# =========================
# Auto-citation (Crossref)
# =========================
CROSSREF_API = "https://api.crossref.org/works"

@dataclass
class Ref:
    key: str
    title: str
    authors: str
    year: str
    venue: str
    doi: str = ""
    url: str = ""

    def to_bibtex(self) -> str:
        fields = {"title": self.title, "author": self.authors, "year": self.year}
        if self.venue: fields["journal"] = self.venue
        if self.doi: fields["doi"] = self.doi
        if self.url: fields["url"] = self.url
        esc = lambda x: x.replace("&", "\\&")
        body = ",\n  ".join(f"{k} = {{{esc(v)}}}" for k, v in fields.items() if v)
        return f"@article{{{self.key},\n  {body}\n}}\n"

def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests not installed (needed for auto-cite).")
    headers = {"User-Agent": os.getenv("AUTOCITE_USER_AGENT", "AutoCite/1.0 (mailto:example@example.com)")}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def search_crossref(query: str, rows: int = 5, min_year: int = 2000) -> List[Ref]:
    data = _http_get(CROSSREF_API, {
        "query": query,
        "rows": rows,
        "sort": "is-referenced-by-count",
        "order": "desc",
        "filter": f"type:journal-article,from-pub-date:{min_year}-01-01",
    })
    items = data.get("message", {}).get("items", [])
    out: List[Ref] = []
    for it in items:
        title = (it.get("title") or [""])[0]
        year = str((it.get("issued", {}).get("date-parts", [[""]])[0] or [""])[0])
        doi = it.get("DOI", "")
        url = it.get("URL", "")
        authors_list = it.get("author", []) or []
        names = []
        for a in authors_list[:8]:
            last = a.get("family") or a.get("name") or ""
            first = a.get("given", "")
            names.append(f"{last}, {first}".strip(", "))
        authors = " and ".join(names)
        venue = (it.get("container-title") or [""])[0]
        base = re.sub(r"\W+", "", (authors_list[0].get("family") if authors_list else (title.split()[0] if title else "ref")).lower())
        yr = re.sub(r"\D", "", year)[:4] or "0000"
        first_word = re.sub(r"\W+", "", (title.split()[0] if title else "paper").lower())[:10]
        key = f"{base}{yr}_{first_word}"[:40]
        out.append(Ref(key=key, title=title, authors=authors, year=yr, venue=venue, doi=doi, url=url))
    return out

def collect_refs_for_chapter(chapter_title: str, key_points: List[str], per_topic: int = 3, budget: int = 12) -> List[Ref]:
    bag: List[Ref] = []
    seen = set()
    topics = [chapter_title] + list(key_points or [])
    for t in topics:
        refs = search_crossref(t, rows=per_topic)
        for r in refs:
            dup_key = r.doi or r.key
            if dup_key in seen:
                continue
            seen.add(dup_key)
            bag.append(r)
            if len(bag) >= budget:
                return bag
        time.sleep(0.3)  # polite pacing
    return bag

# =========================
# Orchestration
# =========================
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
                 temperature: float = 0.7, model: str = "", bibliography: Optional[Dict[str, Dict[str, Any]]] = None):
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
        self.bibliography: Dict[str, Dict[str, Any]] = bibliography or {}
        self.auto_cite: bool = False
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
        text = data.model_dump_json(indent=2, ensure_ascii=False) if isinstance(data, BaseModel) else json.dumps(data, indent=2, ensure_ascii=False)
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
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
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
        self.plan = BookPlan(**data)
        self._save_json(self.plan_dir / "plan.json", self.plan)
        (self.plan_dir / "synopsis.md").write_text(self.plan.synopsis, encoding="utf-8")
        self.checkpoint.log({"stage": "plan", "total_target_words": self.plan.total_target_words})
        return self.plan

    def draft_chapter(self, ch: ChapterPlan, prev_titles: List[str]) -> str:
        meta = self.generate_meta()
        plan = self.generate_plan()

        # Auto-cite: fetch references per chapter and merge into bibliography
        if self.auto_cite:
            refs = collect_refs_for_chapter(ch.title, ch.key_points, per_topic=3, budget=12)
            for r in refs:
                self.bibliography[r.key] = {
                    "title": r.title, "author": r.authors, "year": r.year,
                    "journal": r.venue, "doi": r.doi, "url": r.url,
                }
            # Optional: write per-chapter bib for inspection
            (self.plan_dir / f"auto_refs_ch{ch.number:02d}.bib").write_text(
                "".join(r.to_bibtex() for r in refs), encoding="utf-8"
            )

        bib_keys = "\n".join(
            f"- {v.get('title','(untitled)')} → {k}" for k, v in self.bibliography.items()
        ) or "(none provided)"

        prompt = CHAPTER_DRAFT_PROMPT.format(
            ch_num=ch.number,
            ch_title=ch.title,
            target_words=ch.target_words,
            meta_json=meta.model_dump_json(indent=2, ensure_ascii=False),
            synopsis=plan.synopsis,
            chapter_json=ch.model_dump_json(indent=2, ensure_ascii=False),
            prev_titles="\n".join(f"- {t}" for t in prev_titles) if prev_titles else "(none)",
            bib_keys=bib_keys,
        )
        return self.llm.complete(SYSTEM_POLICY, prompt, temperature=self.temperature, max_tokens=5000, model=self.model)

    def run(self):
        meta = self.generate_meta()
        plan = self.generate_plan()

        chapter_files: List[Path] = []
        prev_titles: List[str] = []

        for ch in tqdm(plan.chapters, desc="Drafting chapters"):
            ch_name = f"{ch.number:02d}_{self._sanitize_filename(ch.title)}.md"
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

    # ----- Helpers -----
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        s = re.sub(r"[^A-Za-z0-9\- _]+", "", name).strip().replace(" ", "_")
        return s[:120] if s else "chapter"

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
        manuscript = "\n\n".join(front + parts) + "\n"

        # Append References chapter (only cited keys)
        if self.bibliography:
            cited = self._extract_citekeys(manuscript)
            bib_md = self._format_bibliography(cited)
            manuscript += "\n\n---\n\n# References\n\n" + bib_md + "\n"
        return manuscript

    def _extract_citekeys(self, text: str) -> List[str]:
        keys = set(re.findall(r"\[@([^\]]+)\]", text))
        expanded = set()
        for k in keys:
            for part in re.split(r"[;,]\s*", k):
                if part:
                    expanded.add(part.strip(" @"))
        return sorted(expanded)

    def _format_bibliography(self, keys: List[str]) -> str:
        lines = []
        for k in keys:
            entry = self.bibliography.get(k)
            if not entry:
                lines.append(f"- **MISSING** citation for key `{k}` — please supply in bibliography.")
                continue
            authors = entry.get("author") or entry.get("authors") or ""
            year = entry.get("year") or entry.get("date", "")[0:4]
            title = entry.get("title", "Untitled")
            journal = entry.get("journal") or entry.get("booktitle") or entry.get("publisher") or ""
            doi = entry.get("doi", "")
            url = entry.get("url", "")
            tail = f" {journal}" if journal else ""
            tail += f". DOI: {doi}" if doi else ""
            tail += f". {url}" if url else ""
            lines.append(f"- {authors} ({year}). *{title}*.{tail}")
        return "\n".join(lines)

    # ----- Exports -----
    def _export_epub(self, meta: BookMeta, book_md: str):
        if epub is None:
            raise RuntimeError("ebooklib not installed")
        bk = epub.EpubBook()
        bk.set_identifier("ai-ebook-" + re.sub(r"\W+", "-", meta.title.lower()))
        bk.set_title(meta.title)
        bk.set_language("en")
        bk.add_author(meta.author)
        chunks = re.split(r"\n(?=# )", book_md)
        spine = []
        toc = []
        for i, chunk in enumerate(chunks, 1):
            html = markdown.markdown(chunk) if markdown else f"<pre>{chunk}</pre>"
            c = epub.EpubHtml(title=f"Section {i}", file_name=f"sec_{i}.xhtml", content=html)
            bk.add_item(c)
            spine.append(c)
            toc.append(c)
        bk.toc = tuple(toc)
        bk.spine = ["nav"] + spine
        bk.add_item(epub.EpubNcx())
        bk.add_item(epub.EpubNav())
        out = self.outdir / "book.epub"
        epub.write_epub(str(out), bk)
        console.print(f"[green]EPUB written:[/] {out}")

    def _export_docx(self, book_md: str):
        if Document is None:
            raise RuntimeError("python-docx not installed")
        from markdown import markdown as md_to_html
        try:
            from bs4 import BeautifulSoup  # type: ignore
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
                if txt.strip():
                    doc.add_paragraph(txt)
        out = self.outdir / "book.docx"
        doc.save(str(out))
        console.print(f"[green]DOCX written:[/] {out}")

    def _export_pdf(self, book_md: str):
        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.pdfgen import canvas as pdf_canvas
            from reportlab.lib.units import inch
        except Exception:
            raise RuntimeError("reportlab not installed; for better PDFs, use pandoc or LaTeX")
        out_path = self.outdir / "book.pdf"
        c = pdf_canvas.Canvas(str(out_path), pagesize=LETTER)
        width, height = LETTER
        margin = 0.75 * inch
        y = height - margin
        for line in book_md.splitlines():
            if not line.strip():
                y -= 14
                if y < margin:
                    c.showPage(); y = height - margin
                continue
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
        console.print(f"[green]PDF written:[/] {out_path}")

# =========================
# CLI
# =========================
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
    p = argparse.ArgumentParser(description="Generate a full e-book from a single idea using an LLM.")
    p.add_argument("--idea", type=str, required=False, default="An untitled book idea",
                   help="Single-sentence core idea for the book.")
    p.add_argument("--genre", type=str, default="non-fiction")
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
    # Bibliography / auto-cite
    p.add_argument("--bibtex", type=Path, default=None, help="Optional path to a .bib file of sources.")
    p.add_argument("--citations-json", type=Path, default=None, help="Optional path to citations JSON (list or {key: entry}).")
    p.add_argument("--auto-cite", action="store_true", help="Auto-search literature per chapter (Crossref).")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    llm = build_llm(args.backend, args.model)

    # Load user-provided bibliography (optional)
    bibliography: Dict[str, Dict[str, Any]] = {}
    if args.bibtex and args.bibtex.exists():
        if bibtexparser is None:
            console.print("[yellow]BibTeX provided but bibtexparser not installed. Skipping.")
        else:
            with open(args.bibtex, "r", encoding="utf-8") as bf:
                db = bibtexparser.load(bf)
            for entry in db.entries:
                key = entry.get("ID") or entry.get("key") or entry.get("citation_key")
                if key:
                    bibliography[key] = entry
    if args.citations_json and args.citations_json.exists():
        try:
            cj = json.loads(args.citations_json.read_text(encoding="utf-8"))
            if isinstance(cj, list):
                for i, e in enumerate(cj, 1):
                    key = e.get("key") or e.get("id") or f"ref{i:04d}"
                    bibliography[key] = e
            elif isinstance(cj, dict):
                bibliography.update({str(k): v for k, v in cj.items()})
        except Exception as e:
            console.print(f"[yellow]Failed to read citations JSON: {e}")

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
        bibliography=bibliography,
    )
    builder.auto_cite = args.auto_cite

    builder.run()

if __name__ == "__main__":
    main()
