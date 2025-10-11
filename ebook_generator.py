#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ebook_generator_legacy_like.py
==============================
A stable e‑book generator matching the earlier working flow, with separate models for
planning vs drafting and a fallback model:

  --plan-model gpt-4.1 \
  --draft-model gpt-5 \
  --fallback-model gpt-4.1 \

What it does
------------
• Plans a chapter outline from a single idea (JSON, robustly parsed with heuristics)
• Drafts each chapter with continuity prompts (progress bar via tqdm)
• Assembles a single Markdown manuscript with a simple (non-hyperlinked) ToC
• Optional: suppress picture placeholders with --no-figures
• Style guardrails to reduce the “AI-ish” voice (no mid‑sentence hyphen tics)

Dependencies
------------
pip install "openai==1.*" "tqdm==4.*" "rich==13.*"
export OPENAI_API_KEY="sk-..."

Run
---
python ebook_generator_legacy_like.py \
  --idea "How to make 10,000 to one million dollars from stocks" \
  --genre "Technical" \
  --reading-level "High school" \
  --style "rigorous, precise" \
  --min-words 20000 --max-words 30000 \
  --plan-model gpt-4.1 \
  --draft-model gpt-5 \
  --fallback-model gpt-4.1 \
  --author "Xavier Cai" \
  --no-figures \
  --outdir ./book_stock

Outputs
-------
outdir/
  plan/
    plan.json, plan_raw.txt, plan_raw_retry.txt (when needed)
  chapters/
    NN_Title.md
  book.md
"""

from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from rich.console import Console

# Optional retry for flaky calls
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:
    retry = None  # graceful: we won't use decorator
    stop_after_attempt = wait_exponential = retry_if_exception_type = None

# -------- OpenAI client (tolerant to model quirks) --------
try:
    import openai  # type: ignore
except Exception:
    openai = None

console = Console()

def _sanitize_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\- _]+", "", (name or "chapter")).strip().replace(" ", "_")
    return s[:120] if s else "chapter"

class LLMError(Exception):
    pass


class OpenAIClient:
    def __init__(self):
        if openai is None:
            raise RuntimeError("openai package not installed. Run: pip install 'openai==1.*'")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _chat_once(self, system: str, user: str, model: str, max_tokens: int, temperature: Optional[float]) -> str:
        base = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        def call(**extra):
            return self.client.chat.completions.create(**{**base, **extra})

        # Try standard
        try:
            kwargs = {"max_tokens": max_tokens}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = call(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)

            # If temperature unsupported for this model, retry without it
            if "temperature" in msg.lower() and "unsupported" in msg.lower():
                try:
                    resp = call(max_tokens=max_tokens)
                    return resp.choices[0].message.content or ""
                except Exception as e2:
                    msg = str(e2)

            # If needs max_completion_tokens (newer models)
            if "max_completion_tokens" in msg:
                try:
                    extra_body = {"max_completion_tokens": max_tokens}
                    if temperature is not None:
                        extra_body["temperature"] = temperature
                    resp = call(extra_body=extra_body)
                    return resp.choices[0].message.content or ""
                except Exception as e3:
                    msg = str(e3)

            raise LLMError(msg)

    def chat(self, system: str, user: str, *, model: str, fallback_model: Optional[str], max_tokens: int, temperature: Optional[float]) -> str:
        # light retry loop
        for attempt in range(3):
            try:
                return self._chat_once(system, user, model, max_tokens, temperature)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        # fallback model
        if fallback_model:
            console.print(f"[yellow]Primary model failed; trying fallback '{fallback_model}'...[/]")
            for attempt in range(2):
                try:
                    return self._chat_once(system, user, fallback_model, max_tokens, temperature)
                except Exception as e:
                    last_err = e
                    time.sleep(1.5 * (attempt + 1))
        raise last_err  # type: ignore


# -------- JSON helpers --------
def _extract_json_block(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m: return m.group(1)
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return text[s:e+1]
    raise ValueError("No JSON object found")

def _parse_or_repair_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(_extract_json_block(raw))
    except Exception:
        pass
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)  # remove trailing commas
    try:
        return json.loads(_extract_json_block(cleaned))
    except Exception:
        raise ValueError("Could not parse JSON")


# -------- Prompts --------
SYSTEM = (
    "You are a seasoned human author and editor. "
    "Write original, coherent, factual prose. Avoid the 'AI voice': "
    "no filler cliches, no mid‑sentence hyphens used like dashes. "
    "Vary sentence length and rhythm. Prefer smooth transitions and concrete examples."
)

PLAN_PROMPT = """
Plan a complete long‑form book from the idea and constraints below.
Return ONLY a JSON object with keys:
- title (str), subtitle (str), genre (str), audience (str), reading_level (str), style (str), author (str)
- synopsis (str)
- total_target_words (int)
- chapters: [ {{
    number (int), title (str), purpose (str),
    key_points (list[str], >=3),
    target_words (int),
    sections: [ {{title (str), summary (str)}} ]  (>=3)
  }}, ... ]

Constraints:
- 16–22 chapters total.
- Each chapter target_words ~ {per_min}–{per_max}.
- No commentary outside JSON.

Idea: {idea}
Genre: {genre}
Audience: {audience}
Reading level: {reading_level}
Style: {style}
Author: {author}
Total target words: {min_words}–{max_words}
""".strip()

CHAPTER_PROMPT = """
Draft Chapter {num}: "{title}" for the book described below.
Voice: human, confident, not robotic. Do not use spaced hyphens mid‑sentence; use natural punctuation.
Target ~{target_words} words (±10%).

Include:
- A brief *italicized* opener (1–2 sentences).
- H3 section headings that exactly match the plan's section titles.
- Smooth transitions across sections.
{figure_rule}
Finish with 3–5 key takeaways as a bullet list.

Book metadata (JSON):
{meta_json}

Chapter plan (JSON):
{chapter_json}

Previously drafted chapter titles:
{prev_titles}
""".strip()


# -------- Builder --------
class BookBuilder:
    def __init__(self, client: OpenAIClient, outdir: Path,
                 idea: str, genre: str, audience: str, reading_level: str, style: str,
                 min_words: int, max_words: int, author: str,
                 plan_model: str, draft_model: str, fallback_model: Optional[str],
                 no_figures: bool):
        self.client = client
        self.outdir = outdir
        self.plan_dir = outdir / "plan"
        self.chapters_dir = outdir / "chapters"
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)

        self.idea = idea
        self.genre = genre
        self.audience = audience or "general audience"
        self.reading_level = reading_level
        self.style = style
        self.min_words = min_words
        self.max_words = max_words
        self.author = author or "Anonymous"

        self.plan_model = plan_model
        self.draft_model = draft_model
        self.fallback_model = fallback_model
        self.no_figures = no_figures

        self.plan: Dict[str, Any] = {}

    def _ask_json(self, prompt: str, model: str, tokens: int = 4000, temp: float = 0.6) -> dict:
        txt = self.client.chat(SYSTEM, prompt, model=model, fallback_model=self.fallback_model, max_tokens=tokens, temperature=temp)
        return _parse_or_repair_json(txt)

    def generate_plan(self) -> Dict[str, Any]:
        approx_chapters = 18
        per_min = max(1200, self.min_words // max(12, approx_chapters - 2))
        per_max = max(1800, self.max_words // max(10, approx_chapters - 2))
        prompt = PLAN_PROMPT.format(
            idea=self.idea, genre=self.genre, audience=self.audience, reading_level=self.reading_level,
            style=self.style, author=self.author, min_words=self.min_words, max_words=self.max_words,
            per_min=per_min, per_max=per_max,
        )
        try:
            data = self._ask_json(prompt, model=self.plan_model, tokens=4200, temp=0.6)
        except Exception as e:
            console.print(f"[yellow]Initial plan parsing failed: {e}. Retrying with stricter instruction...[/]")
            strict = prompt + "\n\nReturn ONLY valid JSON. No prose."
            data = self._ask_json(strict, model=self.plan_model, tokens=4400, temp=0.6)

        # minimal validation & defaults
        if "chapters" not in data or not isinstance(data["chapters"], list) or len(data["chapters"]) < 12:
            # one more attempt
            strict2 = prompt + "\n\nReturn 16–22 chapters. Only JSON."
            data = self._ask_json(strict2, model=self.plan_model, tokens=4600, temp=0.6)

        data.setdefault("title", "Untitled Book")
        data.setdefault("subtitle", "")
        data.setdefault("genre", self.genre)
        data.setdefault("audience", self.audience)
        data.setdefault("reading_level", self.reading_level)
        data.setdefault("style", self.style)
        data.setdefault("author", self.author)
        data.setdefault("synopsis", "")
        data.setdefault("total_target_words", max(self.min_words, self.max_words))

        # Save raw/parsed
        (self.plan_dir / "plan.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data

    def _draft_whole_chapter(self, ch: Dict[str, Any], meta: Dict[str, Any], prev_titles: List[str], model: str) -> str:
        figure_rule = "Do NOT include any figure or image placeholders." if self.no_figures else (
            "You MAY include an occasional placeholder like *[Figure: short caption]* but do not invent data."
        )
        prompt = CHAPTER_PROMPT.format(
            num=ch["number"], title=ch["title"], target_words=ch.get("target_words", 1600),
            figure_rule=figure_rule,
            meta_json=json.dumps(meta, indent=2, ensure_ascii=False),
            chapter_json=json.dumps(ch, indent=2, ensure_ascii=False),
            prev_titles="\n".join(f"- {t}" for t in prev_titles) if prev_titles else "(none)",
        )
        txt = self.client.chat(SYSTEM, prompt, model=model, fallback_model=self.fallback_model, max_tokens=5200, temperature=0.7).strip()
        # Clean common AI-ish dash patterns
        txt = re.sub(r"\s+-\s+(?=\w)", " — ", txt)
        return txt

    def draft_chapter(self, ch: Dict[str, Any], meta: Dict[str, Any], prev_titles: List[str]) -> str:
        # try primary draft model
        try:
            txt = self._draft_whole_chapter(ch, meta, prev_titles, self.draft_model)
            if len(txt) >= 300:
                return txt
        except Exception as e:
            console.print(f"[yellow]Draft on '{self.draft_model}' failed: {e}[/]")

        # retry once with fallback model
        if self.fallback_model:
            txt = self._draft_whole_chapter(ch, meta, prev_titles, self.fallback_model)
            if len(txt) >= 300:
                return txt
        raise RuntimeError(f"Chapter {ch.get('number')} draft came back empty.")

    def assemble(self, plan: Dict[str, Any], chapter_files: List[Path]) -> str:
        lines = []
        lines.append(f"# {plan.get('title','Untitled Book')}")
        if plan.get("subtitle"):
            lines.append(plan["subtitle"])
        lines.append(f"Author: {plan.get('author','Anonymous')}")
        lines.append("\n---\n")
        lines.append("## About this book")
        lines.append(plan.get("synopsis",""))
        lines.append("\n---\n")
        lines.append("## Table of Contents")
        for ch in plan["chapters"]:
            lines.append(f"- Chapter {ch.get('number')}: {ch.get('title','')}")
        lines.append("\n---\n")
        for f in chapter_files:
            lines.append(f.read_text(encoding="utf-8"))
            lines.append("")
        md = "\n".join(lines)
        (self.outdir / "book.md").write_text(md, encoding="utf-8")
        return md

    def run(self):
        # Save a literal copy of the last plan prompt/output for debugging
        plan_prompt_copy = self.outdir / "plan" / "plan_prompt.txt"
        plan_prompt_copy.parent.mkdir(parents=True, exist_ok=True)

        # Build plan
        prompt_snapshot = f"(Idea: {self.idea}) (Genre: {self.genre}) (Audience: {self.audience}) (Reading: {self.reading_level}) (Style: {self.style})"
        plan_prompt_copy.write_text(prompt_snapshot, encoding="utf-8")
        plan = self.generate_plan()

        meta = {
            "title": plan.get("title","Untitled Book"),
            "subtitle": plan.get("subtitle",""),
            "genre": plan.get("genre", self.genre),
            "audience": plan.get("audience", self.audience),
            "reading_level": plan.get("reading_level", self.reading_level),
            "style": plan.get("style", self.style),
            "author": plan.get("author", self.author),
            "synopsis": plan.get("synopsis",""),
        }

        chapter_files: List[Path] = []
        prev_titles: List[str] = []

        for ch in tqdm(plan["chapters"], desc="Drafting chapters"):
            name = f"{int(ch.get('number', len(prev_titles)+1)):02d}_{_sanitize_filename(ch.get('title','chapter'))}.md"
            path = self.chapters_dir / name
            if path.exists() and path.stat().st_size > 200:
                prev_titles.append(ch.get("title",""))
                chapter_files.append(path)
                continue
            text = self.draft_chapter(ch, meta, prev_titles)
            doc = f"# {meta['title']}: {ch.get('title','')}\n\n{text}\n"
            path.write_text(doc, encoding="utf-8")
            prev_titles.append(ch.get("title",""))
            chapter_files.append(path)

        self.assemble(plan, chapter_files)
        console.print(f"\n[bold green]Done.[/] Manuscript at:", self.outdir / "book.md")


# -------- CLI --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an e‑book from a single idea (plan → draft → assemble).")
    p.add_argument("--idea", type=str, required=True, help="Single‑sentence core idea.")
    p.add_argument("--genre", type=str, default="non-fiction")
    p.add_argument("--audience", type=str, default="general audience")
    p.add_argument("--reading-level", type=str, default="grade 10")
    p.add_argument("--style", type=str, default="clear, engaging, practical")
    p.add_argument("--min-words", type=int, default=50_000)
    p.add_argument("--max-words", type=int, default=60_000)

    p.add_argument("--plan-model", type=str, default="gpt-4.1", help="Model for planning JSON (e.g., gpt-4.1).")
    p.add_argument("--draft-model", type=str, default="gpt-5", help="Model for chapter drafting (e.g., gpt-5).")
    p.add_argument("--fallback-model", type=str, default="gpt-4.1", help="Fallback model for any stage.")

    p.add_argument("--author", type=str, default="Anonymous")
    p.add_argument("--no-figures", action="store_true", help="Suppress figure/image placeholders.")
    p.add_argument("--outdir", type=Path, default=Path("./book_out"))
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    client = OpenAIClient()
    builder = BookBuilder(
        client=client,
        outdir=args.outdir,
        idea=args.idea,
        genre=args.genre,
        audience=args.audience,
        reading_level=args.reading_level,
        style=args.style,
        min_words=args.min_words,
        max_words=args.max_words,
        author=args.author,
        plan_model=args.plan_model,
        draft_model=args.draft_model,
        fallback_model=args.fallback_model,
        no_figures=args.no_figures,
    )
    builder.run()


if __name__ == "__main__":
    main()
