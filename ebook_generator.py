#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ebook_generator_resilient_v3.py
--------------------------------
Robust AI book generator (OpenAI backend) that:
- Plans (strict JSON), drafts with retries/fallbacks, assembles Markdown
- Auto-cites via Crossref (no fabricated citations)
- Outputs a single book.md; convert to EPUB/DOCX/PDF with md_to_book.py
- NEW: --no-figures flag to suppress figure/table placeholders
- NEW: APA-style References chapter
"""
from __future__ import annotations

import argparse, json, os, re, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Optional deps
try:
    import openai  # type: ignore
except Exception:
    openai = None
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ---------- JSON helpers ----------
def _extract_json_block(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        return m.group(1)
    raise ValueError("No fenced JSON block found")

def _extract_balanced_json(text: str) -> str:
    i = text.find("{")
    if i == -1:
        raise ValueError("No '{' found")
    depth = 0
    in_str = False
    esc = False
    start = None
    for idx in range(i, len(text)):
        ch = text[idx]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{":
                depth += 1
                if start is None: start = idx
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:idx+1]
    raise ValueError("No balanced JSON object found")

def _parse_or_repair_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(_extract_json_block(raw))
    except Exception:
        pass
    try:
        blob = _extract_balanced_json(raw)
        return json.loads(blob)
    except Exception:
        pass
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        blob = _extract_balanced_json(cleaned)
        return json.loads(blob)
    except Exception:
        pass
    raise ValueError("Could not parse JSON")

# ---------- Models ----------
class BookMeta(BaseModel):
    title: str
    subtitle: str
    author: str
    genre: str
    audience: str
    reading_level: str
    style: str
    promise: str

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

# ---------- Prompts ----------
SYSTEM_POLICY = (
    "You are a seasoned book author and editor. Write original, coherent, factual prose. "
    "Avoid harmful content. Do not fabricate citations. Maintain consistency."
)

META_PROMPT = """
Return ONLY valid JSON (no prose). Follow this template:

```json
{{
  "title": "Your Title",
  "subtitle": "Optional subtitle",
  "author": "Your Name",
  "genre": "non-fiction",
  "audience": "target readers",
  "reading_level": "university",
  "style": "scholarly, rigorous, precise",
  "promise": "One-sentence market hook"
}}
```

Book idea: {idea}
Preferred genre: {genre}
Target audience: {audience}
Reading level: {reading_level}
Style notes: {style}
""".strip()

PLAN_PROMPT = """
Return ONLY valid JSON (no prose). Follow this template and constraints:

Template:
```json
{{
  "synopsis": "One-paragraph overview",
  "parts": ["Part I", "Part II"],
  "total_target_words": 25000,
  "chapters": [
    {{
      "number": 1,
      "title": "Chapter title",
      "purpose": "What this chapter achieves",
      "key_points": ["point A", "point B", "point C"],
      "target_words": 2400,
      "sections": [
        {{"title": "Section A", "summary": "What this section covers"}},
        {{"title": "Section B", "summary": "What this section covers"}},
        {{"title": "Section C", "summary": "What this section covers"}}
      ]
    }}
  ]
}}
```

Constraints:
- Chapters count: between {min_chapters} and {max_chapters}.
- Each chapter target_words between 2000 and 3200.
- No prose outside the JSON.

Metadata JSON:
{meta_json}

Book idea: {idea}
Target total words: {min_words}–{max_words}
""".strip()

REPAIR_PLAN_PROMPT = """
You are given a book plan JSON below. It currently violates constraints.
Rewrite it to satisfy ALL constraints and return ONLY valid JSON with the same schema.

Constraints:
- Chapters count: between {min_chapters} and {max_chapters}.
- Each chapter must include: number, title, purpose, key_points (>=3), target_words (2000–3200),
  and sections (>=3) each with title and summary.
- Keep synopsis and parts; adjust as needed for coherence.
- Maintain the same keys. No prose outside JSON.

Existing plan JSON:
```json
{plan_json}
```
""".strip()

SECTION_DRAFT_PROMPT = """
Write the section **{section_title}** for Chapter {ch_num}: "{ch_title}".
Follow the book's voice and the section summary strictly.

Output plain Markdown prose (no JSON), ~{words} words.
Use `### {section_title}` as the section heading.
{no_fig_rules}

Citations:
- If a suitable key exists in the bibliography list, cite with `[@key]` near relevant claims.
- If none fits, write without a citation. Never invent keys.

Section summary:
{summary}

Book metadata:
{meta_json}

Book synopsis:
{synopsis}

Bibliography keys (title → key):
{bib_keys}
""".strip()

# ---------- OpenAI client ----------
class LLMError(Exception): pass

class OpenAIClient:
    def __init__(self, model: str):
        if openai is None:
            raise RuntimeError("Install openai: pip install 'openai==1.*'")
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, prompt: str, *, max_tokens: int = 1500, model: Optional[str] = None) -> str:
        m = model or self.model
        base = {"model": m, "messages": [{"role":"system","content":system},{"role":"user","content":prompt}]}
        def chat_call(extra): return self.client.chat.completions.create(**{**base, **extra})
        try:
            r = chat_call({"max_tokens": max_tokens})
            content = r.choices[0].message.content or ""
            if content.strip(): return content
        except Exception as e:
            msg = str(e)
            if "max_completion_tokens" in msg:
                r = chat_call({"extra_body":{"max_completion_tokens": max_tokens}})
                content = r.choices[0].message.content or ""
                if content.strip(): return content
        # Responses fallback
        r2 = self.client.responses.create(
            model=m,
            input=[{"role":"system","content":system},{"role":"user","content":prompt}],
            max_output_tokens=max_tokens,
        )
        return getattr(r2, "output_text", None) or ""

    def structured(self, system: str, prompt: str, *, max_tokens: int = 2000, model: Optional[str] = None) -> dict:
        m = model or self.model
        # A) chat.completions with response_format=json_object
        try:
            r = self.client.chat.completions.create(
                model=m,
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":prompt + "\n\nReturn ONLY a JSON object."}
                ],
                response_format={"type":"json_object"},
                max_tokens=max_tokens,
            )
            content = r.choices[0].message.content or ""
            return _parse_or_repair_json(content)
        except Exception:
            pass
        # B) Responses API
        try:
            r2 = self.client.responses.create(
                model=m,
                input=[{"role":"system","content":system},{"role":"user","content":prompt}],
                max_output_tokens=max_tokens,
            )
            txt = getattr(r2, "output_text", None) or ""
            return _parse_or_repair_json(txt)
        except Exception:
            pass
        # C) Tagged JSON fallback
        tagged = prompt + "\n\nReturn ONLY JSON between tags:\n<<<JSON>>>\n{...}\n<<<ENDJSON>>>"
        try:
            r3 = self.client.chat.completions.create(
                model=m,
                messages=[{"role":"system","content":system},{"role":"user","content":tagged}],
                max_tokens=max_tokens,
            )
            txt = r3.choices[0].message.content or ""
            mtag = re.search(r"<<<JSON>>>(.*)<<<ENDJSON>>>", txt, re.S)
            if not mtag: raise ValueError("No tagged JSON")
            return _parse_or_repair_json(mtag.group(1))
        except Exception as e:
            raise LLMError(f"structured(): {e}")

# ---------- Auto-citation (Crossref) ----------
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
    def to_bib(self) -> Dict[str, Any]:
        d = {"title": self.title, "author": self.authors, "year": self.year}
        if self.venue: d["journal"] = self.venue
        if self.doi: d["doi"] = self.doi
        if self.url: d["url"] = self.url
        return d

def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("Install requests for auto-cite: pip install 'requests==2.*'")
    headers = {"User-Agent": os.getenv("AUTOCITE_USER_AGENT", "AutoCite/1.0 (mailto:example@example.com)")}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def _safe_lower(x: Optional[str]) -> str:
    return (x or "").lower()

def search_crossref(query: str, rows: int = 5, min_year: int = 2000) -> List[Ref]:
    try:
        data = _http_get(CROSSREF_API, {
            "query": query, "rows": rows, "sort": "is-referenced-by-count", "order": "desc",
            "filter": f"type:journal-article,from-pub-date:{min_year}-01-01",
        })
    except Exception:
        return []
    items = data.get("message", {}).get("items", []) or []
    out: List[Ref] = []
    for it in items:
        title = ((it.get("title") or [""])[0]) or ""
        year = str((it.get("issued", {}).get("date-parts", [[""]])[0] or [""])[0])
        doi = it.get("DOI", "") or ""
        url = it.get("URL", "") or ""
        authors_list = it.get("author", []) or []
        names = []
        for a in authors_list[:8]:
            last = a.get("family") or a.get("name") or ""
            first = a.get("given", "")
            nm = f"{last}, {first}".strip(", ")
            if nm: names.append(nm)
        authors = " and ".join(names)
        venue = (it.get("container-title") or [""])[0] or ""
        base_source = (authors_list[0].get("family") if authors_list and isinstance(authors_list[0], dict) else None) \
                      or (title.split()[0] if title else "") \
                      or "ref"
        base = re.sub(r"\W+", "", _safe_lower(base_source))
        yr = re.sub(r"\D", "", year)[:4] or "0000"
        first_word = re.sub(r"\W+", "", _safe_lower((title.split()[0] if title else "")))[:10] or "paper"
        key = (f"{base}{yr}_{first_word}")[:40] or f"ref{yr}"
        out.append(Ref(key=key, title=title, authors=authors, year=yr, venue=venue, doi=doi, url=url))
    return out

def collect_refs_for_chapter(ch_title: str, key_points: List[str], per_topic: int = 3, budget: int = 12) -> Dict[str, Dict[str, Any]]:
    bag: Dict[str, Dict[str, Any]] = {}
    seen = set()
    topics = [t for t in [ch_title] + list(key_points or []) if t]
    if not topics:
        return bag
    for t in topics:
        for r in search_crossref(t, rows=per_topic):
            dup = r.doi or r.key
            if dup in seen: continue
            seen.add(dup)
            bag[r.key] = r.to_bib()
            if len(bag) >= budget: return bag
        time.sleep(0.15)
    return bag

# ---------- Builder ----------
class BookBuilder:
    def __init__(self, llm: OpenAIClient, outdir: Path, idea: str, genre: str, audience: str,
                 reading_level: str, style: str, min_words: int, max_words: int,
                 plan_model: str, draft_model: str, auto_cite: bool,
                 min_chapters: int, max_chapters: int, fallback_model: Optional[str],
                 draft_retries: int, section_fallback: bool, no_figures: bool):
        self.llm = llm
        self.outdir = outdir
        self.plan_dir = outdir / "plan"
        self.chapters_dir = outdir / "chapters"
        self.idea, self.genre, self.audience = idea, genre, audience
        self.reading_level, self.style = reading_level, style
        self.min_words, self.max_words = min_words, max_words
        self.plan_model, self.draft_model = plan_model, draft_model
        self.auto_cite = auto_cite
        self.min_chapters, self.max_chapters = min_chapters, max_chapters
        self.fallback_model = fallback_model
        self.draft_retries = draft_retries
        self.section_fallback = section_fallback
        self.no_figures = no_figures
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        self.meta: Optional[BookMeta] = None
        self.plan: Optional[BookPlan] = None
        self.bibliography: Dict[str, Dict[str, Any]] = {}
        self.checkpoints = outdir / "checkpoints.jsonl"

    def _log(self, event: Dict[str, Any]):
        event["ts"] = datetime.now(timezone.utc).isoformat()
        with self.checkpoints.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_json(self, path: Path, data: Any):
        if isinstance(data, BaseModel):
            payload = data.model_dump()
        else:
            payload = data
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def generate_meta(self) -> BookMeta:
        if self.meta: return self.meta
        prompt = META_PROMPT.format(
            idea=self.idea, genre=self.genre, audience=self.audience,
            reading_level=self.reading_level, style=self.style
        )
        try:
            meta_obj = self.llm.structured(SYSTEM_POLICY, prompt, max_tokens=800, model=self.plan_model)
        except Exception as e:
            print(f"structured meta failed, falling back: {e}")
            text = self.llm.complete(SYSTEM_POLICY, prompt, max_tokens=1200, model=self.plan_model)
            (self.plan_dir / "meta_raw.txt").write_text(text, encoding="utf-8")
            meta_obj = _parse_or_repair_json(text)

        def f(x, d): return (x or "").strip() or d
        meta_obj = {
            "title": f(meta_obj.get("title"), "Untitled Book"),
            "subtitle": f(meta_obj.get("subtitle"), "A Project Generated by AI"),
            "author": f(meta_obj.get("author"), "Anonymous"),
            "genre": f(meta_obj.get("genre"), self.genre),
            "audience": f(meta_obj.get("audience"), self.audience),
            "reading_level": f(meta_obj.get("reading_level"), self.reading_level),
            "style": f(meta_obj.get("style"), self.style),
            "promise": f(meta_obj.get("promise"), "A fresh perspective created from a single idea.")
        }
        self.meta = BookMeta(**meta_obj)
        self._save_json(self.plan_dir / "meta.json", self.meta)
        self._log({"stage":"meta","meta":self.meta.model_dump()})
        return self.meta

    def _repair_plan_to_constraints(self, plan_obj: dict) -> dict:
        prompt = REPAIR_PLAN_PROMPT.format(
            min_chapters=self.min_chapters, max_chapters=self.max_chapters,
            plan_json=json.dumps(plan_obj, ensure_ascii=False, indent=2)
        )
        try:
            repaired = self.llm.structured(SYSTEM_POLICY, prompt, max_tokens=4500, model=self.plan_model)
            return repaired
        except Exception:
            text = self.llm.complete(SYSTEM_POLICY, prompt, max_tokens=5000, model=self.plan_model)
            return _parse_or_repair_json(text)

    def generate_plan(self) -> BookPlan:
        if self.plan: return self.plan
        meta = self.generate_meta()
        prompt = PLAN_PROMPT.format(
            meta_json=json.dumps(meta.model_dump(), indent=2, ensure_ascii=False),
            idea=self.idea, min_words=self.min_words, max_words=self.max_words,
            min_chapters=self.min_chapters, max_chapters=self.max_chapters
        )
        try:
            plan_obj = self.llm.structured(SYSTEM_POLICY, prompt, max_tokens=4000, model=self.plan_model)
        except Exception as e:
            print(f"structured plan failed, falling back: {e}")
            text = self.llm.complete(SYSTEM_POLICY, prompt, max_tokens=4000, model=self.plan_model)
            (self.plan_dir / "plan_raw.txt").write_text(text, encoding="utf-8")
            try:
                plan_obj = _parse_or_repair_json(text)
            except Exception:
                strict = prompt + "\n\nReturn ONLY valid JSON. No prose outside the JSON."
                text2 = self.llm.complete(SYSTEM_POLICY, strict, max_tokens=4500, model=self.plan_model)
                (self.plan_dir / "plan_raw_retry.txt").write_text(text2, encoding="utf-8")
                plan_obj = _parse_or_repair_json(text2)

        chs = plan_obj.get("chapters") or []
        if not isinstance(chs, list) or not (self.min_chapters <= len(chs) <= self.max_chapters):
            plan_obj = self._repair_plan_to_constraints(plan_obj)
            chs = plan_obj.get("chapters") or []

        if not isinstance(chs, list) or not (self.min_chapters <= len(chs) <= self.max_chapters):
            (self.plan_dir / "PLAN_PARSE_ERROR.txt").write_text(
                "Plan JSON missing/invalid. Check plan_raw*.txt for the model output.\n"
                f"Chapters found: {len(chs) if isinstance(chs, list) else 'N/A'}; "
                f"expected {self.min_chapters}-{self.max_chapters}.",
                encoding="utf-8"
            )
            raise RuntimeError("Planning failed: chapter count outside constraints.")

        self.plan = BookPlan(**plan_obj)
        self._save_json(self.plan_dir / "plan.json", self.plan)
        self._log({"stage":"plan","total_target_words":self.plan.total_target_words,"chapters":len(self.plan.chapters)})
        return self.plan

    def _chapter_common_rules(self) -> str:
        if self.no_figures:
            return (
                "Rules:\\n"
                "- Do NOT include any figure or table placeholders.\\n"
                "- Do NOT use image syntax or captions.\\n"
            )
        return (
            "Optional figure/table placeholders are allowed (e.g., *[Figure: ...]*), but do not invent data."
        )

    def _draft_whole_chapter(self, ch: ChapterPlan, prev_titles: List[str], model: str) -> str:
        meta = self.generate_meta(); plan = self.generate_plan()
        if self.auto_cite and not self.bibliography:
            refs = collect_refs_for_chapter(ch.title, ch.key_points)
            self.bibliography.update(refs)

        bib_keys = "\\n".join(f"- {v.get('title','(untitled)')} → {k}" for k,v in self.bibliography.items()) or "(none provided)"
        no_fig_rules = self._chapter_common_rules()

        prompt = f"""
Draft Chapter {ch.number}: "{ch.title}" for the book below. Use the plan faithfully.
Target ~{ch.target_words} words (±10%).

Include:
- *Italicized* opener (1–2 sentences).
- Markdown H3 section headings (### Title).
- {no_fig_rules}
- End with 3–5 reflective questions as bullets.

Citations policy:
- Cite only keys present in the bibliography list below, using Markdown `[@key]`.
- If no suitable source exists, write without a citation rather than fabricating one.

Book metadata:
{json.dumps(meta.model_dump(), indent=2, ensure_ascii=False)}

Book synopsis:
{plan.synopsis}

Chapter plan:
{json.dumps(ch.model_dump(), indent=2, ensure_ascii=False)}

Available bibliography keys (title → key):
{bib_keys}

Previously drafted chapter titles:
{chr(10).join(f"- {t}" for t in prev_titles) if prev_titles else "(none)"}
""".strip()
        txt = self.llm.complete(SYSTEM_POLICY, prompt, max_tokens=5500, model=model).strip()
        return txt

    def _draft_by_sections(self, ch: ChapterPlan, model: str) -> str:
        meta = self.generate_meta(); plan = self.generate_plan()
        if self.auto_cite and not self.bibliography:
            refs = collect_refs_for_chapter(ch.title, ch.key_points)
            self.bibliography.update(refs)
        bib_keys = "\\n".join(f"- {v.get('title','(untitled)')} → {k}" for k,v in self.bibliography.items()) or "(none provided)"
        words_per = max(300, int(ch.target_words / max(len(ch.sections), 1)))
        no_fig_rules = "Rules:\\n- Do NOT include any figure or table placeholders.\\n- Do NOT use image syntax or captions.\\n" if self.no_figures else ""

        parts = [f"# {meta.title}: {ch.title}", ""]
        parts.append("_This chapter explores the topic in depth, linking methods and evidence with practical implications._\\n")
        for sec in ch.sections:
            prompt = SECTION_DRAFT_PROMPT.format(
                section_title=sec.title, ch_num=ch.number, ch_title=ch.title,
                words=words_per, summary=sec.summary,
                meta_json=json.dumps(meta.model_dump(), indent=2, ensure_ascii=False),
                synopsis=plan.synopsis, bib_keys=bib_keys, no_fig_rules=no_fig_rules
            )
            text = self.llm.complete(SYSTEM_POLICY, prompt, max_tokens=1800, model=model).strip()
            if not text.startswith("### "):
                text = f"### {sec.title}\\n\\n{text}"
            parts.append(text)
            parts.append("")
            time.sleep(0.15)
        parts.append("**Reflective questions**")
        parts.append("- What assumptions were challenged in this section?")
        parts.append("- Which methods are most robust, and why?")
        parts.append("- Where could future research resolve uncertainties?")
        return "\\n".join(parts).strip() + "\\n"

    def draft_chapter(self, ch: ChapterPlan, prev_titles: List[str]) -> str:
        for attempt in range(self.draft_retries):
            txt = self._draft_whole_chapter(ch, prev_titles, self.draft_model)
            (self.chapters_dir / f"ch{ch.number:02d}_raw_attempt{attempt+1}.txt").write_text(txt, encoding="utf-8")
            if len(txt) > 300:
                return f"# {self.generate_meta().title}: {ch.title}\\n\\n{txt}\\n"
            time.sleep(0.4)
        if self.fallback_model:
            txt = self._draft_whole_chapter(ch, prev_titles, self.fallback_model)
            (self.chapters_dir / f"ch{ch.number:02d}_raw_fallback.txt").write_text(txt, encoding="utf-8")
            if len(txt) > 300:
                return f"# {self.generate_meta().title}: {ch.title}\\n\\n{txt}\\n"
        if self.section_fallback:
            doc = self._draft_by_sections(ch, self.fallback_model or self.draft_model)
            (self.chapters_dir / f"ch{ch.number:02d}_section_fallback.md").write_text(doc, encoding="utf-8")
            if len(doc) > 300:
                return doc
        raise RuntimeError(f"Chapter {ch.number} draft came back empty after retries and fallbacks.")

    # --- APA references ---
    def _apa_author_list(self, authors_str: str) -> str:
        parts = [a.strip() for a in re.split(r"\\s+and\\s+|,\\s*(?=[A-Z][a-z]+,)", authors_str) if a.strip()]
        def initials(name):
            if "," in name:
                last, firsts = [x.strip() for x in name.split(",", 1)]
            else:
                toks = name.split()
                last, firsts = toks[-1], " ".join(toks[:-1])
            inits = " ".join(f"{w[0]}." for w in re.split(r"[\\s\\-]+", firsts) if w)
            return f"{last}, {inits}".strip()
        formatted = [initials(p) for p in parts[:20]]
        if len(formatted) <= 2:
            return " & ".join(formatted)
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]

    def _format_bibliography_apa(self, keys: List[str]) -> str:
        lines = []
        for k in keys:
            e = self.bibliography.get(k)
            if not e:
                lines.append(f"- **MISSING** citation for key `{k}`")
                continue
            authors = e.get("author") or e.get("authors") or ""
            year = (e.get("year") or e.get("date","")[:4] or "n.d.").strip()
            title = e.get("title","Untitled").rstrip(".")
            journal = e.get("journal") or e.get("booktitle") or e.get("publisher") or ""
            volume = e.get("volume",""); issue = e.get("number") or e.get("issue","")
            pages = e.get("pages","")
            doi = e.get("doi",""); url = e.get("url","")

            author_apa = self._apa_author_list(authors) if authors else ""
            core = f"{author_apa} ({year}). {title}."
            if journal:
                core += f" *{journal}*"
                if volume:
                    core += f", *{volume}*"
                    if issue:
                        core += f"({issue})"
                if pages:
                    core += f", {pages}"
                core += "."
            if doi:
                core += f" https://doi.org/{doi.lstrip('https://doi.org/').lstrip('doi:').strip()}"
            elif url:
                core += f" {url}"
            lines.append(f"- {core}")
        return "\\n".join(lines)

    def _extract_citekeys(self, text: str) -> List[str]:
        #keys = set(re.findall(r"\\[@([^\\]]+)\\]", text))
        keys = set(re.findall(r"\[@([^\]]+)\]", text))
        expanded = set()
        for k in keys:
            for part in re.split(r"[;,]\\s*", k):
                if part:
                    expanded.add(part.strip(" @"))
        return sorted(expanded)

    
    def _strip_fig_placeholders(self, text: str) -> str:
    # Remove Markdown image lines like: ![alt](url)
        text = re.sub(r'^\s*!\[[^\]]*\]\([^)]+\)\s*$', '', text, flags=re.M)
    # Remove plain "Figure:" placeholders, optionally italicized with *...*
        text = re.sub(r'^\s*\*?\[?Figure:[^\]\n]*\]?\*?\s*$', '', text, flags=re.M)
    # Collapse extra blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text


    def assemble_references(self, manuscript: str) -> str:
        cited = set(self._extract_citekeys(manuscript))
        if not cited: return ""
        return self._format_bibliography_apa(sorted(cited))

    def run(self):
        meta = self.generate_meta()
        plan = self.generate_plan()

        chapter_docs: List[str] = []
        prev_titles: List[str] = []
        for ch in tqdm(plan.chapters, desc="Drafting chapters"):
            safe = re.sub(r'[^A-Za-z0-9_-]+','', ch.title.replace(' ','_'))[:120]
            ch_path = self.chapters_dir / f"{ch.number:02d}_{safe}.md"
            if ch_path.exists() and ch_path.stat().st_size > 100:
                chapter_docs.append(ch_path.read_text(encoding="utf-8"))
                prev_titles.append(ch.title)
                continue
            doc = self.draft_chapter(ch, prev_titles)
            ch_path.write_text(doc, encoding="utf-8")
            chapter_docs.append(doc)
            prev_titles.append(ch.title)

        toc_lines = [f"- Chapter {c.number}: {c.title}" for c in plan.chapters]
        front = f"""# {meta.title}
## {meta.subtitle}
**Author:** {meta.author}

---

## About this book
{plan.synopsis}

---

## Table of Contents
{chr(10).join(toc_lines)}

---

"""
        manuscript = front + "\\n\\n".join(chapter_docs)
        if self.no_figures:
            manuscript = self._strip_fig_placeholders(manuscript)

        refs_md = self.assemble_references(manuscript)
        if refs_md:
            manuscript += "\\n\\n---\\n\\n# References (APA)\\n\\n" + refs_md + "\\n"

        (self.outdir / "book.md").write_text(manuscript, encoding="utf-8")
        self._log({"stage":"assemble","path": str(self.outdir / "book.md")})
        print("Done. Wrote:", self.outdir / "book.md")

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a full e-book from a single idea (Markdown output).")
    p.add_argument("--idea", type=str, required=True)
    p.add_argument("--genre", type=str, default="non-fiction")
    p.add_argument("--audience", type=str, default="general audience")
    p.add_argument("--reading-level", type=str, default="grade 10")
    p.add_argument("--style", type=str, default="clear, engaging, practical")
    p.add_argument("--min-words", type=int, default=20000)
    p.add_argument("--max-words", type=int, default=30000)
    p.add_argument("--backend", type=str, choices=["openai"], default="openai")
    p.add_argument("--model", type=str, default="gpt-4.1", help="Default model (used if plan/draft not set).")
    p.add_argument("--plan-model", type=str, default="", help="Model for meta/plan JSON (e.g., gpt-4.1 or gpt-4o).")
    p.add_argument("--draft-model", type=str, default="", help="Model for chapter drafting (e.g., gpt-5).")
    p.add_argument("--fallback-model", type=str, default="gpt-4o", help="Secondary model if the draft model returns empty.")
    p.add_argument("--outdir", type=Path, default=Path("./book_out"))
    p.add_argument("--auto-cite", action="store_true")
    p.add_argument("--min-chapters", type=int, default=16, help="Minimum chapters required in the plan.")
    p.add_argument("--max-chapters", type=int, default=22, help="Maximum chapters allowed in the plan.")
    p.add_argument("--draft-retries", type=int, default=2, help="Whole-chapter drafting attempts before fallback.")
    p.add_argument("--no-section-fallback", action="store_true", help="Disable per-section drafting fallback.")
    p.add_argument("--no-figures", action="store_true", help="Do not include figure/table placeholders or images.")
    return p.parse_args()

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.backend != "openai":
        raise RuntimeError("Only 'openai' backend is included in this compact version.")
    if openai is None:
        raise RuntimeError("Install openai: pip install 'openai==1.*'")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY env var.")

    llm = OpenAIClient(model=args.model)
    plan_model = args.plan_model or args.model
    draft_model = args.draft_model or args.model

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
        plan_model=plan_model,
        draft_model=draft_model,
        auto_cite=args.auto_cite,
        min_chapters=args.min_chapters,
        max_chapters=args.max_chapters,
        fallback_model=args.fallback_model,
        draft_retries=args.draft_retries,
        section_fallback=not args.no_section_fallback,
        no_figures=args.no_figures,
    )
    builder.run()

if __name__ == "__main__":
    main()
