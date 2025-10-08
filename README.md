# AI e‑Book Generator

Generate a full‑length e‑book from a **single idea**. This tool plans the book, drafts each chapter with continuity prompts, and assembles the manuscript (Markdown) with optional exports (EPUB/DOCX/PDF). It supports **24–36 chapters** for ~400 pages (~110k words) or fewer chapters for ~200 pages (~50–60k words).

> Works with OpenAI, Anthropic, or local models via Ollama.

---

## Features

* **Outline‑first workflow** → metadata → synopsis → parts → chapters → sections → scenes.
* **Citations‑ready** (optional): supply BibTeX/JSON; model cites only known keys (e.g., `[@key]`); automatic **References** chapter.
* **Resumable**: checkpointed JSONL; safe to stop & resume.
* **Pluggable backends**: OpenAI, Anthropic, or Ollama/local via a common interface.
* **Style controls**: genre, audience, reading level, tone.
* **Multiple outputs**: Markdown (always), EPUB/DOCX/PDF (optional deps).

---

## Requirements

* Python 3.9+
* One LLM backend (choose at least one):

  * OpenAI (`OPENAI_API_KEY`)
  * Anthropic (`ANTHROPIC_API_KEY`)
  * Ollama (local server at `http://localhost:11434`)

### Install

```bash
# Core
pip install pydantic==2.* rich==13.* tqdm==4.* tenacity==8.* python-dateutil==2.* requests==2.*

# Choose at least one backend
pip install openai==1.*            # OpenAI
# pip install anthropic==0.*       # Anthropic
# (Ollama uses requests over HTTP; pull a model with `ollama pull llama3.1`)

# Optional exports
pip install ebooklib==0.* markdown==3.* python-docx==1.* reportlab==4.*

# Optional bibliography
pip install bibtexparser==1.* beautifulsoup4==4.*
```

### Configure keys

```bash
export OPENAI_API_KEY=sk-...        # if using OpenAI
# or
export ANTHROPIC_API_KEY=...        # if using Anthropic
```

---

## Quick start

**400‑page target (~110k words)**

```bash
python ebook_generator.py \
  --idea "A microbiologist uses AI to decode the hidden ecology of dairy farm microbes" \
  --genre "popular science" \
  --audience "curious adults" \
  --reading-level "grade 10" \
  --style "engaging, clear, lightly humorous" \
  --min-words 105000 --max-words 125000 \
  --backend openai --model gpt-4.1 \
  --outdir ./book_out
```

**200‑page target (~50–60k words)**

```bash
python ebook_generator.py \
  --idea "Your idea here" \
  --min-words 50000 --max-words 60000 \
  --genre "textbook" --audience "upper-undergrad/grad" \
  --style "scholarly, clear" \
  --backend openai --model gpt-4.1 \
  --outdir ./book_out_200p
```

**Resume a run** (safe to interrupt and continue later):

```bash
python ebook_generator.py --resume --outdir ./book_out
```

---

## Citations & References (optional)

Provide a bibliography as **BibTeX** or **JSON**. The model is instructed to cite only keys you supply (e.g., `[@smith2021]`). The script then scans used keys and appends a **References** chapter.

**BibTeX example**

```bibtex
@article{smith2021,
  author  = {Smith, J. and Lee, A.},
  title   = {Microbiomes in Dairy},
  journal = {AgriBio},
  year    = {2021},
  doi     = {10.1234/abcd},
  url     = {https://doi.org/10.1234/abcd}
}
```

Run with:

```bash
python ebook_generator.py --idea "Textbook idea" --genre textbook --bibtex sources.bib --outdir ./book_with_refs
```

**JSON example** (`citations.json`)

```json
{
  "doe2019": {
    "author": "Doe, R.",
    "year": "2019",
    "title": "Biofilms 101",
    "publisher": "SciPress"
  }
}
```

Run with:

```bash
python ebook_generator.py --idea "Textbook idea" --citations-json citations.json --outdir ./book_with_refs
```

> If a key is cited but missing in your bibliography, the References chapter marks it as **MISSING** so you can fix it.

---

## Outputs

```
book_out/
├─ plan/               # metadata + plan JSON/MD
├─ chapters/           # per‑chapter Markdown drafts
├─ book.md             # concatenated manuscript
├─ book.epub           # if ebooklib/markdown installed
├─ book.docx           # if python-docx + bs4 installed
├─ book.pdf            # if reportlab installed (basic layout)
└─ checkpoints.jsonl   # resumable logs
```

---

## Tuning length & structure

* **Page count ≈ word count.** 200 pages ≈ 50–60k words; 400 pages ≈ 105–125k words.
* For tighter books, you can also reduce chapter count and per‑chapter target in `PLAN_PROMPT`:

  * Change “24–36” to “16–22” chapters.
  * Set chapter `target_words` to **2,500–3,000**.

---

## Architecture

1. **Metadata** → title, subtitle, audience, voice.
2. **Plan** → synopsis, parts, chapters (titles, purposes, sections, target words).
3. **Draft** → chapter‑by‑chapter with continuity hints and structure.
4. **Assemble** → combine Markdown; optionally build EPUB/DOCX/PDF.
5. **References** → scan `[@key]` citations and append formatted bibliography (if provided).

---

## CLI reference

Run `python ebook_generator.py -h` for full help. Key options:

* `--idea` (str): single‑sentence core idea.
* `--genre` (str), `--audience` (str), `--reading-level` (str), `--style` (str).
* `--min-words` / `--max-words`: total manuscript target.
* `--backend` (`openai` | `anthropic` | `ollama`) and `--model`.
* `--outdir` (path), `--resume`.
* `--bibtex` (path to .bib) and/or `--citations-json` (path to JSON).
* `--no-exports`: skip EPUB/DOCX/PDF builds (Markdown only).

---

## Tips & costs

* 110k words is **expensive** on cloud LLMs. Dry‑run at **20k** first, then scale.
* Local models (Ollama) can save costs, but you may need to iterate prompts/models for quality.
* Use `--resume` for long runs.

---

## Troubleshooting

* **`openai`/`anthropic` not installed** → install the chosen backend package.
* **Missing API key** → set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
* **EPUB/DOCX/PDF not created** → ensure optional deps are installed; otherwise you will always get `book.md`.
* **BibTeX not parsed** → install `bibtexparser`; check that `@entry{key, ...}` keys match `[@key]` in text.
* **Very short/long chapters** → adjust `--temperature` (0.6–0.8) or edit `PLAN_PROMPT` targets.

---

## Ethics & safety

* The model is instructed to avoid harmful or fabricated content and to **not invent citations**. Review the manuscript for accuracy and attribution, especially for textbooks.

---


