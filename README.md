# AI e-Book Generator

Create a full-length e-book from a single idea.
This Generator plans the book, drafts each chapter, **auto-cites** scholarly sources, and assembles a clean Markdown manuscript you can convert to EPUB/DOCX/PDF.


* **APA-style References** chapter (built automatically from in-text citations)
* **`--no-figures`** flag to suppress figure/table placeholders and image lines
* Hardened JSON planning + retries/fallbacks so runs complete more reliably
* Separate **converter** (`md_to_book.py`) for high-quality EPUB/DOCX/PDF

---

## Features

* **Outline-first workflow** → metadata → synopsis → parts → chapters → sections
* **Self-citing (`--auto-cite`)**

  * Searches **Crossref** per chapter (no fabricated citations)
  * In-text citations like `[@key]` are restricted to known keys only
  * Appends **References (APA)** with just the *cited* items
* **Controls for length & structure**

  * Set total word range; target chapter count (defaults 16–22)
* **Robust drafting**

  * Multiple attempts per chapter; optional section-by-section fallback
  * Optional **fallback model** if the main drafting model returns weak output
* **Outputs**

  * Always: `book.md`
  * Optional (via converter): `book.epub`, `book.docx`, `book.pdf`

---

## Requirements

* **Python 3.9+**
* **OpenAI** account and API key (`OPENAI_API_KEY`)
* Internet access (for model + Crossref when `--auto-cite` is used)

### Install

```bash
# Core
python -m pip install 'openai==1.*' 'pydantic==2.*' 'tenacity==8.*' 'tqdm==4.*' 'requests==2.*'

brew install --cask mactex-no-gui
# then ensure TeX binaries are on PATH:
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc && source ~/.zshrc
pdflatex --version   # should print a version


# (Optional) converters for EPUB/DOCX/PDF
python -m pip install 'requests==2.*' 'markdown==3.*' 'beautifulsoup4==4.*' \
                      'weasyprint==61.*' 'reportlab==4.*'
# Make sure pandoc + a TeX engine (XeLaTeX) are installed for best PDFs.

```

> **zsh tip:** quote pins like `'openai==1.*'` to avoid globbing errors.

### Configure your key

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Quick Start

Generate a ~20–30k-word textbook with APA references and **no figure placeholders**:

```bash
python ebook_generator.py \
  --idea "A microbiologist uses AI to decode the hidden ecology of dairy farm microbes" \
  --genre "textbook" \
  --audience "upper-undergrad and graduate students" \
  --reading-level "university" \
  --style "scholarly, rigorous, precise" \
  --min-words 20000 --max-words 30000 \
  --plan-model gpt-4.1 \
  --draft-model gpt-5 \
  --fallback-model gpt-4o \
  --auto-cite \
  --no-figures \
  --outdir ./book_out
```

Convert the Markdown to EPUB/DOCX/PDF:

```bash
python md_to_book.py \
  --input ./book_out/book.md \
  --outdir ./book_out \
  --all \
  --title "Your Book Title" \
  --author "Your Name"
```

---

## How Auto-Cite (APA) Works

* With `--auto-cite`, each chapter triggers a **Crossref** search using the chapter title + key points.
* The script builds a small, deduplicated bibliography (by DOI) and exposes the **keys** to the model.
* The model is instructed to cite **only** known keys (never fabricate).
* After drafting, the manuscript is scanned for `[@keys]` and a **References (APA)** chapter is generated, including:

  * Authors (with initials), year, title, journal/book/publisher, volume/issue/pages when present, and DOI/URL.
  * Missing metadata is **omitted** (never invented).

---

## Options & Flags

**Core generation**

* `--idea` *(str, required)*: one-sentence core idea
* `--genre`, `--audience`, `--reading-level`, `--style` *(str)*
* `--min-words`, `--max-words` *(int)*: target total words

**Models**

* `--model` *(str)*: default model if you don’t split plan/draft
* `--plan-model` *(str)*: model for strict JSON planning (e.g. `gpt-4.1`, `gpt-4o`)
* `--draft-model` *(str)*: model for chapter drafting (e.g. `gpt-5`)
* `--fallback-model` *(str)*: backup if drafts are short/empty (default `gpt-4o`)

**Structure & resilience**

* `--min-chapters` *(int, default 16)* / `--max-chapters` *(int, default 22)*
* `--draft-retries` *(int, default 2)*: whole-chapter retries
* `--no-section-fallback`: disable section-by-section fallback

**Citations & figures**

* `--auto-cite`: enable Crossref-based self-citation
* `--no-figures`: prevents figure/table placeholders and Markdown image lines

**Paths**

* `--outdir` *(path, default `./book_out`)*

---

## Typical Output

```
book_out/
├─ plan/
│  ├─ meta.json              # generated metadata
│  └─ plan.json              # chapter plan used for drafting
├─ chapters/
│  ├─ 01_...md
│  ├─ 02_...md
│  └─ ...
├─ book.md                   # assembled manuscript
└─ checkpoints.jsonl         # logs (for debugging)
```

---

## Page Count vs. Word Count

* **Rule of thumb**: 200 pages ≈ **50–60k** words; 400 pages ≈ **105–125k** words.
* Use `--min-words` / `--max-words` and chapter count bounds to hit your target.

  * Example: 16–22 chapters × 2,000–3,200 words ≈ 20–70k total.

---

## Reading Levels (examples)

`--reading-level` is free-form:

* *Elementary*, *Middle school*, *High school (grade 9–12)*
* *College / Undergraduate*, *University (upper-undergrad & graduate)*
* *Professional / Technical*, *Executive / Policy*

Examples:

```bash
--reading-level "grade 10"
--reading-level "college"
--reading-level "university"
--reading-level "professional/technical"
```

Pair with tone controls via `--style` (e.g., `"scholarly, rigorous, precise"`).

---

## Setting the Author

The generator proposes metadata (including `author`). To **force** the author name:

1. After planning, edit `plan/meta.json` and set `"author": "Your Name"`; or
2. After generation, edit the **header** of `book.md` (the line starting with `**Author:**`).

*(Keeping it editable avoids brittle prompt plumbing and gives you full control.)*

---

## Troubleshooting

* **401 (Unauthorized)**
  Set your key: `export OPENAI_API_KEY="sk-..."`

* **429 (Quota/Rate limits)**
  Add billing or switch to smaller word targets; use `--fallback-model gpt-4o`.

* **“Plan invalid (too few chapters)” or JSON errors**
  The script will try to repair automatically. If it still fails, check `plan/plan_raw*.txt` and rerun.

* **Empty/short chapter**
  The script retries; if still short it switches to the fallback model and (optionally) section-by-section drafting.

* **PDF layout not perfect**
  Use the converter with Pandoc (`--all`) for best quality. ReportLab output is a minimal fallback.

---

## Security

* Never commit API keys.
* Rotate/revoke if exposed.

---

## License

MIT

---

### Handy Commands

**Generate (no figures, with APA references):**

```bash
python ebook_generator.py \
  --idea "..." --genre "textbook" \
  --audience "upper-undergrad and graduate students" \
  --reading-level "university" \
  --style "scholarly, rigorous, precise" \
  --min-words 20000 --max-words 30000 \
  --plan-model gpt-4.1 --draft-model gpt-5 \
  --fallback-model gpt-4o \
  --auto-cite --no-figures \
  --outdir ./book_out
```

**Convert to EPUB/DOCX/PDF:**

```bash
python md_to_book.py --input ./book_out/book.md --outdir ./book_out --all \
  --title "Your Book Title" --author "Your Name"
```
