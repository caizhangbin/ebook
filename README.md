# AI Book Builder

A Book Builder pipeline to go from a **single idea** → **planned outline** → **drafted chapters** → **clean Markdown manuscript**, then convert to **EPUB/DOCX/PDF**.


* `ebook_generator.py` (planner/drafter)
* `md_to_book.py` (converter with **auto ToC rebuild**)


* `--plan-model gpt-4.1`
* `--draft-model gpt-5` (with fallback to `gpt-4.1`)
* `--fallback-model gpt-4.1`
* `--no-figures` to suppress figure placeholders


---

## 1) Install

```bash
# Core libs
python -m pip install "openai==1.*" "tqdm==4.*" "rich==13.*" "requests==2.*" "pydantic==2.*" "tenacity==8.*"

# Converters
python -m pip install "markdown==3.*" "beautifulsoup4==4.*" "ebooklib==0.*" "python-docx==1.*" "reportlab==4.*"
```

> zsh users: always **quote** version pins (e.g., `"openai==1.*"`) to avoid globbing errors.

### API key

```bash
export OPENAI_API_KEY="sk-..."   # required for OpenAI
```

---

## 2) Generate the Book

Example:

```bash
python /Users/you/ebook_generator.py \
  --idea "your idea" \
  --genre "Technical" \
  --reading-level "High school" \
  --style "rigorous, precise" \
  --min-words 20000 --max-words 30000 \
  --plan-model gpt-4.1 \
  --draft-model gpt-5 \
  --fallback-model gpt-4.1 \
  --author "your name" \
  --no-figures \
  --outdir ./book_stock
```

### What it does

1. Plans: metadata → synopsis → parts → chapters (robust JSON parsing + repair).
2. Drafts chapters with a **tqdm** progress bar.
3. Produces `book.md`.


### Notes on models

* If `gpt-5` isn’t available/allowed for your key (or rejects params), the script **auto-falls back to `gpt-4.1`** and continues drafting.
* For maximum stability, set:

  ```bash
  --plan-model gpt-4.1 --draft-model gpt-4.1 --fallback-model gpt-4.1
  ```

---

## 3) Convert Markdown → EPUB/DOCX/PDF (with ToC Auto-Sync)

```bash
python /Users/you/md_to_book_legacy_like.py \
  --input ./book_stock/book.md \
  --outdir ./book_stock \
  --title "Book Title" \
  --author "your name" \
  --all
```

### What it does

* **Auto-rebuilds the ToC** in your Markdown from `#` headings **by default** every run.

  * To keep your manual ToC, pass `--no-rebuild-toc`.
* **EPUB**: split by H1 headings (`#`), reader-side navigation included.
* **DOCX**: markdown → HTML → python-docx (clean and dependable).
* **PDF**: ReportLab renderer (simple, no LaTeX/WeasyPrint headaches).

### Useful flags

* `--strip-citations` → remove `[@cite_key]` inline citations in the final book.
* `--strip-figures` → remove image lines (e.g., `![alt](url)`) and `*[Figure: ...]*` placeholders.
* `--no-rebuild-toc` → don’t touch your Table of Contents block.

---

## 4) Folder Layout

```
book_stock/
├─ plan/                 # planning artifacts (if generator writes them)
├─ chapters/             # individual chapter drafts (if enabled)
├─ book.md               # concatenated manuscript from generator
├─ book.cleaned.md       # auto-cleaned (and ToC-synced) MD from converter
├─ book.epub             # export
├─ book.docx             # export
└─ book.pdf              # export
```

---

## 5) Style Tuning

* **Reading level** examples: `"High school"`, `"college"`, `"university"`, `"professional/technical"`.
* Pair with tone in `--style`, e.g., `"scholarly, rigorous, precise"` or `"engaging, clear"`.

**Make it less “AI-like”:**

* Keep `--no-figures` if you don’t want those placeholders.
* Post-edit with your own voice; the pipeline can be rerun as many times as you like.
* The converter normalizes weird hyphenation across line breaks and replaces spaced hyphens with em-dashes.

---

## 6) FAQ

**Q: I edited `book.md`. Will the ToC update?**
**A:** Yes. The converter rebuilds the ToC from your current headings by default.

**Q: The script prints “Primary model failed; trying fallback …”**
**A:** That’s expected if `gpt-5` isn’t available/compatible. Drafts continue with `gpt-4.1`.

**Q: My PDF looks too simple.**
**A:** ReportLab is intentionally minimal and dependable. If you want rich typographic PDFs, we can add a Pandoc/LaTeX path later, but it’s more brittle.

**Q: I don’t want citations or figures.**
**A:** Use `--strip-citations` and/or `--strip-figures` with the converter.

---

## 7) Troubleshooting

* **401 / Missing key**: `export OPENAI_API_KEY="sk-..."`
* **429 / Quota**: add billing or switch to `--draft-model gpt-4.1`.
* **Drafting stalls on a chapter**: it will retry once; if still empty, reduce target words or run again.
* **Strange hyphenation (e.g., “Introduc-\ntion”)**: converter fixes this automatically.
* **Formulas look off**: keep math inline as `$ ... $` in the Markdown; ReportLab doesn’t render LaTeX, but will display the math text cleanly.

---

## 8) Example End-to-End

```bash
# 1) Generate manuscript
python ebook_generator.py \
  --idea "how to make stay healthy" \
  --genre "Technical" \
  --reading-level "High school" \
  --style "rigorous, precise" \
  --min-words 20000 --max-words 30000 \
  --plan-model gpt-4.1 \
  --draft-model gpt-5 \
  --fallback-model gpt-4.1 \
  --author "Xavier J" \
  --no-figures \
  --outdir ./book_stock

# 2) Convert to EPUB/DOCX/PDF (ToC auto-sync)
python md_to_book_legacy_like.py \
  --input ./book_stock/book.md \
  --outdir ./book_stock \
  --title "A Rigorous Guide to Body Building" \
  --author "Xavier J" \
  --all
```

---

## 9) License

MIT
