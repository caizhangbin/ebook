# AI e-Book Generator

Create a full e-book from a single idea. 
---



## Install

```bash
# Python 3.9+ recommended
python -m pip install "openai==1.*" "requests==2.*"

# Optional (for the converter & exports)
python -m pip install "pypandoc-binary==1.*"

# If you want XeLaTeX PDFs on macOS:
# brew install --cask mactex-no-gui
# echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc && source ~/.zshrc
# pdflatex --version   # sanity check
```

> zsh users: always quote pins like `"openai==1.*"` to avoid globbing.

---

## Quick start

### 1) Generate the manuscript (Markdown)

**OpenAI**

```bash
export OPENAI_API_KEY="sk-..."
python ebook_generator_full.py \
  --idea "A microbiologist uses AI to decode the hidden ecology of dairy farm microbes" \
  --genre "textbook" \
  --audience "upper-undergrad and graduate students" \
  --reading-level "university" \
  --style "scholarly, rigorous, precise" \
  --min-words 20000 --max-words 30000 \
  --backend openai --model gpt-4.1 \
  --auto-cite \
  --no-pictures \
  --outdir ./book_out
```

**Local (Ollama)**

```bash
# ollama pull llama3.1
python ebook_generator.py \
  --idea "Intro to practical Bayesian stats" \
  --genre "popular science" \
  --reading-level "college" \
  --min-words 10000 --max-words 15000 \
  --backend ollama --model llama3.1 \
  --outdir ./book_out_local
```

> Defaults: if you omit `--audience`, **"general audience"** is used.

You’ll get:

```
book_out/
├─ plan/               # meta/plan and raw model outputs
├─ chapters/           # per-chapter drafts (Markdown)
└─ book.md             # full manuscript
```

---

### 2) Make the book polished (optional but recommended)

```bash
python md_to_book.py \
  --input ./book_out/book.md \
  --outdir ./book_out \
  --title "Statistical Methods in Microbiology" \
  --author "Your name" \
  --all
```

What this does:

* Adds **clickable TOC** and **stable anchors** (`{#chapter-1-…}`) to `book.md`
* Converts `[@keys]` → “(Author, Year)”, writes **# References (APA)** and `citations.bib`
* Exports **EPUB/DOCX/PDF**

**To remove all citations in export:** add `--strip-citations`.

---

## Common workflows

### Edit after generation

* Open `book.md`, add/rename chapters, insert pictures:

  ```md
  ![Schematic of habitats](images/habitats.png){width=70%}
  # New Chapter Title {#new-chapter}
  ```
* Re-run the converter:

  ```bash
  python md_to_book.py --input ./book_out/book.md --outdir ./book_out --all
  ```

  The TOC and pagination/bookmarks update automatically.

### Skip images entirely

* Run the generator with `--no-pictures`. This removes:

  * image markdown: `![alt](path)`
  * figure/table placeholders like `*[Figure: ...]*`

### Citations

* **`--auto-cite`**: Chapter prompts allow citing **only** known keys fetched from Crossref for that chapter; no fabricated refs.
* **`--no-citations`**: Generator strips all citations from the final `book.md`.
* **Converter**: Resolves remaining `[@keys]` into APA inline `(Author, Year)` and compiles a **References (APA)** chapter.

---

## Flags (generator)

```txt
--idea               (str)  single-sentence core idea
--genre              (str)
--audience           (str)  default: "general audience"
--reading-level      (str)  e.g., "grade 10", "college", "university", "professional"
--style              (str)  voice/tone hints
--min-words          (int)  total target (book)
--max-words          (int)
--backend            (openai | ollama | local)
--model              (str)  e.g., gpt-4.1, llama3.1
--outdir             (path) output directory

--auto-cite                 enable Crossref lookups per chapter
--no-citations              strip all citations from the manuscript
--no-pictures               remove images and figure/table placeholders
```

---

## Tips

* **Length**: ~250–300 words/page (trade paperback).

  * 20–30k words ≈ ~80–120 pages
  * 50–60k words ≈ ~200 pages
* **Stability**: If you keep custom anchors (`{#id}`) on headings, your section links stay stable even if you rename titles later.
* **Retries**: If a chapter returns short, the generator auto-nudges once. You can delete a chapter `.md` file and rerun to regenerate just that chapter.

---

## Troubleshooting

* **401 / missing key**: `export OPENAI_API_KEY="sk-..."` (for OpenAI).
* **429 / quota**: reduce word targets, try a cheaper model, or Ollama.
* **LaTeX/PDF errors**: ensure XeLaTeX is installed (`pdflatex --version`) and rerun `md_to_book_allinone_v3.py`.
* **Weird line breaks**: generator normalizes `\n` artifacts and de-hyphenates splits; the converter also cleans common math markup.
* **Figure text but no images**: add your own images in Markdown; or keep `--no-pictures` to suppress all figure placeholders.

---

## FAQ

**Can I use different models for planning vs drafting?**
This v2 uses one model for simplicity. If you want split models again, say the word and we’ll provide a variant with `--plan-model/--draft-model/--fallback-model`.

**Will TOC and pages update when I edit?**
Yes. Re-run the converter; it rebuilds anchors/TOC and PDF pagination.

**Can I add new chapters and pictures after AI finishes?**
Yes—edit `book.md` freely and re-export.

**What if I don’t want any citations?**
Use the generator with `--no-citations`, or the converter with `--strip-citations`.

---

## License

MIT.
