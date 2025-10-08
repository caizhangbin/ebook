# AI e‑Book Generator (Self‑Citing Edition)

Generate a full‑length e‑book from a **single idea**. The script plans the outline, drafts each chapter with continuity prompts, and assembles the manuscript to Markdown (plus optional EPUB/DOCX/PDF). It now includes **Auto‑Cite**: per‑chapter literature search (Crossref) with strict in‑text citations (e.g., `[@smith2021]`) and an auto‑compiled **References** chapter — no manual BibTeX required (though you can still supply your own sources).

---

## Features
- **Outline‑first workflow** → metadata → synopsis → parts → chapters → sections.
- **Self‑citing mode (`--auto-cite`)**
  - Searches **Crossref** per chapter for relevant papers (configurable budget).
  - Populates a local bibliography and instructs the model to **cite only known keys**.
  - Appends a **References** chapter containing exactly the cited sources.
- **Resumable**: checkpointed JSONL; safe to stop & resume.
- **Pluggable backends**: OpenAI, Anthropic, or local Ollama.
- **Style controls**: genre, audience, reading level, tone.
- **Multiple outputs**: Markdown (always), optional EPUB/DOCX/PDF (lightweight exporters).

---

## Requirements
- Python 3.9+
- One LLM backend (choose at least one):
  - OpenAI (`OPENAI_API_KEY`)
  - Anthropic (`ANTHROPIC_API_KEY`)
  - Ollama (local, `http://localhost:11434`)

### Install
```bash
# Core deps
python -m pip install 'pydantic==2.*' 'rich==13.*' 'tqdm==4.*' 'tenacity==8.*' 'python-dateutil==2.*' 'requests==2.*'

# Backend (pick one)
python -m pip install 'openai==1.*'             # OpenAI
# python -m pip install 'anthropic==0.*'        # Anthropic
# (Ollama uses HTTP; no extra pip package)

# Optional exports
python -m pip install 'ebooklib==0.*' 'markdown==3.*' 'python-docx==1.*' 'reportlab==4.*' 'beautifulsoup4==4.*'

# Optional: if you plan to pass --bibtex
python -m pip install 'bibtexparser==1.*'
```
> zsh users: quote the `==1.*` style pins (as shown) to avoid globbing errors.

### Configure keys
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
# Anthropic (if used)
# export ANTHROPIC_API_KEY="..."
```

---

## Quick start
### A. Quality‑focused (OpenAI, 200 pages ~50–60k words)
```bash
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
```

### B. Budget/Local (Ollama, smoke test ~20–30k words)
```bash
ollama pull llama3.1
python ebook_generator.py \
  --idea "Your idea here" \
  --genre "popular science" --audience "curious adults" \
  --reading-level "grade 10" --style "engaging, clear" \
  --min-words 20000 --max-words 30000 \
  --backend ollama --model llama3.1 \
  --auto-cite \
  --outdir ./book_out_local
```

### C. Resume a run
```bash
python ebook_generator.py --resume --outdir ./book_out
```

---

## Self‑Citing: how it works
- When you pass `--auto-cite`, the generator queries **Crossref** using your chapter title + key points.
- It builds a small per‑chapter bibliography (deduped by DOI) and exposes the keys to the model.
- The prompt **forbids fabricated citations** and allows citing only known keys (e.g., `[@garcia2020_t7ss]`).
- After drafting, the script scans the manuscript for used keys and creates a **References** chapter.

### Add your own sources (optional)
- Supply `--bibtex sources.bib` and/or `--citations-json citations.json`. These are merged with auto‑fetched items.
- The final References chapter includes only **actually cited keys**.

---

## Outputs
```
book_out/
├─ plan/               # metadata + plan JSON/MD
├─ chapters/           # per‑chapter Markdown drafts
├─ auto_refs_chXX.bib  # (when --auto-cite) per-chapter Crossref picks
├─ book.md             # concatenated manuscript
├─ book.epub           # if ebooklib/markdown installed
├─ book.docx           # if python-docx + bs4 installed
├─ book.pdf            # if reportlab installed (basic layout)
└─ checkpoints.jsonl   # resumable logs
```

---

## Tuning length & structure
- **Page count ≈ word count.** 200 pages ≈ 50–60k words; 400 pages ≈ 105–125k words.
- For tighter books, reduce chapter count or per‑chapter targets in the generated plan:
  - 16–22 chapters × 2,500–3,000 words → ~50–60k words (≈200 pages).
  - 24–36 chapters × 3,000–3,500 words → ~105–125k words (≈400 pages).

## Reading level presets & examples
`--reading-level` is free‑form. Common choices:
- **Elementary**, **Middle school**, **High school (grade 9–12)**
- **College / Undergraduate**
- **University (upper‑undergrad & graduate)**
- **Professional / Technical**
- **Executive / Policy**

Examples:
```bash
--reading-level "grade 10"
--reading-level "college"
--reading-level "university"
--reading-level "professional/technical"
```
Pair with `--style` (e.g., `--style "scholarly, rigorous, precise"`).

---

## CLI reference
Run `python ebook_generator.py -h` for full help. Key options:
- `--idea` (str): single‑sentence core idea.
- `--genre` (str), `--audience` (str), `--reading-level` (str), `--style` (str).
- `--min-words` / `--max-words`: total manuscript target.
- `--backend` (`openai` | `anthropic` | `ollama` | `local`) and `--model`.
- `--outdir` (path), `--resume`, `--no-exports`.
- **Citations**: `--auto-cite` (Crossref), `--bibtex` (path to .bib), `--citations-json` (path to JSON).

---

## Cost & performance tips
- Start with a **smoke test**: `--min-words 20000 --max-words 30000`.
- Use **OpenAI** for best coherence; use **Ollama** to iterate cheaply and privately.
- Always run with `--resume` so retries don’t redo finished chapters.
- Lower `--temperature` (0.6–0.7) for planning; ~0.7 for drafting for a bit more voice.

---

## Troubleshooting
- **401 / missing API key**: ensure `export OPENAI_API_KEY="sk-..."` (or `ANTHROPIC_API_KEY`).
- **429 / quota**: add billing or switch to a cheaper model / Ollama; try smaller word targets.
- **zsh globbing errors**: quote version pins `'openai==1.*'`.
- **PDF line break issue**: exporter is basic; for print‑ready PDFs, consider Pandoc/LaTeX.
- **No References chapter**: ensure you used `--auto-cite` (or passed your own `--bibtex/--citations-json`) and that chapters actually contain `[@keys]`.

---

## Security
- **Never commit API keys**. Use environment variables or a local secrets manager.
- Rotate/revoke keys if exposed.

---

## License
MIT (or your preferred license).
