#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_code_figures.py
----------------------
Execute embedded Python/R plotting code blocks in a Markdown manuscript and
replace them with generated images.

Code block syntax in book.md:
```pyplot
# your Python plotting code (matplotlib)
# optionally call plt.savefig(...) yourself
```
```rplot
# your R plotting code (requires Rscript in PATH; ggplot2 optional)
# optionally call ggsave(...) or png()/dev.off() yourself
```

Usage:
  python render_code_figures.py --input ./book_out/book.md --outdir ./book_out --keep-code

Notes:
- For Python blocks, matplotlib is required. The script forces the 'Agg' backend.
- For R blocks, you need R installed. If code does not save explicitly,
  the runner will attempt a generic save.
"""
import argparse, os, re, subprocess, sys, textwrap, tempfile, shutil
from pathlib import Path

PY_FENCE = "pyplot"
R_FENCE = "rplot"

def run_python_plot(code: str, output_path: Path) -> str:
    runner = f"""
import os, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_PATH = os.environ.get('OUTPUT_PATH', r'{output_path.as_posix()}')
# --- user code start ---
{code}
# --- user code end ---

# If user didn't save a file, try to save current figure(s).
import os
if not os.path.exists(OUTPUT_PATH):
    try:
        plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    except Exception as e:
        raise SystemExit(f"[pyplot] Failed to save figure automatically: {{e}}")
plt.close('all')
print(OUTPUT_PATH)
"""
    with tempfile.NamedTemporaryFile('w', suffix=".py", delete=False) as tf:
        tf.write(runner)
        tf.flush()
        tmp_path = tf.name
    env = os.environ.copy()
    env["OUTPUT_PATH"] = str(output_path)
    try:
        p = subprocess.run([sys.executable, tmp_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        if p.returncode != 0:
            raise RuntimeError(p.stdout)
        return p.stdout.strip()
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

def run_r_plot(code: str, output_path: Path) -> str:
    # Write an R script that executes the user's code and saves a PNG if they didn't
    r_runner = f"""
args <- commandArgs(trailingOnly=TRUE)
output <- Sys.getenv('OUTPUT_PATH')
if (output == '') {{ output <- args[1] }}
# --- user code start ---
{code}
# --- user code end ---
if (!file.exists(output)) {{
  try({{
    grDevices::png(output, width=1200, height=800, res=150)
    # attempt to draw something basic if possible
    try({{print(get('last_plot', envir=asNamespace('ggplot2'))())}}, silent=TRUE)
    try({{plot.new(); text(0.5, 0.5, 'No explicit save; auto placeholder.', cex=1.2)}}, silent=TRUE)
    grDevices::dev.off()
  }}, silent=TRUE)
}}
cat(output)
"""
    with tempfile.NamedTemporaryFile('w', suffix=".R", delete=False) as tf:
        tf.write(r_runner)
        tf.flush()
        tmp_path = tf.name
    env = os.environ.copy()
    env["OUTPUT_PATH"] = str(output_path)
    try:
        p = subprocess.run(["Rscript", tmp_path, str(output_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        if p.returncode != 0:
            raise RuntimeError(p.stdout)
        return p.stdout.strip()
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

def process_markdown(md_in: Path, outdir: Path, keep_code: bool) -> Path:
    text = md_in.read_text(encoding="utf-8")
    images_dir = outdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Regex to find fenced code blocks with language pyplot or rplot
    pattern = re.compile(r"```(?P<lang>pyplot|rplot)\s*\n(?P<code>.*?)(?:```)", re.S)
    idx = 0
    generated = 0
    new_parts = []
    last_end = 0

    for m in pattern.finditer(text):
        lang = m.group("lang").strip().lower()
        code_block = m.group("code")
        start, end = m.span()

        # Keep text before block
        new_parts.append(text[last_end:start])

        fig_name = f"fig_{generated+1:03d}.png"
        fig_path = images_dir / fig_name

        try:
            if lang == PY_FENCE:
                run_python_plot(code_block, fig_path)
            else:
                run_r_plot(code_block, fig_path)
            generated += 1
            img_md = f"![Figure {generated}]({(Path('images')/fig_name).as_posix()})"
            if keep_code:
                replacement = f"{img_md}\n\n```{lang}\n{code_block}\n```"
            else:
                replacement = img_md
            new_parts.append(replacement)
        except Exception as e:
            # If execution fails, keep the original block and add a warning note
            warn = f"\n> **Render warning:** {str(e).strip()}\n"
            new_parts.append(warn + text[start:end])

        last_end = end

    new_parts.append(text[last_end:])
    out_md = outdir / "book.figures.md"
    out_md.write_text("".join(new_parts), encoding="utf-8")
    return out_md

def main():
    ap = argparse.ArgumentParser(description="Execute embedded Python/R plotting code blocks in Markdown and replace them with images.")
    ap.add_argument("--input", type=Path, required=True, help="Path to input book.md")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory (images/ and new markdown will be written here)")
    ap.add_argument("--keep-code", action="store_true", help="Keep the original code blocks under each image")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    out_md = process_markdown(args.input, args.outdir, args.keep_code)
    print(f"Wrote {out_md} and image files under {args.outdir/'images'}")

if __name__ == "__main__":
    main()
