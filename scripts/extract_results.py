"""Extract key results from a just-executed run_pipeline.ipynb into RESULTS.md.

Usage:
    python3 scripts/extract_results.py
        --notebook ../run_pipeline.ipynb
        --out ../RESULTS.md

We pull text outputs from a hand-picked set of cells (config, MLP train,
SAE train, LLM1 + verifier loop, polarity summary, explanation examples,
final summary) and assemble a tidy markdown report.

The intent is to keep a curated "this is what a run looks like" document
in version control while keeping the notebook itself output-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Source-substring → markdown section title for every cell we want to surface.
# Match is on the first ~80 characters of the cell source after stripping
# leading "## ", "from ", etc.
INCLUDED = [
    ("device =", "1. Config + device"),
    ("D = generate_offline_data", "2. POLAR data generation"),
    ("X = build_features", "3. Derived features + outcome labels"),
    ("model, train_res, scaler", "4. MLP blackbox training"),
    ("sae, sae_res = train_sae", "5. SAE training"),
    ("evidence = collect_evidence", "6. Evidence collection"),
    ("client = QwenLocalClient", "7. LLM client setup"),
    ("PASS 1: cold-start naming", "8a. LLM1 — pass 1 (cold start)"),
    ("compute_polarities", "8b. Feature polarity"),
    ("save_pipeline_state", "8c. Snapshot save"),
    ("explain_with_selection", "9. Explanations on test inputs (LLM2 + Judge + K-selection)"),
    ("if results:", "10. Final summary"),
]


def cell_source(c: dict) -> str:
    src = c.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def cell_text_outputs(c: dict) -> str:
    chunks = []
    for o in c.get("outputs", []):
        text = o.get("text") or ""
        if isinstance(text, list):
            text = "".join(text)
        if text:
            chunks.append(text)
        # Some outputs put the body in data["text/plain"]
        d = o.get("data") or {}
        if "text/plain" in d:
            t = d["text/plain"]
            chunks.append("".join(t) if isinstance(t, list) else t)
    return "".join(chunks).rstrip()


def build_markdown(nb: dict, source_path: Path) -> str:
    out = ["# Run results\n\n"]
    out.append(
        f"Generated from `{source_path.name}`. Captures the printed output "
        "from a single executed run, trimmed to the cells most useful as a "
        "reference.\n\n"
    )
    out.append(
        "To reproduce: open the notebook, switch the LLM client cell from "
        "`StubClient` to `QwenLocalClient(...)`, run all.\n\n"
    )

    matched_idxs = set()
    for needle, title in INCLUDED:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            if i in matched_idxs:
                continue
            if needle in cell_source(c):
                text_out = cell_text_outputs(c)
                if not text_out:
                    continue
                out.append(f"## {title}\n\n")
                out.append("```\n")
                out.append(text_out)
                if not text_out.endswith("\n"):
                    out.append("\n")
                out.append("```\n\n")
                matched_idxs.add(i)
                break

    return "".join(out).rstrip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notebook", default="run_pipeline.ipynb",
                   help="path to the executed .ipynb file (default: ./run_pipeline.ipynb)")
    p.add_argument("--out", default="RESULTS.md",
                   help="output markdown path")
    args = p.parse_args()

    nb_path = Path(args.notebook).resolve()
    out_path = Path(args.out).resolve()

    with nb_path.open() as f:
        nb = json.load(f)

    md = build_markdown(nb, nb_path)
    out_path.write_text(md)
    print(f"wrote {len(md)} bytes to {out_path}")


if __name__ == "__main__":
    main()
