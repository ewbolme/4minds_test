#!/usr/bin/env python3
"""
prepare_musique.py

Loads the MuSiQue train split and produces:

  musique/chunks/   — one .txt file per paragraph chunk, named {id}_{idx}.txt
  musique/eval.csv  — one row per question with columns:
                        id, question, answer, supporting_context
                      where supporting_context is the concatenation of all
                      is_supporting=True paragraph texts (separated by \n\n).

Usage:
    python prepare_musique.py
"""

import csv
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CHUNKS_DIR = HERE / "musique" / "chunks"
EVAL_CSV   = HERE / "musique" / "eval.csv"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading MuSiQue train split from HuggingFace...")
    df = pd.read_json(
        "hf://datasets/dgslibisey/MuSiQue/musique_ans_v1.0_train.jsonl",
        lines=True,
    )
    print(f"Loaded {len(df):,} rows.")

    eval_rows = []
    chunks_written = 0

    for _, row in df.iterrows():
        record_id  = row["id"]
        question   = row["question"]
        answer     = row["answer"]
        paragraphs = row["paragraphs"]

        supporting_texts = []

        for para in paragraphs:
            idx            = para["idx"]
            text           = para["paragraph_text"]
            is_supporting  = para.get("is_supporting", False)

            # Write individual chunk file
            chunk_file = CHUNKS_DIR / f"{record_id}_{idx}.txt"
            chunk_file.write_text(text, encoding="utf-8")
            chunks_written += 1

            if is_supporting:
                supporting_texts.append(text)

        eval_rows.append({
            "id":                 record_id,
            "question":           question,
            "answer":             answer,
            "supporting_context": "\n\n".join(supporting_texts),
        })

    # Write eval CSV
    with open(EVAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "supporting_context"])
        writer.writeheader()
        writer.writerows(eval_rows)

    print(f"Chunk files written : {chunks_written:,}  →  {CHUNKS_DIR}")
    print(f"Eval CSV written    : {len(eval_rows):,} rows  →  {EVAL_CSV}")


if __name__ == "__main__":
    main()
