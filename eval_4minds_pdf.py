#!/usr/bin/env python3
"""
eval_4minds_pdf.py

For each row in an eval CSV:
  1. Send the query to the 4minds inference API via WebSocket.
  2. Pass (question, ground-truth answer, 4minds answer) to an OpenAI
     judge model (LLM-as-a-judge).
  3. Write per-row results to a timestamped CSV in results/ and a running
     log file.

Judge prompt is versioned under prompts/judge/ — edit current.txt to
switch versions, or add a new vN.txt and point current.txt at it.
The judge returns a JSON object with true_positives, false_positives,
and false_negatives arrays. Per-row metrics (precision, recall, F1)
are computed from these counts.

Usage:
    python eval_4minds_pdf.py [path/to/Eval_data.csv]

Defaults to eval_questions/Eval_data.csv if no path is given.
"""

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from common import fourmind_client, openai_client, prompt_loader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

FOURMIND_MODEL_ID = int(os.getenv("FOURMIND_MODEL_ID", "0"))
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))

HERE = Path(__file__).parent
DEFAULT_CSV = HERE / "pdf_eval_questions" / "Eval_data.csv"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_DIR = HERE / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "eval_4minds_pdf.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1) from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def judge(
    judge_prompt: str,
    question: str,
    reference: str,
    system_answer: str,
) -> tuple[int, int, int, float, float, float, str, int]:
    """
    Call the OpenAI judge model with the TP/FP/FN rubric.

    Returns:
        tp_count, fp_count, fn_count, precision, recall, f1,
        raw_json_str, cached_tokens
    """
    client = openai_client.get_client()
    model = OPENAI_JUDGE_MODEL

    user_message = (
        f"GROUND TRUTH:\n{reference}\n\n"
        f"ANSWER:\n{system_answer}"
    )

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    if openai_client.supports_temperature(model):
        kwargs["temperature"] = 0

    completion = openai_client.call_with_retry(
        lambda: client.chat.completions.create(**kwargs)
    )
    raw = completion.choices[0].message.content.strip()
    cached = openai_client.cached_tokens(completion)

    parsed = json.loads(raw)
    tp = len(parsed.get("true_positives",  []))
    fp = len(parsed.get("false_positives", []))
    fn = len(parsed.get("false_negatives", []))
    precision, recall, f1 = _prf(tp, fp, fn)

    return tp, fp, fn, precision, recall, f1, raw, cached


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    missing = [v for v in ("FOURMIND_API_KEY", "FOURMIND_MODEL_ID", "OPENAI_API_KEY")
               if not os.getenv(v)]
    if missing:
        log.error("Missing required env vars: %s — check your .env file.", ", ".join(missing))
        sys.exit(1)

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        log.error("Eval CSV not found: %s", csv_path)
        sys.exit(1)

    judge_prompt, prompt_version = prompt_loader.load_prompt("judge")

    results_csv = RESULTS_DIR / f"eval_results_{timestamp}.csv"
    result_fields = [
        "query_id", "query", "reference_answer", "fourmind_answer",
        "tp", "fp", "fn", "precision", "recall", "f1",
        "judge_raw", "cached_tokens",
        "fourmind_tokens", "fourmind_processing_ms",
        "prompt_version", "fourmind_status", "error",
    ]

    log.info("=== Eval session started %s ===", timestamp)
    log.info("Input CSV    : %s", csv_path)
    log.info("Results CSV  : %s", results_csv)
    log.info("Judge model  : %s  (prompt %s)", OPENAI_JUDGE_MODEL, prompt_version)
    log.info("4minds model : %s", FOURMIND_MODEL_ID)

    total = judged = 0
    sum_tp = sum_fp = sum_fn = total_cached = 0

    with (
        open(csv_path, newline="", encoding="utf-8") as in_f,
        open(results_csv, "w", newline="", encoding="utf-8") as out_f,
    ):
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=result_fields)
        writer.writeheader()

        for row in reader:
            total += 1
            query_id  = row.get("query_id", str(total))
            question  = row.get("query",  "").strip()
            reference = row.get("answer", "").strip()

            log.info("[%s] Querying 4minds...", query_id)

            fourmind_answer = ""
            fourmind_tokens = 0
            fourmind_ms     = 0
            fourmind_status = "ok"
            error_msg       = ""
            tp = fp = fn    = 0
            precision = recall = f1 = 0.0
            judge_raw = ""
            cached    = 0

            try:
                result          = fourmind_client.query(question, FOURMIND_MODEL_ID)
                fourmind_answer = result["answer"]
                fourmind_tokens = result.get("total_tokens", 0)
                fourmind_ms     = result.get("processing_time_ms", 0)
            except Exception as exc:
                fourmind_status = "error"
                error_msg = str(exc)
                log.error("[%s] 4minds error: %s", query_id, exc)

            if fourmind_answer:
                try:
                    tp, fp, fn, precision, recall, f1, judge_raw, cached = judge(
                        judge_prompt, question, reference, fourmind_answer
                    )
                    judged += 1
                    sum_tp += tp
                    sum_fp += fp
                    sum_fn += fn
                    total_cached += cached
                    log.info(
                        "[%s] TP=%d FP=%d FN=%d | P=%.2f R=%.2f F1=%.2f | cached=%d",
                        query_id, tp, fp, fn, precision, recall, f1, cached,
                    )
                except Exception as exc:
                    error_msg = str(exc)
                    log.error("[%s] OpenAI judge error: %s", query_id, exc)

            writer.writerow({
                "query_id":            query_id,
                "query":               question,
                "reference_answer":    reference,
                "fourmind_answer":     fourmind_answer,
                "tp":                  tp,
                "fp":                  fp,
                "fn":                  fn,
                "precision":           f"{precision:.4f}",
                "recall":              f"{recall:.4f}",
                "f1":                  f"{f1:.4f}",
                "judge_raw":           judge_raw,
                "cached_tokens":       cached,
                "fourmind_tokens":     fourmind_tokens,
                "fourmind_processing_ms": fourmind_ms,
                "prompt_version":      prompt_version,
                "fourmind_status":     fourmind_status,
                "error":               error_msg,
            })
            out_f.flush()
            time.sleep(REQUEST_DELAY)

    # Aggregate micro-averaged metrics across all rows
    agg_p, agg_r, agg_f1 = _prf(sum_tp, sum_fp, sum_fn)

    log.info("=== Session complete ===")
    log.info("Rows processed     : %d", total)
    log.info("Rows judged        : %d", judged)
    log.info("Aggregate TP/FP/FN : %d / %d / %d", sum_tp, sum_fp, sum_fn)
    log.info("Micro precision    : %.4f", agg_p)
    log.info("Micro recall       : %.4f", agg_r)
    log.info("Micro F1           : %.4f", agg_f1)
    log.info("Total cached tokens: %d", total_cached)
    log.info("Results saved      : %s", results_csv)


if __name__ == "__main__":
    main()
