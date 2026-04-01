#!/usr/bin/env python3
"""
Upload files to a 4minds AI dataset in batches under 95 MB.
Config is read from .env in the same directory.
Results are logged to upload.log.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("FOURMIND_API_KEY")
DATASET_ID = os.getenv("FOURMIND_DATASET_ID")
UPLOAD_URL = "https://api.4minds.ai/api/v1/user/dataset/upload"
MAX_BATCH_BYTES = 95 * 1024 * 1024  # 95 MB

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = Path(__file__).parent / "upload.log"
OVERSIZED_LOG = Path(__file__).parent / "oversized_files.log"

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

# Separate logger that only writes to oversized_files.log
_oversized_handler = logging.FileHandler(OVERSIZED_LOG, encoding="utf-8")
_oversized_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
oversized_log = logging.getLogger("oversized")
oversized_log.setLevel(logging.INFO)
oversized_log.addHandler(_oversized_handler)
oversized_log.propagate = False  # don't double-write to the main log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIME_TYPES = {
    ".jsonl": "application/x-jsonlines",
    ".json": "application/json",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}


def mime_for(path: Path) -> str:
    return MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


def batch_files(files: list[Path]) -> list[list[Path]]:
    """Group files into batches where each batch is under MAX_BATCH_BYTES."""
    batches: list[list[Path]] = []
    current_batch: list[Path] = []
    current_size = 0

    for f in files:
        size = f.stat().st_size
        if size > MAX_BATCH_BYTES:
            mb = size / 1024 / 1024
            log.warning("Skipping %s (%.1f MB) — exceeds 95 MB limit. See oversized_files.log.", f.name, mb)
            oversized_log.info("%.1f MB  %s", mb, f.resolve())
            continue
        if current_size + size > MAX_BATCH_BYTES and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(f)
        current_size += size

    if current_batch:
        batches.append(current_batch)

    return batches


def upload_batch(batch: list[Path], batch_num: int, total_batches: int) -> bool:
    """Upload a single batch. Returns True on success."""
    log.info("--- Batch %d / %d  (%d file(s), %.2f MB) ---",
             batch_num, total_batches,
             len(batch),
             sum(f.stat().st_size for f in batch) / 1024 / 1024)

    file_handles = []
    try:
        files_payload = []
        for f in batch:
            fh = open(f, "rb")
            file_handles.append(fh)
            files_payload.append(("files", (f.name, fh, mime_for(f))))

        response = requests.post(
            UPLOAD_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            files=files_payload,
            data={"dataset_id": DATASET_ID},
            timeout=300,
        )
    finally:
        for fh in file_handles:
            fh.close()

    log.info("HTTP %s", response.status_code)

    if response.status_code == 200:
        result = response.json()
        log.info("Files uploaded : %s", result.get("files_uploaded"))
        log.info("Total in dataset: %s", result.get("total_files"))
        log.info("Raw response   : %s", result)
        return True
    else:
        log.error("Upload failed — status %s", response.status_code)
        log.error("Response body  : %s", response.text)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        log.error("FOURMIND_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)
    if not DATASET_ID:
        log.error("FOURMIND_DATASET_ID is not set. Add it to your .env file.")
        sys.exit(1)

    # Accept a folder or individual files as CLI arguments;
    # default to files in the current directory if none provided.
    targets = [Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else []

    if not targets:
        log.error("Usage: python upload.py <file_or_folder> [file_or_folder ...]")
        sys.exit(1)

    # Expand folders to their direct children files
    all_files: list[Path] = []
    for t in targets:
        if t.is_dir():
            all_files.extend(f for f in sorted(t.iterdir()) if f.is_file())
        elif t.is_file():
            all_files.append(t)
        else:
            log.warning("Path not found, skipping: %s", t)

    if not all_files:
        log.error("No files found to upload.")
        sys.exit(1)

    log.info("=== Upload session started  %s ===", datetime.now().isoformat(timespec="seconds"))
    log.info("Dataset ID : %s", DATASET_ID)
    log.info("Files found: %d", len(all_files))

    batches = batch_files(all_files)
    log.info("Batches    : %d", len(batches))

    success_count = 0
    for i, batch in enumerate(batches, start=1):
        ok = upload_batch(batch, i, len(batches))
        if ok:
            success_count += 1

    log.info("=== Session complete: %d / %d batches succeeded ===",
             success_count, len(batches))

    if success_count < len(batches):
        sys.exit(1)


if __name__ == "__main__":
    main()
