"""
This script was generated with Claude Sonnet 5.

Check for custom_id mismatches between a batch input file (requests) and
a batch output file (responses).

Verifies:
  - Every custom_id in the input file appears exactly once in the output file.
  - Every custom_id in the output file appears exactly once in the input file.
  - No duplicate custom_ids within either file.
  - (Optional) Whether the output file's line order matches the input file's
    line order for custom_id -- informational only, since the Batch API does
    not guarantee this and code should never rely on it.

Usage:
    python check_custom_id_mismatch.py path/to/batch.jsonl path/to/outputs.jsonl
    python check_custom_id_mismatch.py path/to/batch.jsonl path/to/outputs.jsonl --strict

Exit codes:
    0 - no mismatches found
    1 - mismatches found (missing / extra / duplicate custom_ids)
    2 - bad input (file not found, malformed json, missing custom_id field)
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_custom_ids(path: Path, label: str) -> list[str]:
    """Read a .jsonl file and return the list of custom_id values, in file order."""
    if not path.is_file():
        logger.error(f"{label} file does not exist: {path}")
        sys.exit(2)

    custom_ids = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"{label} file {path}: malformed JSON on line {line_num}: {e}")
                sys.exit(2)

            if "custom_id" not in record:
                logger.error(f"{label} file {path}: missing 'custom_id' field on line {line_num}")
                sys.exit(2)

            custom_ids.append(record["custom_id"])

    return custom_ids


def check_duplicates(custom_ids: list[str], label: str) -> bool:
    """Log and return True if duplicate custom_ids are found in a single file."""
    counts = Counter(custom_ids)
    duplicates = {cid: n for cid, n in counts.items() if n > 1}
    if duplicates:
        logger.error(f"{label} file has {len(duplicates)} duplicate custom_id(s):")
        for cid, n in list(duplicates.items())[:20]:
            logger.error(f"  {cid!r} appears {n} times")
        if len(duplicates) > 20:
            logger.error(f"  ... and {len(duplicates) - 20} more")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch_file", type=Path, help="Path to the batch input .jsonl (requests)")
    parser.add_argument("outputs_file", type=Path, help="Path to the batch output .jsonl (responses)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail (exit 1) if output file order differs from input file order. "
             "Off by default since the Batch API does not guarantee matching order.",
    )
    args = parser.parse_args()

    input_ids = load_custom_ids(args.batch_file, "Input (batch)")
    output_ids = load_custom_ids(args.outputs_file, "Output")

    logger.info(f"Input file:  {args.batch_file} ({len(input_ids)} requests)")
    logger.info(f"Output file: {args.outputs_file} ({len(output_ids)} responses)")

    had_error = False

    # Duplicates within each file
    had_error |= check_duplicates(input_ids, "Input")
    had_error |= check_duplicates(output_ids, "Output")

    input_set = set(input_ids)
    output_set = set(output_ids)

    # Requests with no matching response
    missing_outputs = input_set - output_set
    if missing_outputs:
        had_error = True
        logger.error(f"{len(missing_outputs)} custom_id(s) in the input file have no matching response:")
        for cid in list(missing_outputs)[:20]:
            logger.error(f"  {cid!r}")
        if len(missing_outputs) > 20:
            logger.error(f"  ... and {len(missing_outputs) - 20} more")

    # Responses with no matching request
    unexpected_outputs = output_set - input_set
    if unexpected_outputs:
        had_error = True
        logger.error(f"{len(unexpected_outputs)} custom_id(s) in the output file have no matching request:")
        for cid in list(unexpected_outputs)[:20]:
            logger.error(f"  {cid!r}")
        if len(unexpected_outputs) > 20:
            logger.error(f"  ... and {len(unexpected_outputs) - 20} more")

    # Informational: does output preserve input order? (Not required, but useful
    # to know -- code should never rely on this being true.)
    if not had_error:
        common_length = min(len(input_ids), len(output_ids))
        order_matches = input_ids[:common_length] == output_ids[:common_length]
        if order_matches:
            logger.info("Output file order matches input file order (not guaranteed by the API; do not rely on this).")
        else:
            logger.warning("Output file order does NOT match input file order. "
                            "This is expected/allowed by the Batch API -- make sure downstream "
                            "code joins on custom_id rather than line position.")
            if args.strict:
                had_error = True

    if had_error:
        logger.error("Mismatch check FAILED.")
        sys.exit(1)

    logger.info(f"Mismatch check PASSED: all {len(input_ids)} custom_ids match 1:1 between input and output.")
    sys.exit(0)


if __name__ == "__main__":
    main()
