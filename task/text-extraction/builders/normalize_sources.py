"""Normalize extraction sources and their manifest entries in place.

Use one ID-prefixed record per line, replace ordinal IDs with opaque suffixes,
remove duplicate text, and keep manifest references consistent.
Normalization steps are idempotent.
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "task" / "text-extraction"
MANIFEST_PATH = TASK_DIR / "task_manifest_text50.json"
RNG = random.Random(20260728)

# Canonical record line with an ID prefix.
RECORD_MARKER = re.compile(r"^\[([A-Za-z0-9]+-[A-Za-z0-9]+)\]")
SUFFIX_CHARS = "0123456789abcdef"
SUFFIX_LEN = 6
# Match ordinal IDs only; longer all-digit suffixes may already be opaque IDs.
ORDINAL_MARKER = re.compile(r"^\[(REC(\d+))-(\d{1,3})\]")

# Remove record-ID workflow instructions duplicated by the prompts and tool schema.
INSTRUCTION_TAILS = (
    "Include each extracted record ID in the output artifact.",
    "Resolve each record you extract into its record ID.",
)


def strip_instruction_tail(instruction: str) -> str:
    """Keep selection criteria without redundant record-ID workflow instructions."""
    text = instruction.strip()
    for tail in INSTRUCTION_TAILS:
        if text.endswith(tail):
            return text[: -len(tail)].strip()
    return text


def reference_id(reference: object) -> str:
    """Read a reference ID from either a mapping or a bare string."""
    if isinstance(reference, dict):
        return str(reference.get("id", "")).strip()
    return str(reference).strip()


def declared_ids(case: dict) -> set[str]:
    return {reference_id(item) for item in case["reference_items"]} | {
        str(marker).strip() for marker in case["invalid_item_markers"]
    }


def new_id(case_prefix: str, taken: set[str]) -> str:
    while True:
        suffix = "".join(RNG.choice(SUFFIX_CHARS) for _ in range(SUFFIX_LEN))
        candidate = f"{case_prefix}-{suffix}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate


def assign_opaque_ids(case: dict, lines: list[str], taken: set[str]) -> list[str]:
    """Rewrite ordinal IDs in place, in the source and in the manifest together."""
    old_ids = [
        match.group(1) + "-" + match.group(3)
        for line in lines
        if (match := ORDINAL_MARKER.match(line))
    ]
    if not old_ids:
        return lines

    # Require source and manifest IDs to agree before rewriting references.
    declared = declared_ids(case)
    if declared != set(old_ids):
        raise ValueError(
            f"{case['task_id']}: manifest and source disagree on record IDs; "
            f"only in manifest={sorted(declared - set(old_ids))}, "
            f"only in source={sorted(set(old_ids) - declared)}"
        )

    case_prefix = ORDINAL_MARKER.match(
        next(line for line in lines if ORDINAL_MARKER.match(line))
    ).group(1)
    mapping = {old: new_id(case_prefix, taken) for old in old_ids}

    rewritten = []
    for line in lines:
        match = ORDINAL_MARKER.match(line)
        if match:
            old = match.group(1) + "-" + match.group(3)
            line = f"[{mapping[old]}]" + line[match.end() :]
        rewritten.append(line)

    case["reference_items"] = [
        {"id": mapping[reference_id(item)]} for item in case["reference_items"]
    ]
    case["invalid_item_markers"] = [
        mapping[str(marker).strip()] for marker in case["invalid_item_markers"]
    ]
    return rewritten


def drop_duplicate_records(case: dict, records: list[str]) -> list[str]:
    """Keep the first occurrence of each record text and update manifest references."""
    seen: set[str] = set()
    kept: list[str] = []
    dropped: set[str] = set()
    for record in records:
        match = RECORD_MARKER.match(record)
        text = record[match.end() :].strip()
        if text in seen:
            dropped.add(match.group(1))
            continue
        seen.add(text)
        kept.append(record)
    if dropped:
        case["reference_items"] = [
            item for item in case["reference_items"] if reference_id(item) not in dropped
        ]
        case["invalid_item_markers"] = [
            marker
            for marker in case["invalid_item_markers"]
            if str(marker).strip() not in dropped
        ]
    return kept


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    taken: set[str] = set()
    reduced = 0
    deduped = 0

    for case in manifest:
        source_path = ROOT / case["source_path"]
        lines = source_path.read_text(encoding="utf-8").split("\n")

        lines = assign_opaque_ids(case, lines, taken)
        records = [line for line in lines if RECORD_MARKER.match(line)]
        if not records:
            raise ValueError(f"{case['task_id']}: source declares no records")
        if declared_ids(case) != {RECORD_MARKER.match(r).group(1) for r in records}:
            raise ValueError(
                f"{case['task_id']}: manifest and source disagree on record IDs"
            )
        if len(records) + 1 != len(lines):
            reduced += 1

        kept = drop_duplicate_records(case, records)
        if len(kept) != len(records):
            deduped += 1
            print(f"  {case['task_id']}: dropped {len(records) - len(kept)} duplicate")
        source_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

        case["extraction_instruction"] = strip_instruction_tail(
            case["extraction_instruction"]
        )
        # Recompute record counts after deduplication.
        case["reference_item_count"] = len(case["reference_items"])

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{len(manifest)} case(s) normalized, {reduced} reduced to records only, "
        f"{deduped} deduplicated"
    )


if __name__ == "__main__":
    main()
