#!/usr/bin/env python3
import argparse, json, os, re, sys
from pathlib import Path
from datetime import datetime

def load_records(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # Try JSON array first; fall back to NDJSON (one JSON object per line)
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("Top-level JSON is not an object or array.")
    except json.JSONDecodeError:
        # NDJSON
        records = []
        for i, line in enumerate(txt.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i} is not valid JSON: {e}") from e
        return records

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]

def infer_run_name(records, fallback: str):
    # Prefer first record's creation_time if present; else fallback
    if records:
        ct = records[0].get("creation_time")
        if isinstance(ct, (int, float)):
            ts = datetime.fromtimestamp(ct)
            return f"run_{ts.strftime('%Y%m%d_%H%M%S')}"
    return f"run_{safe_name(fallback)}"

def main():
    ap = argparse.ArgumentParser(description="Extract steps into folders with code and metadata.")
    ap.add_argument("--input", type=Path, help="Path to JSON/NDJSON file containing step records.", default="/home/as2637/dojo_debug/aira-dojo/shared/logs/aira-dojo/user_as2637_issue_example/user_as2637_issue_example_seed_42_id_9b/checkpoint/journal.jsonl")
    ap.add_argument("-o", "--out-dir", type=Path, default="/home/as2637/dojo_debug/aira-dojo/shared/logs/aira-dojo/user_as2637_issue_example/user_as2637_issue_example_seed_42_id_9b", help="Base output directory.")
    ap.add_argument("--run-name", type=str, default=None, help="Optional run name (folder under out-dir).")
    ap.add_argument("--code-filename", type=str, default="code.py", help="Filename for code file in each step dir.")
    args = ap.parse_args()

    records = load_records(args.input)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    run_name = args.run_name or infer_run_name(records, args.input.stem)
    run_dir = args.out_dir / safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional: keep an index of steps
    index = []

    for rec in records:
        step = rec.get("step")
        # Ensure step is usable as a folder suffix
        step_str = f"{int(step):04d}" if isinstance(step, int) else safe_name(str(step or "unknown"))
        step_dir = run_dir / f"step_{step_str}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Write raw record for traceability
        (step_dir / "raw.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

        # Split code from metadata
        code_text = rec.get("code") or ""
        meta = dict(rec)
        if "code" in meta:
            del meta["code"]

        # Write code (even if empty, create a file)
        # Add header if empty to make it import-safe
        if code_text.strip() == "":
            code_text = "# (empty) — no code provided for this step\n"
        (step_dir / args.code_filename).write_text(code_text, encoding="utf-8")

        # Pretty metadata
        (step_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

        index.append({
            "step": rec.get("step"),
            "id": rec.get("id"),
            "path": str(step_dir),
            "has_code": bool(rec.get("code")),
        })

    # Write a simple run index
    (run_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} step folders under: {run_dir}")

if __name__ == "__main__":
    main()
