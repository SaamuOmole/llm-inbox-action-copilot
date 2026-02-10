from __future__ import annotations
import json
from pathlib import Path

RESULTS_DIR = Path("data/results")

COLS = [
    ("model", "Model"),
    ("intent_accuracy", "Intent Acc"),
    ("action_precision", "Action Prec"),
    ("action_recall", "Action Rec"),
    ("action_f1", "Action F1"),
    ("action_accuracy", "Action Acc"),
]

def main():
    files = sorted(RESULTS_DIR.glob("metrics__v4__*.json"))
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]

    # sort by Action F1 desc, then Intent Acc desc
    rows.sort(key=lambda r: (r["action_f1"], r["intent_accuracy"]), reverse=True)

    # Markdown table
    header = "| " + " | ".join(name for _, name in COLS) + " |"
    sep = "| " + " | ".join(["---"] * len(COLS)) + " |"
    lines = [header, sep]

    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k, _ in COLS) + " |")

    md = "\n".join(lines)
    out_md = RESULTS_DIR / "model_comparison_v4.md"
    out_md.write_text(md, encoding="utf-8")

    # CSV
    out_csv = RESULTS_DIR / "model_comparison_v4.csv"
    csv_lines = [",".join(k for k, _ in COLS)]
    for r in rows:
        csv_lines.append(",".join(str(r[k]) for k, _ in COLS))
    out_csv.write_text("\n".join(csv_lines), encoding="utf-8")

    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
