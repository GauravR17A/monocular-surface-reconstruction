"""Summarise saved development evidence without promoting or mutating a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from msr.models.rgb_segmenter import CLASS_NAMES

def report(experiment):
    experiment=Path(experiment)
    baseline=json.loads((experiment/"baseline.json").read_text())
    history=[json.loads(line) for line in (experiment/"metrics.jsonl").read_text().splitlines() if line.strip()]
    lines=["# RGB multi-region V3: development results", "", "Classification only. App/height model unchanged. These development scores select checkpoints; they are not final unseen-test results.","",
        "| Epoch | GAMUS macro F1 | OEM pooled macro F1 | OEM region-mean macro F1 | Safety | Benefit |",
        "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in history:
        v=row["validation"]
        region=sum(g["six_class_identification"]["macro_f1"] for g in v["oem"]["by_city"].values())/len(v["oem"]["by_city"])
        lines.append(f"| {row['epoch']} | {v['gamus']['overall']['six_class_identification']['macro_f1']:.2%} | {v['oem']['overall']['six_class_identification']['macro_f1']:.2%} | {region:.2%} | {row['gate']['safety_pass']} | {row['gate']['benefit_pass']} |")
    outcome_path=experiment/"outcome.json"
    outcome=json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    selected=outcome.get("best_safe") or outcome.get("best_development")
    if selected:
        row=next(r for r in history if r["epoch"]==selected["epoch"])
        lines += ["",f"## Selected review checkpoint: epoch {row['epoch']}","",
            "Selection: "+("best safety-passing development checkpoint" if outcome.get("best_safe") else "best unrestricted development checkpoint; NO safety-passing checkpoint available"),"",
            f"Full benefit + safety acceptance: {row['gate']['eligible_development_candidate']}. No automatic promotion.","",
            "| Dataset / class | Frozen V1 F1 | Candidate F1 | Change (percentage points) |", "| --- | ---: | ---: | ---: |"]
        for domain in ("gamus","oem"):
            for cls in CLASS_NAMES:
                a=baseline[domain]["overall"]["six_class_identification"]["per_class"][cls]["f1"]
                b=row["validation"][domain]["overall"]["six_class_identification"]["per_class"][cls]["f1"]
                lines.append(f"| {domain} / {cls} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |")
        lines += ["","Failed safety checks: "+(", ".join(row["gate"]["failed_checks"]) or "none"),"",
            "Next: inspect per-region errors and HighBuild outline coverage. Only after candidate selection should a fresh reserved geography be tested. Neither classification scores nor this experiment establish height accuracy."]
    target=experiment/"DEVELOPMENT_REPORT.md"
    target.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return target

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("experiment",type=Path)
    print(report(parser.parse_args().experiment))
