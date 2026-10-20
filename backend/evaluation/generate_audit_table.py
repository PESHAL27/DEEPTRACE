import json
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
manifest_path = os.path.join(backend_dir, "manifests", "external_benchmark_manifest.json")
results_path = os.path.join(backend_dir, "results", "external_benchmark_results.json")
output_path = os.path.join(backend_dir, "results", "sample_level_audit_table.md")

with open(manifest_path, "r", encoding="utf-8") as fm:
    manifest = json.load(fm)

with open(results_path, "r", encoding="utf-8") as fr:
    results = json.load(fr)

hash_map = {s["filename"]: s["sha256"] for s in manifest["samples"]}
records = results["evaluated_records"]

lines = [
    "| # | Sample ID | Filename | SHA-256 (Prefix) | Ground Truth | Predicted | Fake Score | Correct | Margin Status |",
    "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
]

for idx, r in enumerate(records, 1):
    fname = r["filename"]
    sha = hash_map.get(fname, "")[:12] + "..."
    sid = r["sample_id"]
    gt = "**AUTHENTIC**" if r["ground_truth"] == "authentic" else "**MANIPULATED**"
    pred = "`real`" if r["pf_prediction"] == "real" else "`fake`"
    prob = f"{r['pf_fake_probability']:.4f}"
    corr = "✅ TRUE" if r["pf_correct"] else "❌ FALSE"
    status = "⚠️ Low Margin" if r["pf_is_low_margin"] else "Standard"
    lines.append(f"| {idx} | `{sid}` | `{fname}` | `{sha}` | {gt} | {pred} | `{prob}` | {corr} | {status} |")

with open(output_path, "w", encoding="utf-8") as out:
    out.write("\n".join(lines))

print(f"Successfully generated {len(records)} sample rows into {output_path}")
