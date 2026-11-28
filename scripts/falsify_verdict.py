#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the pre-registered decision rules of docs/plans/07 (Exp-Falsify) to four evals.

    python scripts/falsify_verdict.py \
        --b0 b0/eval.json --b16-fp32 b16_k1_fp32/eval.json \
        --b16-lowmem b16_k1/eval.json --mn2 mn2/eval.json \
        [--proof-json proof_24gb.json] --out-json verdict.json --out-md verdict.md

The four checkpoints form a chain of single-variable steps:

    B0 (32x1, fp32 AdamW) -> b16_k1_fp32 (batch)  -> b16_k1 (optimizer) -> mn2 (multi-noise K=2)

Rules, fixed in the plan before any of these evals existed (paired, same seeds):

* Verdict A, accuracy (b16_k1 vs b16_k1_fp32):
    PASS   diff >= -4 pts and McNemar p >= 0.05
    ALARM  diff < -5 pts, or p < 0.05 with b16_k1 lower
    GRAY   anything else (diff in [-5, -4) with p >= 0.05): not decided at this n.
  One reading added here before the data: b16_k1 *significantly better* (p < 0.05, diff > 0)
  counts as PASS, since the rule exists to catch a loss, and it is flagged in the output.
* Verdict A, memory (Step 1 bench, optional input): b16_k1 runs under the 24 GiB allocator cap
  with peak <= 20 GB.
* Verdict B (mn2 vs b16_k1): KEEP multi-noise only if mn2 is higher with McNemar p < 0.05
  (then a second training seed must reproduce it); otherwise CUT. A tie means CUT: b16_k1 is
  faster per step and uses less memory, so it dominates mn2 unless mn2 is better.

"PASS" at 100 paired episodes means "no loss larger than the interval shows", not "equal":
the 95 % interval on a paired difference at n=100 is roughly +-7 points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_eval_stats import compare_evals  # noqa: E402

A_PASS_MIN_DIFF = -0.04
A_ALARM_DIFF = -0.05
ALPHA = 0.05
MEM_MAX_GB = 20.0
GROUPS = ("b0", "b16_fp32", "b16_lowmem", "mn2")
LABELS = {
    "b0": "B0 (32x1, fp32 AdamW)",
    "b16_fp32": "G_b16_fp32 (16x1, fp32 AdamW)",
    "b16_lowmem": "G_b16_lowmem (16x1, LowMemAdamW bf16)",
    "mn2": "G_b16_mn2 (16x2, LowMemAdamW bf16)",
}


def verdict_a_accuracy(cmp: dict[str, Any]) -> dict[str, Any]:
    _require_paired(cmp, "verdict A")
    diff, p = cmp["diff"], cmp["mcnemar_p"]
    note = None
    if diff < A_ALARM_DIFF or (p < ALPHA and diff < 0):
        result = "ALARM"
    elif diff >= A_PASS_MIN_DIFF and (p >= ALPHA or diff > 0):
        result = "PASS"
        if p < ALPHA:
            note = "b16_lowmem is significantly better than b16_fp32 (p < 0.05); counted as PASS"
    else:
        result = "GRAY"
        note = f"diff {100 * diff:+.1f} pts is between the alarm ({100 * A_ALARM_DIFF:+.0f}) and pass ({100 * A_PASS_MIN_DIFF:+.0f}) lines"
    return {"result": result, "diff": diff, "mcnemar_p": p, "diff_ci95": cmp["diff_ci95"], "note": note}


def verdict_a_memory(proof: dict[str, Any] | None) -> dict[str, Any]:
    if proof is None:
        return {"result": "NOT RUN", "note": "no --proof-json given"}
    runs = {r["mode"]: r for r in proof.get("runs", [])}
    rec = runs.get("b16_k1_cap24") or runs.get("b16_k1")
    if rec is None:
        return {"result": "NOT RUN", "note": "no b16_k1 bench run in the proof JSON"}
    capped = rec.get("cuda_mem_limit_gib") is not None
    peak = rec.get("peak_gpu_mem_gb")
    ok = rec.get("status") == "ok" and capped and peak is not None and peak <= MEM_MAX_GB
    return {
        "result": "PASS" if ok else "FAIL",
        "mode": rec["mode"],
        "status": rec.get("status"),
        "peak_gpu_mem_gb": peak,
        "peak_torch_alloc_gb": rec.get("peak_torch_alloc_gb"),
        "cuda_mem_limit_gib": rec.get("cuda_mem_limit_gib"),
        "note": None if capped else "run was not under the 24 GiB cap",
    }


def verdict_b(cmp: dict[str, Any]) -> dict[str, Any]:
    _require_paired(cmp, "verdict B")
    keep = cmp["diff"] > 0 and cmp["mcnemar_p"] < ALPHA
    return {
        "result": "KEEP (preliminary: needs a second training seed)" if keep else "CUT",
        "diff": cmp["diff"],
        "mcnemar_p": cmp["mcnemar_p"],
        "diff_ci95": cmp["diff_ci95"],
    }


def route(a_acc: str, a_mem: str, b: str) -> str:
    if a_acc == "ALARM" or a_mem == "FAIL":
        return "Stop: do not build on LowMemAdamW; check clip_scale and bf16 norms first, or drop it."
    if a_acc == "GRAY":
        return "Undecided: the optimizer is neither cleared nor convicted at this n; more paired episodes needed."
    if b.startswith("KEEP"):
        return "Both hold: reproduce mn2 vs b16_k1 with a second training seed, then the final matrix."
    return "Optimizer holds, multi-noise cut: multi-tensor kernel + clip_scale + fp32 norms + compile, then the final matrix."


def _require_paired(cmp: dict[str, Any], what: str) -> None:
    if not cmp.get("paired"):
        raise SystemExit(f"{what}: evals are not paired ({cmp.get('why_unpaired')}); the rules need paired evals")


def build(evals: dict[str, dict[str, Any]], proof: dict[str, Any] | None) -> dict[str, Any]:
    pairs = {}
    for i, base in enumerate(GROUPS):
        for cand in GROUPS[i + 1 :]:
            pairs[f"{cand}_vs_{base}"] = compare_evals(evals[base], evals[cand], margin=-A_PASS_MIN_DIFF)
    a_acc = verdict_a_accuracy(pairs["b16_lowmem_vs_b16_fp32"])
    a_mem = verdict_a_memory(proof)
    b = verdict_b(pairs["mn2_vs_b16_lowmem"])
    rates = {g: {k: evals[g].get(k) for k in ("mean_success_rate", "successes", "valid_episodes", "ci95")} for g in GROUPS}
    return {
        "rules": {"a_pass_min_diff": A_PASS_MIN_DIFF, "a_alarm_diff": A_ALARM_DIFF, "alpha": ALPHA, "mem_max_gb": MEM_MAX_GB},
        "success_rates": rates,
        "verdict_a_accuracy": a_acc,
        "verdict_a_memory": a_mem,
        "verdict_b_multi_noise": b,
        "batch_effect_b16_fp32_vs_b0": _brief(pairs["b16_fp32_vs_b0"]),
        "route": route(a_acc["result"], a_mem["result"], b["result"]),
        "pairs": pairs,
    }


def _brief(cmp: dict[str, Any]) -> dict[str, Any]:
    return {k: cmp.get(k) for k in ("diff", "diff_ci95", "mcnemar_p", "paired", "pairs")}


def to_markdown(v: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def pts(x: float) -> str:
        return f"{100 * x:+.1f}"

    lines = ["| group | LIBERO-10 success | episodes | 95% CI |", "|---|---:|---:|---|"]
    for g in GROUPS:
        r = v["success_rates"][g]
        ci = r.get("ci95") or [None, None]
        lines.append(
            f"| {LABELS[g]} | {pct(r['mean_success_rate'])} | {r['successes']}/{r['valid_episodes']} "
            f"| [{pct(ci[0])}, {pct(ci[1])}] |"
        )
    lines += [
        "",
        "| paired comparison | diff (pts) | 95% CI | discordant (cand-only / base-only) | McNemar p |",
        "|---|---:|---|---:|---:|",
    ]
    for name, c in v["pairs"].items():
        lo, hi = c["diff_ci95"]
        lines.append(
            f"| `{name}` | {pts(c['diff'])} | [{pts(lo)}, {pts(hi)}] | "
            f"{c.get('candidate_only_success', '-')} / {c.get('baseline_only_success', '-')} | {c.get('mcnemar_p')} |"
        )
    a, m, b = v["verdict_a_accuracy"], v["verdict_a_memory"], v["verdict_b_multi_noise"]
    lines += [
        "",
        f"- **Verdict A, accuracy** (G_b16_lowmem vs G_b16_fp32): **{a['result']}**, "
        f"diff {pts(a['diff'])} pts, p={a['mcnemar_p']}" + (f" ({a['note']})" if a.get("note") else ""),
        f"- **Verdict A, memory** (b16_k1 under the 24 GiB cap, peak <= {MEM_MAX_GB:.0f} GB): **{m['result']}**"
        + (f", peak {m.get('peak_gpu_mem_gb')} GB" if m.get("peak_gpu_mem_gb") is not None else "")
        + (f" ({m['note']})" if m.get("note") else ""),
        f"- **Verdict B, multi-noise** (G_b16_mn2 vs G_b16_lowmem): **{b['result']}**, diff {pts(b['diff'])} pts, p={b['mcnemar_p']}",
        f"- Batch 32 -> 16 (G_b16_fp32 vs B0), for information: diff {pts(v['batch_effect_b16_fp32_vs_b0']['diff'])} pts, "
        f"p={v['batch_effect_b16_fp32_vs_b0']['mcnemar_p']}",
        f"- **Route**: {v['route']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b0", required=True)
    ap.add_argument("--b16-fp32", required=True)
    ap.add_argument("--b16-lowmem", required=True)
    ap.add_argument("--mn2", required=True)
    ap.add_argument("--proof-json", default=None, help="turbo_run_checks.py table --out-json of the Step 1 bench")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    paths = {"b0": args.b0, "b16_fp32": args.b16_fp32, "b16_lowmem": args.b16_lowmem, "mn2": args.mn2}
    evals = {g: json.loads(Path(p).read_text()) for g, p in paths.items()}
    for g, ev in evals.items():
        if not ev.get("complete", False):
            print(f"[verdict] WARNING: {g} eval is incomplete ({ev.get('valid_episodes')}/{ev.get('total_episodes')})")
    proof = json.loads(Path(args.proof_json).read_text()) if args.proof_json else None
    v = build(evals, proof)
    v["inputs"] = {**paths, "proof_json": args.proof_json}
    md = to_markdown(v)
    print(md)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(v, indent=2), encoding="utf-8")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
