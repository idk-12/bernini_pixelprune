#!/usr/bin/env python3
"""
Analyze and compare baseline vs PixelPrune timing and accuracy.

Usage:
  python scripts/analyze_timing.py \
      --baseline-log  eval/outputs/full_baseline_qwen35/logs/e2e.jsonl \
      --pp-log        eval/outputs/pixelprune_doc_qwen35/logs/e2e.jsonl \
      --pp-vit-log    eval/outputs/pixelprune_doc_qwen35/logs/vit.jsonl \
      --baseline-dir  eval/outputs/full_baseline_qwen35 \
      --pp-dir        eval/outputs/pixelprune_doc_qwen35 \
      [--dataset DocVQA_VAL]
"""
import argparse
import glob
import json
import os
import statistics


def load_jsonl(path):
    """Load all records from one or more rank JSONL files."""
    records = []
    if not path:
        return records
    base, ext = os.path.splitext(path)
    # Collect matching rank files: base.e2e.rank0.jsonl etc.
    # Also try direct path
    candidates = glob.glob(f"{base}*{ext}") + ([path] if os.path.exists(path) else [])
    seen = set()
    for p in sorted(set(candidates)):
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def summarize_e2e(records):
    """Extract per-sample total_time_s from e2e records."""
    times = [r['total_time_s'] for r in records if r.get('type') == 'e2e' and 'total_time_s' in r]
    ttfts = [r['ttft_ms'] for r in records if r.get('type') == 'e2e' and r.get('ttft_ms') is not None]
    tok_counts = [r['num_input_tokens'] for r in records if r.get('type') == 'e2e' and 'num_input_tokens' in r]
    return times, ttfts, tok_counts


def summarize_vit(records):
    """Extract per-image retain ratios from vit records."""
    all_ratios = []
    for r in records:
        if r.get('type') == 'vit' and r.get('retain_ratios'):
            all_ratios.extend(r['retain_ratios'])
    org_lens = [l for r in records if r.get('type') == 'vit' for l in r.get('org_merged_lens', [])]
    new_lens = [l for r in records if r.get('type') == 'vit' for l in r.get('new_merged_lens', [])]
    return all_ratios, org_lens, new_lens


def find_accuracy(result_dir, dataset):
    """Search for VLMEvalKit accuracy result files."""
    if not result_dir or not dataset:
        return None
    patterns = [
        os.path.join(result_dir, '**', f'*{dataset}*'),
        os.path.join(result_dir, f'*{dataset}*'),
    ]
    for pat in patterns:
        files = glob.glob(pat, recursive=True)
        for f in sorted(files):
            if f.endswith('.json') or f.endswith('.xlsx') or f.endswith('.csv'):
                return f
    return None


def stats_line(vals, unit='s'):
    if not vals:
        return 'N/A'
    return (f"n={len(vals)}  mean={statistics.mean(vals):.3f}{unit}  "
            f"median={statistics.median(vals):.3f}{unit}  "
            f"total={sum(vals):.1f}{unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-log', default='', help='Baseline e2e JSONL (base path)')
    ap.add_argument('--pp-log', default='', help='PixelPrune e2e JSONL (base path)')
    ap.add_argument('--pp-vit-log', default='', help='PixelPrune VIT stats JSONL (base path)')
    ap.add_argument('--baseline-dir', default='', help='Baseline output dir for accuracy lookup')
    ap.add_argument('--pp-dir', default='', help='PixelPrune output dir for accuracy lookup')
    ap.add_argument('--dataset', default='', help='Dataset name for accuracy lookup')
    args = ap.parse_args()

    # Auto-detect log paths if not given
    if not args.baseline_log and args.baseline_dir:
        args.baseline_log = os.path.join(args.baseline_dir, 'logs', 'e2e.jsonl')
    if not args.pp_log and args.pp_dir:
        args.pp_log = os.path.join(args.pp_dir, 'logs', 'e2e.jsonl')
    if not args.pp_vit_log and args.pp_dir:
        args.pp_vit_log = os.path.join(args.pp_dir, 'logs', 'vit.jsonl')

    print("=" * 64)
    print("PixelPrune Timing & Accuracy Comparison")
    print("=" * 64)

    # ── E2E timing ────────────────────────────────────────────────
    bl_recs = load_jsonl(args.baseline_log)
    pp_recs = load_jsonl(args.pp_log)

    bl_times, bl_ttfts, bl_toks = summarize_e2e(bl_recs)
    pp_times, pp_ttfts, pp_toks = summarize_e2e(pp_recs)

    print("\n[Inference Time per Sample]")
    print(f"  Baseline:   {stats_line(bl_times)}")
    print(f"  PixelPrune: {stats_line(pp_times)}")
    if bl_times and pp_times:
        speedup = statistics.mean(bl_times) / statistics.mean(pp_times)
        total_speedup = sum(bl_times) / sum(pp_times)
        print(f"  Speedup:    {speedup:.2f}x (per-sample mean)  |  {total_speedup:.2f}x (total)")

    print("\n[TTFT (Time To First Token)]")
    print(f"  Baseline:   {stats_line(bl_ttfts, 'ms')}")
    print(f"  PixelPrune: {stats_line(pp_ttfts, 'ms')}")
    if bl_ttfts and pp_ttfts:
        print(f"  TTFT speedup: {statistics.mean(bl_ttfts)/statistics.mean(pp_ttfts):.2f}x")

    print("\n[Input Token Count]")
    print(f"  Baseline:   {stats_line(bl_toks, ' tok')}")
    print(f"  PixelPrune: {stats_line(pp_toks, ' tok')}")
    if bl_toks and pp_toks:
        token_reduction = 1 - statistics.mean(pp_toks) / statistics.mean(bl_toks)
        print(f"  Token reduction: {token_reduction:.1%}")

    # ── Compression ratio (VIT stats) ─────────────────────────────
    vit_recs = load_jsonl(args.pp_vit_log)
    ratios, org_lens, new_lens = summarize_vit(vit_recs)

    print("\n[PixelPrune Compression Ratio (per image)]")
    if ratios:
        print(f"  n={len(ratios)} images")
        print(f"  Mean retain ratio: {statistics.mean(ratios):.3f}  "
              f"({(1-statistics.mean(ratios)):.1%} pruned)")
        print(f"  Median:  {statistics.median(ratios):.3f}")
        print(f"  Min/Max: {min(ratios):.3f} / {max(ratios):.3f}")
        if org_lens and new_lens:
            total_org = sum(org_lens)
            total_new = sum(new_lens)
            print(f"  Total merged tokens: {total_org} → {total_new} "
                  f"(kept {total_new/total_org:.1%})")
    else:
        print("  No VIT log found (set PIXELPRUNE_LOG_FILE and PIXELPRUNE_VERBOSE=true)")

    # ── Accuracy lookup ───────────────────────────────────────────
    print("\n[Accuracy]")
    bl_acc = find_accuracy(args.baseline_dir, args.dataset)
    pp_acc = find_accuracy(args.pp_dir, args.dataset)
    if bl_acc:
        print(f"  Baseline result file:   {bl_acc}")
    else:
        print(f"  Baseline result file:   not found in {args.baseline_dir!r}")
    if pp_acc:
        print(f"  PixelPrune result file: {pp_acc}")
    else:
        print(f"  PixelPrune result file: not found in {args.pp_dir!r}")

    print("\n  (open result files above for exact accuracy numbers)")
    print("=" * 64)


if __name__ == '__main__':
    main()
