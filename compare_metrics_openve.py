#!/usr/bin/env python3
"""Compare latency, memory, and token costs from two profiled OpenVE runs.

Metric records are first collapsed over each Ulysses sequence-parallel group.
For a group of size two, ranks ``0/1``, ``2/3``, ... are one logical DP sample;
the per-sample latency and peak memory are the maximum over its ranks.  This
reports the critical-path latency a user observes, rather than counting the
same inference twice.  OpenVE records have unique ``sample_id`` values, so
this copy defaults to pairing after Ulysses aggregation by ``sample_id``.
Ordinal pairing remains available for compatible runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LATENCY_FIELDS = (
    "preprocess_total_latency_ms",
    "transform_inputs_latency_ms",
    "source_vit_latency_ms",
    "pixelprune_selector_latency_ms",
    "pixelprune_index_conversion_latency_ms",
    "pixelprune_sequence_rebuild_latency_ms",
    "pixelprune_overhead_ms",
    "planner_mllm_forward_latency_ms",
    "connector_latency_ms",
    "vit_decoder_latency_ms",
    "planner_latency_ms",
    "dit_diffusion_latency_ms",
    "vae_encode_latency_ms",
    "vae_decode_latency_ms",
    "e2e_excluding_io_ms",
    "e2e_including_io_ms",
)

MEMORY_FIELDS = ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")
TOKEN_FIELDS = (
    "visual_input_tokens_before",
    "visual_input_tokens_after",
    "visual_input_token_reduction_ratio",
    "mllm_sequence_length_before",
    "mllm_sequence_length_after",
)


@dataclass(frozen=True)
class Record:
    """A metric record plus the key used to pair it with the other method."""

    key: tuple[int, int]
    values: dict[str, Any]


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric_values(records: Iterable[Record], field: str) -> list[float]:
    values = []
    for record in records:
        value = record.values.get(field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def aggregate(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def load_rank_records(run_dir: Path) -> list[Record]:
    paths = sorted(run_dir.glob("metrics.rank*.jsonl"))
    if not paths:
        raise ValueError(f"No metrics.rank*.jsonl files found in {run_dir}")

    records = []
    for path in paths:
        try:
            rank = int(path.stem.removeprefix("metrics.rank"))
        except ValueError as error:
            raise ValueError(f"Unexpected metric filename: {path.name}") from error
        with path.open(encoding="utf-8") as file:
            for ordinal, line in enumerate(file):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{ordinal + 1}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object in {path}:{ordinal + 1}")
                records.append(Record((rank, ordinal), value))
    return records


def collapse_ulysses_records(records: list[Record], ulysses: int) -> list[Record]:
    """Collapse duplicate metric rows written by ranks in one Ulysses group."""
    if ulysses < 1:
        raise ValueError(f"ulysses must be >= 1, got {ulysses}")
    by_key: dict[tuple[int, int], list[Record]] = {}
    for record in records:
        rank, ordinal = record.key
        by_key.setdefault((rank // ulysses, ordinal), []).append(record)

    collapsed = []
    for key, group in sorted(by_key.items()):
        ranks = sorted(record.key[0] for record in group)
        expected = list(range(key[0] * ulysses, (key[0] + 1) * ulysses))
        if ranks != expected:
            raise ValueError(
                f"Incomplete Ulysses group for logical sample {key}: "
                f"found ranks {ranks}, expected {expected}"
            )
        sample_ids = {record.values.get("sample_id") for record in group}
        if len(sample_ids) != 1:
            raise ValueError(
                f"Ulysses ranks describe different samples at {key}: {sorted(sample_ids)!r}"
            )

        # All timing fields are local rank measurements.  The sample completes
        # only after the slowest Ulysses rank, so max is the right wall-time
        # statistic.  The same rule gives the actual peak device memory needed.
        values = dict(group[0].values)
        numeric_keys = {
            field
            for record in group
            for field, value in record.values.items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        for field in numeric_keys:
            field_values = [
                float(record.values[field])
                for record in group
                if isinstance(record.values.get(field), (int, float))
                and math.isfinite(record.values[field])
            ]
            values[field] = max(field_values)
        values["ulysses_ranks"] = ranks
        collapsed.append(Record(key, values))
    return collapsed


def pair_records_by_ordinal(
    baseline: list[Record], pixelprune: list[Record], allow_unpaired: bool
) -> tuple[list[tuple[Record, Record]], dict[str, list[tuple[int, int]]]]:
    baseline_by_key = {record.key: record for record in baseline}
    pixelprune_by_key = {record.key: record for record in pixelprune}
    only_baseline = sorted(baseline_by_key.keys() - pixelprune_by_key.keys())
    only_pixelprune = sorted(pixelprune_by_key.keys() - baseline_by_key.keys())
    if only_baseline or only_pixelprune:
        if not allow_unpaired:
            raise ValueError(
                "Metrics cannot be paired by (DP group, record order). "
                f"Only baseline: {only_baseline[:5]}; only PixelPrune: {only_pixelprune[:5]}. "
                "Use --allow-unpaired only if dropping these samples is intended."
            )
    common_keys = sorted(baseline_by_key.keys() & pixelprune_by_key.keys())
    return (
        [(baseline_by_key[key], pixelprune_by_key[key]) for key in common_keys],
        {"only_baseline": only_baseline, "only_pixelprune": only_pixelprune},
    )


def pair_records_by_sample_id(
    baseline: list[Record], pixelprune: list[Record], allow_unpaired: bool
) -> tuple[list[tuple[Record, Record]], dict[str, list[str]]]:
    """Pair logical OpenVE samples by their unique sample_id values."""

    def index_records(records: list[Record], label: str) -> dict[str, Record]:
        indexed: dict[str, Record] = {}
        for record in records:
            sample_id = record.values.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{label} record {record.key} has no usable sample_id")
            if sample_id in indexed:
                raise ValueError(
                    f"{label} contains duplicate sample_id {sample_id!r}; "
                    "use --pair-by ordinal for a benchmark with non-unique IDs"
                )
            indexed[sample_id] = record
        return indexed

    baseline_by_id = index_records(baseline, "Baseline")
    pixelprune_by_id = index_records(pixelprune, "PixelPrune")
    only_baseline = sorted(baseline_by_id.keys() - pixelprune_by_id.keys())
    only_pixelprune = sorted(pixelprune_by_id.keys() - baseline_by_id.keys())
    if (only_baseline or only_pixelprune) and not allow_unpaired:
        raise ValueError(
            "Metrics cannot be paired by sample_id. "
            f"Only baseline: {only_baseline[:5]}; only PixelPrune: {only_pixelprune[:5]}. "
            "Use --allow-unpaired only if dropping these samples is intended."
        )
    common_ids = sorted(baseline_by_id.keys() & pixelprune_by_id.keys())
    return (
        [(baseline_by_id[sample_id], pixelprune_by_id[sample_id]) for sample_id in common_ids],
        {"only_baseline": only_baseline, "only_pixelprune": only_pixelprune},
    )


def pair_records(
    baseline: list[Record],
    pixelprune: list[Record],
    allow_unpaired: bool,
    pair_by: str,
) -> tuple[list[tuple[Record, Record]], dict[str, list[Any]]]:
    if pair_by == "ordinal":
        return pair_records_by_ordinal(baseline, pixelprune, allow_unpaired)
    if pair_by == "sample-id":
        return pair_records_by_sample_id(baseline, pixelprune, allow_unpaired)
    raise ValueError(f"Unknown pairing mode: {pair_by}")


def compare_field(
    pairs: list[tuple[Record, Record]], field: str, statistic: str
) -> dict[str, Any] | None:
    baseline_values = []
    pixelprune_values = []
    paired_delta_values = []
    for baseline, pixelprune in pairs:
        baseline_value = baseline.values.get(field)
        pixelprune_value = pixelprune.values.get(field)
        if not (
            isinstance(baseline_value, (int, float))
            and isinstance(pixelprune_value, (int, float))
            and math.isfinite(baseline_value)
            and math.isfinite(pixelprune_value)
        ):
            continue
        baseline_values.append(float(baseline_value))
        pixelprune_values.append(float(pixelprune_value))
        paired_delta_values.append(float(pixelprune_value) - float(baseline_value))

    baseline_summary = aggregate(baseline_values)
    pixelprune_summary = aggregate(pixelprune_values)
    paired_delta = aggregate(paired_delta_values)
    if baseline_summary is None or pixelprune_summary is None or paired_delta is None:
        return None

    baseline_stat = float(baseline_summary[statistic])
    pixelprune_stat = float(pixelprune_summary[statistic])
    result: dict[str, Any] = {
        "baseline": baseline_summary,
        "pixelprune": pixelprune_summary,
        "paired_delta_pixelprune_minus_baseline": paired_delta,
        "delta_at_selected_stat": pixelprune_stat - baseline_stat,
    }
    if baseline_stat:
        result["reduction_ratio_at_selected_stat"] = 1 - pixelprune_stat / baseline_stat
    if pixelprune_stat:
        result["speedup_at_selected_stat"] = baseline_stat / pixelprune_stat
    return result


def selected(summary: dict[str, Any], statistic: str) -> float:
    return float(summary[statistic])


def component_summary(records: list[Record], statistic: str) -> dict[str, Any]:
    components = {
        field: aggregate(numeric_values(records, field))
        for field in (
            "pixelprune_selector_latency_ms",
            "pixelprune_index_conversion_latency_ms",
            "pixelprune_sequence_rebuild_latency_ms",
            "pixelprune_overhead_ms",
        )
    }
    components = {field: value for field, value in components.items() if value is not None}
    return {
        "components": components,
        "selected_statistic": statistic,
        "reported_total_overhead_ms": components.get("pixelprune_overhead_ms"),
    }


def net_vit_path(
    pairs: list[tuple[Record, Record]], statistic: str
) -> dict[str, float] | None:
    baseline_vit = []
    pixelprune_vit = []
    overhead = []
    net_savings = []
    for baseline, pixelprune in pairs:
        baseline_value = baseline.values.get("source_vit_latency_ms")
        pixelprune_value = pixelprune.values.get("source_vit_latency_ms")
        overhead_value = pixelprune.values.get("pixelprune_overhead_ms")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (baseline_value, pixelprune_value, overhead_value)
        ):
            continue
        baseline_vit.append(float(baseline_value))
        pixelprune_vit.append(float(pixelprune_value))
        overhead.append(float(overhead_value))
        net_savings.append(
            float(baseline_value) - float(pixelprune_value) - float(overhead_value)
        )
    summaries = [aggregate(values) for values in (baseline_vit, pixelprune_vit, overhead, net_savings)]
    if any(summary is None for summary in summaries):
        return None
    baseline_summary, pixelprune_summary, overhead_summary, net_summary = summaries
    assert baseline_summary and pixelprune_summary and overhead_summary and net_summary
    baseline_stat = selected(baseline_summary, statistic)
    pixelprune_stat = selected(pixelprune_summary, statistic)
    overhead_stat = selected(overhead_summary, statistic)
    return {
        "baseline_source_vit_ms": baseline_stat,
        "pixelprune_source_vit_ms": pixelprune_stat,
        "gross_source_vit_saving_ms": baseline_stat - pixelprune_stat,
        "pixelprune_overhead_ms": overhead_stat,
        "net_source_vit_path_saving_ms": selected(net_summary, statistic),
        "net_source_vit_path_reduction_ratio": selected(net_summary, statistic) / baseline_stat
        if baseline_stat
        else math.nan,
    }


def build_report(
    baseline_dir: Path,
    pixelprune_dir: Path,
    statistic: str = "mean",
    ulysses: int = 1,
    allow_unpaired: bool = False,
    pair_by: str = "sample-id",
) -> dict[str, Any]:
    baseline_rank_records = load_rank_records(baseline_dir)
    pixelprune_rank_records = load_rank_records(pixelprune_dir)
    baseline = collapse_ulysses_records(baseline_rank_records, ulysses)
    pixelprune = collapse_ulysses_records(pixelprune_rank_records, ulysses)
    pairs, unpaired = pair_records(baseline, pixelprune, allow_unpaired, pair_by)
    if not pairs:
        raise ValueError("No paired logical samples remain after Ulysses grouping")
    fields = LATENCY_FIELDS + MEMORY_FIELDS + TOKEN_FIELDS
    comparisons = {
        field: comparison
        for field in fields
        if (comparison := compare_field(pairs, field, statistic)) is not None
    }
    ranks = sorted({record.key[0] for record in baseline_rank_records})
    return {
        "baseline_dir": str(baseline_dir),
        "pixelprune_dir": str(pixelprune_dir),
        "pairing": (
            "sample_id after Ulysses aggregation"
            if pair_by == "sample-id"
            else "(DP group, zero-based non-empty record ordinal)"
        ),
        "logical_sample_count": len(pairs),
        "raw_rank_record_count": {
            "baseline": len(baseline_rank_records),
            "pixelprune": len(pixelprune_rank_records),
        },
        "world_ranks": ranks,
        "ulysses": ulysses,
        "group_aggregation": "max across Ulysses ranks (critical-path time / peak memory)",
        "unpaired_logical_samples": unpaired,
        "selected_statistic": statistic,
        "comparisons": comparisons,
        # Use the same paired sample set as the headline comparison.  This is
        # important when --allow-unpaired drops a partial rerun.
        "pixelprune_overhead": component_summary(
            [pixelprune_record for _, pixelprune_record in pairs], statistic
        ),
        "net_source_vit_path": net_vit_path(pairs, statistic),
    }


def ms(value: float) -> str:
    return f"{value:,.1f} ms"


def print_report(report: dict[str, Any], detailed: bool = True) -> None:
    statistic = report["selected_statistic"]
    print(
        f"Paired {report['logical_sample_count']} logical samples over world ranks "
        f"{report['world_ranks']} (Ulysses={report['ulysses']}; {statistic})."
    )
    unpaired = report["unpaired_logical_samples"]
    if unpaired["only_baseline"] or unpaired["only_pixelprune"]:
        print(
            "Dropped unpaired logical samples: "
            f"baseline-only={len(unpaired['only_baseline'])}, "
            f"pixelprune-only={len(unpaired['only_pixelprune'])}."
        )

    print("\nHeadline latency (PixelPrune - baseline; negative means faster)")
    print(f"{'metric':42} {'baseline':>14} {'pixelprune':>14} {'delta':>14} {'reduction':>10}")
    for field in ("planner_latency_ms", "dit_diffusion_latency_ms", "e2e_excluding_io_ms"):
        item = report["comparisons"].get(field)
        if item is None:
            continue
        baseline_value = selected(item["baseline"], statistic)
        pixelprune_value = selected(item["pixelprune"], statistic)
        reduction = item.get("reduction_ratio_at_selected_stat")
        reduction_text = f"{reduction:.1%}" if reduction is not None else "n/a"
        print(
            f"{field:42} {ms(baseline_value):>14} {ms(pixelprune_value):>14} "
            f"{ms(float(item['delta_at_selected_stat'])):>14} {reduction_text:>10}"
        )
    overhead = report["pixelprune_overhead"].get("reported_total_overhead_ms")
    if overhead:
        print(f"{'pixelprune_overhead_ms':42} {'0.0 ms':>14} {ms(selected(overhead, statistic)):>14}")

    if not detailed:
        return
    print("\nLatency (PixelPrune - baseline; negative means faster)")
    print(f"{'metric':42} {'baseline':>14} {'pixelprune':>14} {'delta':>14} {'reduction':>10}")
    for field in LATENCY_FIELDS:
        item = report["comparisons"].get(field)
        if item is None:
            continue
        baseline_value = selected(item["baseline"], statistic)
        pixelprune_value = selected(item["pixelprune"], statistic)
        reduction = item.get("reduction_ratio_at_selected_stat")
        reduction_text = f"{reduction:.1%}" if reduction is not None else "n/a"
        print(
            f"{field:42} {ms(baseline_value):>14} {ms(pixelprune_value):>14} "
            f"{ms(float(item['delta_at_selected_stat'])):>14} {reduction_text:>10}"
        )

    print("\nMemory (values shown in GiB)")
    print(f"{'metric':42} {'baseline':>14} {'pixelprune':>14} {'delta':>14} {'reduction':>10}")
    for field in MEMORY_FIELDS:
        item = report["comparisons"].get(field)
        if item is None:
            continue
        baseline_value = selected(item["baseline"], statistic) / 1024**3
        pixelprune_value = selected(item["pixelprune"], statistic) / 1024**3
        delta = pixelprune_value - baseline_value
        reduction = item.get("reduction_ratio_at_selected_stat")
        reduction_text = f"{reduction:.1%}" if reduction is not None else "n/a"
        print(
            f"{field:42} {baseline_value:>12.2f} GiB {pixelprune_value:>12.2f} GiB "
            f"{delta:>+12.2f} GiB {reduction_text:>10}"
        )

    print("\nToken counts")
    print(f"{'metric':42} {'baseline':>14} {'pixelprune':>14} {'delta':>14} {'reduction':>10}")
    for field in (
        "visual_input_tokens_before",
        "visual_input_tokens_after",
        "mllm_sequence_length_before",
        "mllm_sequence_length_after",
    ):
        item = report["comparisons"].get(field)
        if item is None:
            continue
        baseline_value = selected(item["baseline"], statistic)
        pixelprune_value = selected(item["pixelprune"], statistic)
        reduction = item.get("reduction_ratio_at_selected_stat")
        reduction_text = f"{reduction:.1%}" if reduction is not None else "n/a"
        print(
            f"{field:42} {baseline_value:>14,.0f} {pixelprune_value:>14,.0f} "
            f"{pixelprune_value - baseline_value:>+14,.0f} {reduction_text:>10}"
        )

    net = report.get("net_source_vit_path")
    if net:
        print("\nPixelPrune overhead and source-ViT-path effect")
        components = report["pixelprune_overhead"]["components"]
        for field in (
            "pixelprune_selector_latency_ms",
            "pixelprune_index_conversion_latency_ms",
            "pixelprune_sequence_rebuild_latency_ms",
        ):
            component = components.get(field)
            if component:
                print(f"  {field}: {ms(selected(component, statistic))}")
        print(f"  gross source-ViT saving: {ms(net['gross_source_vit_saving_ms'])}")
        print(f"  PixelPrune overhead:     {ms(net['pixelprune_overhead_ms'])}")
        print(f"  net source-ViT saving:   {ms(net['net_source_vit_path_saving_ms'])} "
              f"({net['net_source_vit_path_reduction_ratio']:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path, help="Directory containing baseline metrics.rank*.jsonl")
    parser.add_argument("pixelprune_dir", type=Path, help="Directory containing PixelPrune metrics.rank*.jsonl")
    parser.add_argument(
        "--statistic",
        choices=("mean", "median", "p90"),
        default="mean",
        help="Aggregate used for the headline comparison (default: mean)",
    )
    parser.add_argument(
        "--ulysses",
        type=int,
        default=1,
        help="Ulysses sequence-parallel degree; use 2 for the DP4 × Ulysses2 runs",
    )
    parser.add_argument(
        "--pair-by",
        choices=("sample-id", "ordinal"),
        default="sample-id",
        help="Logical-sample pairing mode (default: sample-id for OpenVE)",
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Compare the paired intersection and report dropped logical samples",
    )
    parser.add_argument("--output", type=Path, help="Write the full machine-readable JSON report here")
    args = parser.parse_args()

    try:
        report = build_report(
            args.baseline_dir,
            args.pixelprune_dir,
            args.statistic,
            args.ulysses,
            args.allow_unpaired,
            args.pair_by,
        )
    except ValueError as error:
        parser.error(str(error))
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nFull JSON report: {args.output}")


if __name__ == "__main__":
    main()
