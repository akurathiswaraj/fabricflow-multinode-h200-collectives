"""Command-line interface for offline planning, simulation, and calibration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from .calibration import CalibrationProfile, fit_calibration, load_jsonl
from .planner import Planner
from .sizes import format_size, parse_size
from .topology import ClusterTopology, topology_summary


def _planner(args: argparse.Namespace) -> Planner:
    topology = ClusterTopology.load(args.topology)
    calibration = (
        CalibrationProfile.load(args.calibration) if args.calibration else None
    )
    return Planner(
        topology,
        calibration=calibration,
        pipeline_chunks=args.pipeline_chunks,
    )


def _plan(args: argparse.Namespace) -> int:
    planner = _planner(args)
    size = parse_size(args.size)
    plan = planner.plan(size, runtime_only=not args.include_design_targets)
    print(topology_summary(planner.topology))
    print(f"message: {format_size(size)}")
    print(f"selected: {plan.selected.strategy.value} ({plan.selected.latency_us:.2f} us)")
    print(f"reason: {plan.reason}")
    print("\ncandidates:")
    for item in sorted(plan.candidates, key=lambda value: value.latency_us):
        support = "runtime" if item.runtime_supported else "design-target"
        print(f"  {item.strategy.value:24} {item.latency_us:12.2f} us  {support}")
        if args.explain:
            for phase in item.phases:
                print(
                    f"    - {phase.name}: {phase.duration_us:.2f} us "
                    f"({phase.link}, {phase.steps} steps)"
                )
            print(f"      assumption: {item.assumption}")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    planner = _planner(args)
    sizes = [parse_size(value) for value in args.sizes.split(",")]
    rows: list[dict[str, object]] = []
    for size in sizes:
        plan = planner.plan(size, runtime_only=not args.include_design_targets)
        for prediction in plan.candidates:
            rows.append(
                {
                    "topology": planner.topology.name,
                    "world_size": planner.topology.world_size,
                    "message_bytes": size,
                    "message": format_size(size),
                    "strategy": prediction.strategy.value,
                    "predicted_us": round(prediction.latency_us, 4),
                    "runtime_supported": prediction.runtime_supported,
                    "selected": prediction.strategy == plan.selected.strategy,
                    "calibration_factor": prediction.calibration_factor,
                    "provenance": "analytical_simulation_not_hardware_measurement",
                }
            )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {output}")
    else:
        print(json.dumps(rows, indent=2))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    profile = fit_calibration(rows, source=str(args.input))
    if not profile.factors:
        raise ValueError(
            "no usable rows; expected strategy, predicted_us, and measured_p50_us"
        )
    profile.save(args.output)
    print(json.dumps(profile.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabricflow",
        description="Topology-aware multi-node all-reduce planning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_planner_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--topology", required=True)
        command.add_argument("--calibration")
        command.add_argument("--pipeline-chunks", type=int, default=8)
        command.add_argument(
            "--include-design-targets",
            action="store_true",
            help="allow the modeled pipelined strategy to win selection",
        )

    plan = subparsers.add_parser("plan", help="choose a strategy for one size")
    add_planner_options(plan)
    plan.add_argument("--size", required=True)
    plan.add_argument("--explain", action="store_true")
    plan.set_defaults(handler=_plan)

    simulate = subparsers.add_parser("simulate", help="sweep message sizes")
    add_planner_options(simulate)
    simulate.add_argument("--sizes", required=True)
    simulate.add_argument("--output")
    simulate.set_defaults(handler=_simulate)

    calibrate = subparsers.add_parser(
        "calibrate", help="fit correction factors from benchmark JSONL"
    )
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(handler=_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

