"""Command-line interface for planning, running, and summarizing benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from ltm_sampling.benchmark import run_benchmark
from ltm_sampling.config import BenchmarkConfig, load_config
from ltm_sampling.reporting import summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ltm-sampling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("plan", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Path to a TOML benchmark configuration",
        )

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("results", type=Path, help="Raw JSONL results")
    summarize.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        output = summarize_results(args.results, args.output)
        print(f"Wrote paired comparisons to {output}")
        return

    config = load_config(args.config)
    if args.command == "plan":
        _print_plan(config)
        return

    _print_plan(config)
    results = run_benchmark(config)
    successes = sum(result["status"] == "success" for result in results)
    comparisons = summarize_results(config.output)
    print(
        f"Completed {successes}/{len(results)} conditions. "
        f"Raw results: {config.output}. Comparisons: {comparisons}"
    )


def _print_plan(config: BenchmarkConfig) -> None:
    tasks = ", ".join(
        f"{dataset.name} ({dataset.problem_type})" for dataset in config.datasets
    )
    samplers = ", ".join(sampler.name for sampler in config.samplers)
    print(f"Datasets: {tasks}")
    print(f"Samplers: {samplers}")
    print(f"Seeds: {', '.join(map(str, config.seeds))}")
    print(
        f"Model: TabFM {config.model.backend}, {config.model.preset} preset, "
        f"{config.model.n_estimators} estimators"
    )
    print(f"Conditions: {config.run_count}; output: {config.output}")
