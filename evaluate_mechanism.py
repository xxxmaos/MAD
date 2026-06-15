"""
Enhanced evaluation for mechanism-based debate experiments.

Adds early-exit, compression, Devil's Advocate, summarizer, and baseline
comparison metrics while preserving the baseline result shape.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from evaluate import evaluate_results


def evaluate_mechanism_results(
    results_data: Dict[str, Any],
    baseline_debate: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate mechanism results and add mechanism-specific metrics.

    Args:
        results_data: Mechanism experiment result dict.
        baseline_debate: Optional baseline debate results for comparison.
        verbose: Whether to print a concise analysis.

    Returns:
        Combined metrics dict.
    """
    eval_data = {
        **results_data,
        "config": {**results_data.get("config", {}), "num_rounds": 0},
    }
    base_metrics = evaluate_results(eval_data, verbose=verbose)
    results = results_data.get("results", [])
    mechanism_metrics = compute_mechanism_metrics(results, baseline_debate)
    base_metrics["mechanism_metrics"] = mechanism_metrics

    if verbose:
        _print_mechanism_metrics(mechanism_metrics)

    return base_metrics


def compute_mechanism_metrics(
    results: List[Dict[str, Any]],
    baseline_debate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute mechanism-specific metrics from per-question results.

    Args:
        results: Per-question mechanism results.
        baseline_debate: Optional baseline debate results.

    Returns:
        Dict of mechanism metrics.
    """
    total = len(results)
    phase_zero_exit_count = 0
    phase_one_exit_count = 0
    max_rounds_reached_count = 0
    actual_round_sum = 0
    early_exit_correct = 0
    early_exit_count = 0
    compression_distribution: Counter[str] = Counter()
    exit_round_histogram: Counter[str] = Counter()
    da_triggered_count = 0
    da_effective_count = 0
    summarizer_total = _zero_usage()
    da_total = _zero_usage()
    total_with_overhead = 0

    for result in results:
        mechanism = result.get("mechanism", {})
        actual_rounds = int(mechanism.get("actual_rounds", 0) or 0)
        exit_reason = mechanism.get("exit_reason", "")
        early_exit = bool(mechanism.get("early_exit", False))

        actual_round_sum += actual_rounds
        exit_round_histogram[f"R{actual_rounds}"] += 1

        if exit_reason == "phase_zero_all_agree":
            phase_zero_exit_count += 1
        elif exit_reason == "max_rounds_reached":
            max_rounds_reached_count += 1
        elif early_exit:
            phase_one_exit_count += 1

        if early_exit:
            early_exit_count += 1
            if result.get("final_answer") == result.get("ground_truth"):
                early_exit_correct += 1

        for log_item in mechanism.get("mechanism_log", []):
            compression = log_item.get("compression_level")
            if compression:
                compression_distribution[compression] += 1
            da_info = log_item.get("da")
            if da_info:
                da_triggered_count += 1
                if da_info.get("action") == "da_effective":
                    da_effective_count += 1

        _add_usage(
            summarizer_total,
            mechanism.get("summarizer_token_usage", {}).get("total", {}),
        )
        _add_usage(da_total, mechanism.get("da_token_usage", {}).get("total", {}))

        total_with_overhead += int(
            mechanism.get("total_token_usage_with_overhead", {}).get(
                "total_tokens",
                result.get("token_usage", {}).get("total", {}).get("total_tokens", 0),
            )
            or 0
        )

    baseline_comparison = _compare_with_baseline(results, baseline_debate)
    baseline_tokens = _baseline_total_tokens(baseline_debate)
    tokens_saved = baseline_tokens - total_with_overhead if baseline_tokens else 0
    tokens_saved_ratio = tokens_saved / baseline_tokens if baseline_tokens else 0.0

    compression_total = sum(compression_distribution.values())
    compression_ratios = {
        key: (value / compression_total if compression_total else 0.0)
        for key, value in sorted(compression_distribution.items())
    }

    return {
        "phase_zero_exit_count": phase_zero_exit_count,
        "phase_one_exit_count": phase_one_exit_count,
        "max_rounds_reached_count": max_rounds_reached_count,
        "average_actual_rounds": actual_round_sum / total if total else 0.0,
        "exit_round_histogram": dict(sorted(exit_round_histogram.items())),
        "compression_distribution": {
            key: compression_distribution.get(key, 0)
            for key in ("full", "partial", "none", "dispute")
        },
        "compression_ratios": {
            key: compression_ratios.get(key, 0.0)
            for key in ("full", "partial", "none", "dispute")
        },
        "da_triggered_count": da_triggered_count,
        "da_effective_count": da_effective_count,
        "da_effective_rate": (
            da_effective_count / da_triggered_count if da_triggered_count else 0.0
        ),
        "early_exit_accuracy": (
            early_exit_correct / early_exit_count if early_exit_count else 0.0
        ),
        "summarizer_total_tokens": summarizer_total["total_tokens"],
        "summarizer_avg_tokens_per_question": (
            summarizer_total["total_tokens"] / total if total else 0.0
        ),
        "da_total_tokens": da_total["total_tokens"],
        "total_tokens_with_overhead": total_with_overhead,
        "tokens_saved_vs_baseline": tokens_saved,
        "tokens_saved_ratio_vs_baseline": tokens_saved_ratio,
        "baseline_four_way": baseline_comparison,
    }


def print_compare_table(
    single_results: Dict[str, Any],
    sc_results: Dict[str, Any],
    debate_results: Dict[str, Any],
    mechanism_results: Dict[str, Any],
) -> None:
    """
    Print comparison table for Single, SC, Debate, and Mechanism.

    Args:
        single_results: Latest single-mode results.
        sc_results: Latest self-consistency results.
        debate_results: Latest baseline debate results.
        mechanism_results: Latest mechanism results.
    """
    mechanism_metrics = mechanism_results.get("metrics", {}).get("mechanism_metrics")
    if mechanism_metrics is None:
        mechanism_metrics = compute_mechanism_metrics(
            mechanism_results.get("results", []),
            debate_results,
        )

    single_tokens = _total_tokens(single_results)
    sc_tokens = _total_tokens(sc_results)
    debate_tokens = _total_tokens(debate_results)
    mechanism_tokens = int(mechanism_metrics.get("total_tokens_with_overhead", 0) or 0)

    count = max(len(mechanism_results.get("results", [])), 1)
    early_exit_count = (
        mechanism_metrics.get("phase_zero_exit_count", 0)
        + mechanism_metrics.get("phase_one_exit_count", 0)
    )

    print("\n" + "=" * 78)
    print("COMPARISON: Single vs SC vs Debate vs Mechanism")
    print("=" * 78)
    print(f"{'':<14} {'Single':>12} {'SC':>12} {'Debate':>12} {'Mechanism':>14}")
    print("-" * 78)
    print(
        f"{'准确率:':<14} "
        f"{_accuracy(single_results):>11.1%} "
        f"{_accuracy(sc_results):>11.1%} "
        f"{_accuracy(debate_results):>11.1%} "
        f"{_accuracy(mechanism_results):>13.1%}"
    )
    print(
        f"{'总Token:':<14} "
        f"{single_tokens:>12,} {sc_tokens:>12,} "
        f"{debate_tokens:>12,} {mechanism_tokens:>14,}"
    )
    print(
        f"{'每题Token:':<14} "
        f"{single_tokens / max(len(single_results.get('results', [])), 1):>12.0f} "
        f"{sc_tokens / max(len(sc_results.get('results', [])), 1):>12.0f} "
        f"{debate_tokens / max(len(debate_results.get('results', [])), 1):>12.0f} "
        f"{mechanism_tokens / count:>14.0f}"
    )
    print(
        f"{'平均轮数:':<14} "
        f"{0:>12.1f} {0:>12.1f} "
        f"{debate_results.get('config', {}).get('num_rounds', 0):>12.1f} "
        f"{mechanism_metrics.get('average_actual_rounds', 0.0):>14.1f}"
    )
    print(
        f"{'早退率:':<14} {'-':>12} {'-':>12} {'-':>12} "
        f"{early_exit_count / count:>13.1%}"
    )
    print(
        f"{'早退准确率:':<14} {'-':>12} {'-':>12} {'-':>12} "
        f"{mechanism_metrics.get('early_exit_accuracy', 0.0):>13.1%}"
    )
    print("=" * 78)


def _compare_with_baseline(
    mechanism_results: List[Dict[str, Any]],
    baseline_debate: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    """Compute ✓→✓, ✗→✓, ✓→✗, ✗→✗ against baseline debate."""
    counts = {"correct_to_correct": 0, "wrong_to_correct": 0, "correct_to_wrong": 0, "wrong_to_wrong": 0}
    if not baseline_debate:
        return counts

    baseline_by_id = {
        item.get("question_id"): item
        for item in baseline_debate.get("results", [])
    }
    for item in mechanism_results:
        baseline = baseline_by_id.get(item.get("question_id"))
        if not baseline:
            continue
        gt = item.get("ground_truth")
        baseline_correct = baseline.get("final_answer") == gt
        mechanism_correct = item.get("final_answer") == gt
        if baseline_correct and mechanism_correct:
            counts["correct_to_correct"] += 1
        elif not baseline_correct and mechanism_correct:
            counts["wrong_to_correct"] += 1
        elif baseline_correct and not mechanism_correct:
            counts["correct_to_wrong"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    return counts


def _print_mechanism_metrics(metrics: Dict[str, Any]) -> None:
    """Print mechanism-specific metrics."""
    print("\n  Mechanism Metrics:")
    print(f"    Phase zero exits: {metrics['phase_zero_exit_count']}")
    print(f"    Phase one exits: {metrics['phase_one_exit_count']}")
    print(f"    Max rounds reached: {metrics['max_rounds_reached_count']}")
    print(f"    Average actual rounds: {metrics['average_actual_rounds']:.2f}")
    print(f"    Exit round histogram: {metrics['exit_round_histogram']}")
    print(f"    Compression distribution: {metrics['compression_distribution']}")
    print(
        f"    Devil's Advocate: {metrics['da_triggered_count']} triggered, "
        f"{metrics['da_effective_count']} effective"
    )
    print(f"    Summarizer tokens: {metrics['summarizer_total_tokens']:,}")
    print(
        f"    Tokens saved vs baseline: "
        f"{metrics['tokens_saved_vs_baseline']:,} "
        f"({metrics['tokens_saved_ratio_vs_baseline']:.1%})"
    )


def _baseline_total_tokens(baseline: Optional[Dict[str, Any]]) -> int:
    """Return baseline total tokens if available."""
    if not baseline:
        return 0
    return _total_tokens(baseline)


def _total_tokens(results_data: Dict[str, Any]) -> int:
    """Return total token usage from metrics or per-result fallback."""
    metrics_tokens = (
        results_data.get("metrics", {})
        .get("token_usage", {})
        .get("total_tokens")
    )
    if metrics_tokens is not None:
        return int(metrics_tokens or 0)
    return sum(
        int(
            item.get("token_usage", {})
            .get("total", {})
            .get("total_tokens", 0)
            or 0
        )
        for item in results_data.get("results", [])
    )


def _accuracy(results_data: Dict[str, Any]) -> float:
    """Return overall accuracy from metrics or recompute from results."""
    metric = results_data.get("metrics", {}).get("overall_accuracy")
    if metric is not None:
        return float(metric)
    results = results_data.get("results", [])
    if not results:
        return 0.0
    correct = sum(
        1 for item in results if item.get("final_answer") == item.get("ground_truth")
    )
    return correct / len(results)


def _zero_usage() -> Dict[str, int]:
    """Return a zero token usage dict."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_usage(target: Dict[str, int], source: Dict[str, Any]) -> None:
    """Add source usage into target in place."""
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] += int(source.get(key, 0) or 0)
