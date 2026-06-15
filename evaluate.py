"""
Evaluation metrics and result analysis for debate experiments.
Compares accuracy, token usage, and debate patterns across modes.
"""

from typing import Dict, List, Any
from collections import Counter
import json


def evaluate_results(results_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate experiment results and compute metrics.
    
    Computes:
    - Overall accuracy
    - Per-round accuracy (for debate mode)
    - Full agreement rate and groupthink rate (for debate mode)
    - Answer change rate (for debate mode)
    - Per-category accuracy
    - Token consumption statistics
    - Answer distribution patterns
    
    Args:
        results_data: Experiment results dict from run_experiment.py
        verbose: Print detailed metrics
        
    Returns:
        dict: Computed metrics
    """
    mode = results_data["config"]["mode"]
    num_rounds = results_data["config"]["num_rounds"]
    results = results_data["results"]
    
    metrics = {
        "overall_accuracy": 0.0,
        "rounds": {},
        "by_category": {},
        "token_usage": {},
        "answer_distribution": {}
    }
    
    # Overall accuracy
    correct = sum(1 for r in results if r["final_answer"] == r["ground_truth"])
    metrics["overall_accuracy"] = correct / len(results) if results else 0.0
    
    if verbose:
        print(f"\n=== Evaluation Results ({mode.upper()}) ===")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%} ({correct}/{len(results)})")
    
    # Per-round metrics (for debate and SC modes)
    for round_num in range(num_rounds + 1):
        round_key = str(round_num)
        if round_key in results[0].get("answers_by_round", {}):
            round_answers = {}
            round_correct = 0
            full_agreement = 0
            full_agreement_wrong = 0
            answer_distribution = []
            
            for result in results:
                round_answers[result["question_id"]] = result["answers_by_round"][round_key]
                
                # Majority vote for this round
                agent_answers = list(round_answers[result["question_id"]].values())
                agent_answers = [a for a in agent_answers if a is not None]
                if agent_answers:
                    mv_answer = Counter(agent_answers).most_common(1)[0][0]
                    if mv_answer == result["ground_truth"]:
                        round_correct += 1
                    
                    # Check full agreement
                    if len(set(agent_answers)) == 1:
                        full_agreement += 1
                        if agent_answers[0] != result["ground_truth"]:
                            full_agreement_wrong += 1
                
                # Answer distribution
                answer_dist = Counter(agent_answers)
                answer_distribution.append(sorted(answer_dist.values(), reverse=True))
            
            round_acc = round_correct / len(results) if results else 0.0
            full_agree_rate = full_agreement / len(results) if results else 0.0
            groupthink_rate = full_agreement_wrong / len(results) if results else 0.0
            
            metrics["rounds"][round_key] = {
                "majority_vote_accuracy": round_acc,
                "full_agreement_rate": full_agree_rate,
                "full_agreement_wrong_rate": groupthink_rate
            }
            
            # Answer change rate (only for rounds > 0)
            if round_num > 0:
                prev_round = str(round_num - 1)
                answer_changes = 0
                for result in results:
                    prev_answers = result["answers_by_round"][prev_round]
                    curr_answers = result["answers_by_round"][round_key]
                    prev_mv = None
                    curr_mv = None
                    
                    pa = [a for a in prev_answers.values() if a is not None]
                    if pa:
                        prev_mv = Counter(pa).most_common(1)[0][0]
                    
                    ca = [a for a in curr_answers.values() if a is not None]
                    if ca:
                        curr_mv = Counter(ca).most_common(1)[0][0]
                    
                    if prev_mv != curr_mv:
                        answer_changes += 1
                
                metrics["rounds"][round_key]["answer_change_rate"] = (
                    answer_changes / len(results) if results else 0.0
                )
            
            if verbose:
                print(f"  Round {round_num}:")
                print(f"    Majority vote accuracy: {round_acc:.2%}")
                print(f"    Full agreement rate: {full_agree_rate:.2%}")
                if full_agreement > 0:
                    print(f"    Groupthink (wrong agreement): {groupthink_rate:.2%}")
                if "answer_change_rate" in metrics["rounds"][round_key]:
                    print(f"    Answer change rate: {metrics['rounds'][round_key]['answer_change_rate']:.2%}")
    
    # Per-category accuracy
    category_stats: Dict[str, Dict[str, int]] = {}
    for result in results:
        cat = result.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        
        category_stats[cat]["total"] += 1
        if result["final_answer"] == result["ground_truth"]:
            category_stats[cat]["correct"] += 1
    
    for cat, stats in sorted(category_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        metrics["by_category"][cat] = {
            "accuracy": acc,
            "count": stats["total"]
        }
    
    if verbose and metrics["by_category"]:
        print("\n  Per-category accuracy:")
        for cat, stats in sorted(metrics["by_category"].items(), 
                                key=lambda x: x[1]["accuracy"], reverse=True):
            print(f"    {cat}: {stats['accuracy']:.2%} ({stats['count']} questions)")
    
    # Token usage
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    by_round_tokens = {}
    
    for result in results:
        for round_key, round_usage in result.get("token_usage", {}).get("by_round", {}).items():
            total_prompt += round_usage.get("prompt_tokens", 0)
            total_completion += round_usage.get("completion_tokens", 0)
            total_tokens += round_usage.get("total_tokens", 0)
            
            if round_key not in by_round_tokens:
                by_round_tokens[round_key] = {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "count": 0
                }
            by_round_tokens[round_key]["total_prompt_tokens"] += round_usage.get("prompt_tokens", 0)
            by_round_tokens[round_key]["total_completion_tokens"] += round_usage.get("completion_tokens", 0)
            by_round_tokens[round_key]["total_tokens"] += round_usage.get("total_tokens", 0)
            by_round_tokens[round_key]["count"] = len(results)
    
    metrics["token_usage"] = {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "avg_tokens_per_question": total_tokens / len(results) if results else 0,
        "by_round": by_round_tokens
    }
    
    if verbose:
        print(f"\n  Token Usage:")
        print(f"    Total tokens: {metrics['token_usage']['total_tokens']:,}")
        print(f"    Avg tokens/question: {metrics['token_usage']['avg_tokens_per_question']:.1f}")
        if by_round_tokens:
            print(f"    By round:")
            for rn, stats in sorted(by_round_tokens.items()):
                avg = stats["total_tokens"] / stats["count"] if stats["count"] > 0 else 0
                print(f"      Round {rn}: {stats['total_tokens']:,} tokens "
                      f"({avg:.1f} avg/q)")
    
    return metrics


def compare_modes(single_results: Dict[str, Any],
                  sc_results: Dict[str, Any],
                  debate_results: Dict[str, Any]) -> None:
    """
    Compare results across three experimental modes and print comparison table.
    
    Args:
        single_results: Results from E0 (single agent)
        sc_results: Results from E1 (self-consistency)
        debate_results: Results from E2 (chain debate)
    """
    single_acc = single_results["metrics"]["overall_accuracy"]
    sc_acc = sc_results["metrics"]["overall_accuracy"]
    debate_acc = debate_results["metrics"]["overall_accuracy"]
    
    single_tokens = single_results["metrics"]["token_usage"]["total_tokens"]
    sc_tokens = sc_results["metrics"]["token_usage"]["total_tokens"]
    debate_tokens = debate_results["metrics"]["token_usage"]["total_tokens"]
    
    single_avg = single_results["metrics"]["token_usage"]["avg_tokens_per_question"]
    sc_avg = sc_results["metrics"]["token_usage"]["avg_tokens_per_question"]
    debate_avg = debate_results["metrics"]["token_usage"]["avg_tokens_per_question"]
    
    print("\n" + "=" * 70)
    print("COMPARISON: Single vs SC vs Debate")
    print("=" * 70)
    print(f"{'Metric':<25} {'Single':>15} {'SC':>15} {'Debate':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {single_acc:>14.1%} {sc_acc:>14.1%} {debate_acc:>14.1%}")
    print(f"{'Total Tokens':<25} {single_tokens:>15,} {sc_tokens:>15,} {debate_tokens:>15,}")
    print(f"{'Avg Tokens/Question':<25} {single_avg:>15.1f} {sc_avg:>15.1f} {debate_avg:>15.1f}")
    print("=" * 70)
    
    # Relative improvements
    if single_acc > 0:
        sc_improve = (sc_acc - single_acc) / single_acc
        debate_improve = (debate_acc - single_acc) / single_acc
        print(f"\nAccuracy improvement over Single:")
        print(f"  SC:     {sc_improve:+.2%}")
        print(f"  Debate: {debate_improve:+.2%}")
    
    if single_tokens > 0:
        sc_cost = (sc_tokens - single_tokens) / single_tokens
        debate_cost = (debate_tokens - single_tokens) / single_tokens
        print(f"\nToken cost increase vs Single:")
        print(f"  SC:     {sc_cost:+.2%}")
        print(f"  Debate: {debate_cost:+.2%}")


def evaluate_mechanism_results(
    results_data: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate mechanism-mode results with variable debate lengths.

    This avoids the fixed-round assumptions in evaluate_results because
    mechanism questions may exit at different rounds.

    Args:
        results_data: Mechanism experiment results from run_experiment.py.
        verbose: Print detailed metrics.

    Returns:
        Metrics dict with baseline-style fields plus mechanism_metrics.
    """
    mode = results_data["config"]["mode"]
    results = results_data["results"]

    metrics = {
        "overall_accuracy": 0.0,
        "by_category": {},
        "token_usage": {},
        "mechanism_metrics": {}
    }

    correct = sum(1 for result in results
                  if result["final_answer"] == result["ground_truth"])
    metrics["overall_accuracy"] = correct / len(results) if results else 0.0

    category_stats: Dict[str, Dict[str, int]] = {}
    for result in results:
        category = result.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if result["final_answer"] == result["ground_truth"]:
            category_stats[category]["correct"] += 1

    for category, stats in sorted(category_stats.items()):
        metrics["by_category"][category] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            "count": stats["total"]
        }

    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    summarizer_total_tokens = 0
    summarizer_prompt_tokens = 0
    summarizer_completion_tokens = 0
    final_review_total_tokens = 0
    final_review_prompt_tokens = 0
    final_review_completion_tokens = 0
    evidence_verification_total_tokens = 0
    evidence_verification_prompt_tokens = 0
    evidence_verification_completion_tokens = 0
    dispute_resolver_total_tokens = 0
    dispute_resolver_prompt_tokens = 0
    dispute_resolver_completion_tokens = 0

    actual_rounds_sum = 0
    early_exit_count = 0
    early_exit_correct = 0
    exit_reason_distribution: Counter[str] = Counter()
    compression_distribution: Counter[str] = Counter()

    for result in results:
        token_usage = result.get("token_usage", {})
        total_with_summarizer = token_usage.get("total_with_summarizer")
        total = total_with_summarizer or token_usage.get("total", {})
        total_prompt += total.get("prompt_tokens", 0)
        total_completion += total.get("completion_tokens", 0)
        total_tokens += total.get("total_tokens", 0)

        summarizer_usage = token_usage.get("summarizer", {})
        summarizer_prompt_tokens += summarizer_usage.get("prompt_tokens", 0)
        summarizer_completion_tokens += summarizer_usage.get("completion_tokens", 0)
        summarizer_total_tokens += summarizer_usage.get("total_tokens", 0)

        final_review_usage = token_usage.get("final_review", {})
        final_review_prompt_tokens += final_review_usage.get("prompt_tokens", 0)
        final_review_completion_tokens += final_review_usage.get("completion_tokens", 0)
        final_review_total_tokens += final_review_usage.get("total_tokens", 0)

        evidence_usage = token_usage.get("evidence_verification", {})
        evidence_verification_prompt_tokens += evidence_usage.get("prompt_tokens", 0)
        evidence_verification_completion_tokens += evidence_usage.get(
            "completion_tokens",
            0,
        )
        evidence_verification_total_tokens += evidence_usage.get("total_tokens", 0)

        dispute_usage = token_usage.get("dispute_resolver", {})
        dispute_resolver_prompt_tokens += dispute_usage.get("prompt_tokens", 0)
        dispute_resolver_completion_tokens += dispute_usage.get(
            "completion_tokens",
            0,
        )
        dispute_resolver_total_tokens += dispute_usage.get("total_tokens", 0)

        mechanism = result.get("mechanism", {})
        actual_rounds = mechanism.get("actual_rounds", 0)
        actual_rounds_sum += actual_rounds
        exit_reason = mechanism.get("exit_reason", "unknown")
        exit_reason_distribution[exit_reason] += 1

        if mechanism.get("early_exit", False):
            early_exit_count += 1
            if result["final_answer"] == result["ground_truth"]:
                early_exit_correct += 1

        for log_item in mechanism.get("mechanism_log", []):
            level = log_item.get("compression_level")
            if level:
                compression_distribution[level] += 1

    metrics["token_usage"] = {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "avg_tokens_per_question": total_tokens / len(results) if results else 0,
        "summarizer_prompt_tokens": summarizer_prompt_tokens,
        "summarizer_completion_tokens": summarizer_completion_tokens,
        "summarizer_total_tokens": summarizer_total_tokens,
        "final_review_prompt_tokens": final_review_prompt_tokens,
        "final_review_completion_tokens": final_review_completion_tokens,
        "final_review_total_tokens": final_review_total_tokens,
        "evidence_verification_prompt_tokens": evidence_verification_prompt_tokens,
        "evidence_verification_completion_tokens": (
            evidence_verification_completion_tokens
        ),
        "evidence_verification_total_tokens": evidence_verification_total_tokens,
        "dispute_resolver_prompt_tokens": dispute_resolver_prompt_tokens,
        "dispute_resolver_completion_tokens": dispute_resolver_completion_tokens,
        "dispute_resolver_total_tokens": dispute_resolver_total_tokens,
    }

    metrics["mechanism_metrics"] = {
        "average_actual_rounds": actual_rounds_sum / len(results) if results else 0.0,
        "early_exit_count": early_exit_count,
        "early_exit_rate": early_exit_count / len(results) if results else 0.0,
        "early_exit_accuracy": (
            early_exit_correct / early_exit_count if early_exit_count else 0.0
        ),
        "exit_reason_distribution": dict(exit_reason_distribution),
        "compression_distribution": {
            key: compression_distribution.get(key, 0)
            for key in ("full", "partial", "none")
        },
        "summarizer_total_tokens": summarizer_total_tokens,
        "final_review_total_tokens": final_review_total_tokens,
        "evidence_verification_total_tokens": evidence_verification_total_tokens,
        "dispute_resolver_total_tokens": dispute_resolver_total_tokens,
    }

    if verbose:
        print(f"\n=== Evaluation Results ({mode.upper()}) ===")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%} ({correct}/{len(results)})")
        print("\n  Mechanism Metrics:")
        print(f"    Average actual rounds: {metrics['mechanism_metrics']['average_actual_rounds']:.2f}")
        print(f"    Early exit count: {early_exit_count}/{len(results)}")
        print(f"    Early exit rate: {metrics['mechanism_metrics']['early_exit_rate']:.2%}")
        print(f"    Early exit accuracy: {metrics['mechanism_metrics']['early_exit_accuracy']:.2%}")
        print(f"    Exit reasons: {metrics['mechanism_metrics']['exit_reason_distribution']}")
        print(f"    Compression distribution: {metrics['mechanism_metrics']['compression_distribution']}")
        print(f"    Summarizer tokens: {summarizer_total_tokens:,}")
        print(f"    Final review tokens: {final_review_total_tokens:,}")
        print(f"    Evidence verification tokens: {evidence_verification_total_tokens:,}")
        print(f"    Dispute resolver tokens: {dispute_resolver_total_tokens:,}")
        print(f"\n  Token Usage:")
        print(f"    Total tokens with summarizer: {total_tokens:,}")
        print(f"    Avg tokens/question: {metrics['token_usage']['avg_tokens_per_question']:.1f}")

        if metrics["by_category"]:
            print("\n  Per-category accuracy:")
            for category, stats in sorted(
                metrics["by_category"].items(),
                key=lambda item: item[1]["accuracy"],
                reverse=True,
            ):
                print(f"    {category}: {stats['accuracy']:.2%} ({stats['count']} questions)")

    return metrics
