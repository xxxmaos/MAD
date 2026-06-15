"""
Main experiment runner supporting three modes: single (E0), SC (E1), debate (E2).
Loads MMLU-Pro dataset and runs experiments with configurable parameters.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import config as _config
import debate as _debate_module
from config import (
    AGENT_CONFIGS, NUM_AGENTS, NUM_ROUNDS, TEMPERATURE, DEBATE_TEMP,
    NUM_SAMPLES, SEED, SLEEP_BETWEEN_QUESTIONS, MAX_DEBATE_ROUNDS
)
from data_loader import (
    load_mmlu_pro_dataset,
    load_multirc_dataset,
    load_quality_dataset,
    inspect_first_sample,
    sample_questions,
    format_choices,
)
from debate import run_single, run_sc, run_chain_debate
from debate_adaptive_resolver import run_debate_with_adaptive_resolver
from debate_with_mechanism import run_debate_with_mechanism
from evaluate import evaluate_results, evaluate_mechanism_results, compare_modes


MECHANISM_MODES = [
    "mechanism",
    "no_verdicts_stable",
    "no_verdicts_no_deadlock",
    "no_verdict_stable_no_deadlock",
    "adaptive_resolver",
    "adaptive_resolver_v2_all_stable_gate",
    "V3ResolverInfluenceGate",
    "V3MultiAnswerResolverInfluenceGate",
    "V4QuestionAwareEvidenceSnippets",
    "V42EmbeddingTopKSnippets",
    "V43SelectiveEmbeddingSnippets",
    "V5SelectiveReplacement",
    "compression_only_no_early_exit",
]


def apply_experiment_settings(agent_profile: str, prompt_style: str) -> None:
    """
    Apply runtime experiment switches without changing function interfaces.

    Agent configs are mutated in place so modules that imported AGENT_CONFIGS
    keep seeing the selected profile.
    """
    if agent_profile not in _config.AGENT_PROFILES:
        raise ValueError(f"Unknown agent profile: {agent_profile}")
    if prompt_style not in _config.VALID_DEBATE_PROMPT_STYLES:
        raise ValueError(f"Unknown debate prompt style: {prompt_style}")

    selected_agents = [
        cfg.copy() for cfg in _config.AGENT_PROFILES[agent_profile]
    ]
    _config.AGENT_CONFIGS[:] = selected_agents
    _config.AGENT_PROFILE = agent_profile
    _config.NUM_AGENTS = len(selected_agents)
    _debate_module.NUM_AGENTS = len(selected_agents)
    globals()["NUM_AGENTS"] = len(selected_agents)
    _config.DEBATE_PROMPT_STYLE = prompt_style


def prepare_dataset(dataset_name: str = "mmlu_pro", num_samples: Optional[int] = None):
    """
    Load and prepare a dataset.
    
    Args:
        dataset_name: "mmlu_pro" or "quality".
        num_samples: Optional sample count override.

    Returns:
        tuple: (full_dataset, sampled_questions)
    """
    if num_samples is None:
        num_samples = NUM_SAMPLES

    if dataset_name == "mmlu_pro":
        dataset = load_mmlu_pro_dataset()
    elif dataset_name == "quality":
        dataset = load_quality_dataset(split="validation")
    elif dataset_name == "multirc":
        dataset = load_multirc_dataset(split="validation")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    inspect_first_sample(dataset)
    
    sampled = sample_questions(dataset, num_samples=num_samples, seed=SEED)
    return dataset, sampled


def run_experiment_mode(
    mode: str,
    questions: List[Dict[str, Any]],
    dataset_name: str = "mmlu_pro",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run experiment for a specific mode.
    
    Args:
        mode: Experiment mode name.
        questions: Sampled questions.
        dataset_name: Dataset identifier.
        verbose: Print detailed output
        
    Returns:
        dict: Experiment results with config, metrics, and per-question results
    """
    valid_modes = ["single", "sc", "debate"] + MECHANISM_MODES
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of {valid_modes}"
        )
    
    mode_labels = {
        "single": 0,
        "sc": 1,
        "debate": 2,
        "mechanism": 3,
        "no_verdicts_stable": 4,
        "no_verdicts_no_deadlock": 5,
        "no_verdict_stable_no_deadlock": 6,
        "adaptive_resolver": 7,
        "adaptive_resolver_v2_all_stable_gate": 8,
        "V3ResolverInfluenceGate": 9,
        "V3MultiAnswerResolverInfluenceGate": 10,
        "V4QuestionAwareEvidenceSnippets": 11,
        "V42EmbeddingTopKSnippets": 12,
        "V43SelectiveEmbeddingSnippets": 13,
        "V5SelectiveReplacement": 14,
        "compression_only_no_early_exit": 15,
    }
    mode_num = mode_labels[mode]
    
    print(f"\n{'='*70}")
    print(f"Running E{mode_num} ({mode.upper()}) experiment")
    print(f"Mode: {mode}, Samples: {len(questions)}, Questions")
    print(
        f"Agent profile: {_config.AGENT_PROFILE}, "
        f"Prompt style: {_config.DEBATE_PROMPT_STYLE}"
    )
    print(f"{'='*70}\n")
    
    results_list: List[Dict[str, Any]] = []
    
    pbar = tqdm(questions, desc=f"E{mode_num} Progress")
    
    for q_idx, question_data in enumerate(pbar):
        try:
            question = question_data["question"]
            options = question_data["options"]
            ground_truth = question_data["answer"]
            category = question_data.get("category", "unknown")
            
            # Format choices
            choices = format_choices(options)
            
            # Verbose output for first 3 questions
            is_verbose = verbose or q_idx < 3
            
            # Run appropriate debate mode
            if mode == "single":
                result = run_single(question, choices, verbose=is_verbose)
            elif mode == "sc":
                result = run_sc(question, choices, verbose=is_verbose)
            elif mode == "debate":
                result = run_chain_debate(question, choices, verbose=is_verbose)
            elif mode in (
                "adaptive_resolver",
                "adaptive_resolver_v2_all_stable_gate",
                "V3ResolverInfluenceGate",
                "V3MultiAnswerResolverInfluenceGate",
                "V4QuestionAwareEvidenceSnippets",
                "V42EmbeddingTopKSnippets",
                "V43SelectiveEmbeddingSnippets",
                "V5SelectiveReplacement",
            ):
                result = run_debate_with_adaptive_resolver(
                    question,
                    choices,
                    num_options=len(options),
                    verbose=is_verbose,
                    enable_all_stable_safety_gate=(
                        mode in (
                            "adaptive_resolver_v2_all_stable_gate",
                            "V3ResolverInfluenceGate",
                            "V3MultiAnswerResolverInfluenceGate",
                            "V4QuestionAwareEvidenceSnippets",
                            "V42EmbeddingTopKSnippets",
                            "V43SelectiveEmbeddingSnippets",
                            "V5SelectiveReplacement",
                        )
                    ),
                    enable_resolver_influence_gate=(
                        mode in (
                            "V3ResolverInfluenceGate",
                            "V3MultiAnswerResolverInfluenceGate",
                            "V4QuestionAwareEvidenceSnippets",
                            "V42EmbeddingTopKSnippets",
                            "V43SelectiveEmbeddingSnippets",
                            "V5SelectiveReplacement",
                        )
                    ),
                    enable_question_aware_evidence_snippets=(
                        mode in (
                            "V4QuestionAwareEvidenceSnippets",
                            "V42EmbeddingTopKSnippets",
                            "V43SelectiveEmbeddingSnippets",
                            "V5SelectiveReplacement",
                        )
                    ),
                    evidence_snippet_method=(
                        "embedding"
                        if mode in (
                            "V42EmbeddingTopKSnippets",
                            "V43SelectiveEmbeddingSnippets",
                            "V5SelectiveReplacement",
                        )
                        else "lexical"
                    ),
                    evidence_snippet_policy=(
                        "selective"
                        if mode in ("V43SelectiveEmbeddingSnippets", "V5SelectiveReplacement")
                        else "always"
                    ),
                    enable_selective_passage_replacement=(
                        mode == "V5SelectiveReplacement"
                    ),
                    force_llm_summaries=(mode == "V5SelectiveReplacement"),
                    multi_answer=(mode == "V3MultiAnswerResolverInfluenceGate"),
                )
            else:  # mechanism variants
                mechanism_kwargs = {}
                if mode in (
                    "mechanism",
                    "no_verdicts_stable",
                    "no_verdicts_no_deadlock",
                    "no_verdict_stable_no_deadlock",
                ):
                    mechanism_kwargs["enable_verdicts_stable_exit"] = False
                if mode in (
                    "no_verdicts_no_deadlock",
                    "no_verdict_stable_no_deadlock",
                ):
                    mechanism_kwargs["enable_deadlock_exit"] = False
                elif mode == "compression_only_no_early_exit":
                    mechanism_kwargs["enable_early_exit"] = False
                result = run_debate_with_mechanism(
                    question,
                    choices,
                    num_options=len(options),
                    verbose=is_verbose,
                    **mechanism_kwargs,
                )
            
            # Package result
            result_item = {
                "question_id": q_idx,
                "question": question,
                "choices": [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)],
                "ground_truth": ground_truth,
                "category": category,
                "final_answer": result["final_answer"],
                "answers_by_round": result["answers_by_round"],
                "token_usage": result["token_usage"]
            }
            if mode in MECHANISM_MODES:
                result_item["mechanism"] = result.get("mechanism", {})
            
            results_list.append(result_item)
            
            time.sleep(SLEEP_BETWEEN_QUESTIONS)
            
        except Exception as e:
            print(f"\nError on question {q_idx}: {e}")
            result_item = {
                "question_id": q_idx,
                "question": question_data.get("question", ""),
                "choices": [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(question_data.get("options", []))],
                "ground_truth": question_data.get("answer", ""),
                "category": question_data.get("category", "unknown"),
                "final_answer": None,
                "answers_by_round": {},
                "token_usage": {"by_round": {}, "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                "error": str(e)
            }
            if mode in MECHANISM_MODES:
                result_item["mechanism"] = {
                    "actual_rounds": 0,
                    "early_exit": False,
                    "exit_reason": "error",
                    "mechanism_log": []
                }
            results_list.append(result_item)
    
    # Build config section
    config = {
        "mode": mode,
        "agents": [
            {"id": cfg["agent_id"], "model": cfg["model"], "name": cfg["name"]}
            for cfg in AGENT_CONFIGS
        ],
        "num_agents": NUM_AGENTS,
        "num_rounds": MAX_DEBATE_ROUNDS if mode in MECHANISM_MODES else (NUM_ROUNDS if mode == "debate" else 1),
        "topology": "chain" if mode in (["debate"] + MECHANISM_MODES) else "none",
        "dataset": dataset_name,
        "num_samples": len(questions),
        "seed": SEED,
        "agent_profile": _config.AGENT_PROFILE,
        "debate_prompt_style": _config.DEBATE_PROMPT_STYLE,
        "temperature": TEMPERATURE,
        "debate_temp": DEBATE_TEMP if mode in (["debate"] + MECHANISM_MODES) else TEMPERATURE
    }
    if mode in MECHANISM_MODES:
        config["mechanism_variant"] = {
            "enable_early_exit": mode != "compression_only_no_early_exit",
            "enable_verdicts_stable_exit": mode not in (
                "mechanism",
                "no_verdicts_stable",
                "no_verdicts_no_deadlock",
                "no_verdict_stable_no_deadlock",
                "adaptive_resolver",
                "adaptive_resolver_v2_all_stable_gate",
                "V3ResolverInfluenceGate",
                "V3MultiAnswerResolverInfluenceGate",
                "V4QuestionAwareEvidenceSnippets",
                "V42EmbeddingTopKSnippets",
                "V43SelectiveEmbeddingSnippets",
                "V5SelectiveReplacement",
            ),
            "enable_deadlock_exit": mode not in (
                "no_verdicts_no_deadlock",
                "no_verdict_stable_no_deadlock",
                "adaptive_resolver",
                "adaptive_resolver_v2_all_stable_gate",
                "V3ResolverInfluenceGate",
                "V3MultiAnswerResolverInfluenceGate",
                "V4QuestionAwareEvidenceSnippets",
                "V42EmbeddingTopKSnippets",
                "V43SelectiveEmbeddingSnippets",
                "V5SelectiveReplacement",
            ),
            "adaptive_resolver": mode in (
                "adaptive_resolver",
                "adaptive_resolver_v2_all_stable_gate",
                "V3ResolverInfluenceGate",
                "V4QuestionAwareEvidenceSnippets",
                "V42EmbeddingTopKSnippets",
                "V43SelectiveEmbeddingSnippets",
                "V5SelectiveReplacement",
            ),
            "adaptive_resolver_variant": (
                "V5.1 Structured Evidence Snippets"
                if mode == "V5SelectiveReplacement"
                else (
                    "V4.3 Selective Embedding Snippets"
                    if mode == "V43SelectiveEmbeddingSnippets"
                    else (
                        "V4.2 Embedding Top-K Snippets"
                        if mode == "V42EmbeddingTopKSnippets"
                        else (
                            "V4 Question-Aware Evidence Snippets"
                            if mode == "V4QuestionAwareEvidenceSnippets"
                            else (
                                (
                                    "V3 Multi-Answer Resolver Influence Gate"
                                    if mode == "V3MultiAnswerResolverInfluenceGate"
                                    else "V3 Resolver Influence Gate"
                                )
                                if mode in (
                                    "V3ResolverInfluenceGate",
                                    "V3MultiAnswerResolverInfluenceGate",
                                )
                                else (
                                    "V2 + all_stable safety gate"
                                    if mode == "adaptive_resolver_v2_all_stable_gate"
                                    else ("adaptive_resolver_v1" if mode == "adaptive_resolver" else None)
                                )
                            )
                        )
                    )
                )
            ),
            "question_aware_evidence_snippets": (
                mode in (
                    "V4QuestionAwareEvidenceSnippets",
                    "V42EmbeddingTopKSnippets",
                    "V43SelectiveEmbeddingSnippets",
                    "V5SelectiveReplacement",
                )
            ),
            "evidence_snippet_method": (
                "embedding"
                if mode in (
                    "V42EmbeddingTopKSnippets",
                    "V43SelectiveEmbeddingSnippets",
                    "V5SelectiveReplacement",
                )
                else ("lexical" if mode == "V4QuestionAwareEvidenceSnippets" else None)
            ),
            "evidence_snippet_policy": (
                "selective"
                if mode in ("V43SelectiveEmbeddingSnippets", "V5SelectiveReplacement")
                else None
            ),
            "selective_passage_replacement": mode == "V5SelectiveReplacement",
            "answer_mode": (
                "multi_answer_set"
                if mode == "V3MultiAnswerResolverInfluenceGate"
                else "single_answer"
            ),
            "evidence_snippet_strategy": (
                "multi_query_option_contrast_mmr_verification"
                if mode == "V5SelectiveReplacement"
                else None
            ),
            "force_llm_summaries": mode == "V5SelectiveReplacement",
            "compression_enabled": True,
        }
    
    # Evaluate results
    results_data = {
        "config": config,
        "results": results_list
    }
    
    # Determine number of rounds for evaluation
    num_rounds_for_eval = MAX_DEBATE_ROUNDS if mode in MECHANISM_MODES else (NUM_ROUNDS if mode == "debate" else 1)
    results_data["config"]["num_rounds"] = num_rounds_for_eval
    
    if mode in MECHANISM_MODES:
        metrics = evaluate_mechanism_results(results_data, verbose=True)
    else:
        metrics = evaluate_results(results_data, verbose=True)
    
    return {
        "config": config,
        "metrics": metrics,
        "results": results_list
    }


def save_results(results: Dict[str, Any], mode: str) -> str:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Experiment results dict
        mode: Experiment mode (single, sc, debate)
        
    Returns:
        str: Path to saved file
    """
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = results.get("config", {}).get("dataset", "mmlu_pro")
    num_agents = results.get("config", {}).get("num_agents", NUM_AGENTS)
    prefix = "" if dataset_name == "mmlu_pro" else f"{dataset_name}_"
    filename = (
        f"results/{prefix}{mode}_{num_agents}agent_"
        f"{results['config']['num_samples']}q_{timestamp}.json"
    )
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return filename


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate Experiment Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sc", "debate"] + MECHANISM_MODES + ["all"],
        default="single",
        help="Experiment mode: single, sc, debate, mechanism variants, or all"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for debugging"
    )
    parser.add_argument(
        "--dataset",
        choices=["mmlu_pro", "quality", "multirc"],
        default="mmlu_pro",
        help="Dataset to run: mmlu_pro, quality, or multirc"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of sampled questions"
    )
    parser.add_argument(
        "--agent-profile",
        choices=list(_config.AGENT_PROFILES.keys()),
        default=_config.AGENT_PROFILE,
        help="Agent model profile: default or strong"
    )
    parser.add_argument(
        "--prompt-style",
        choices=_config.VALID_DEBATE_PROMPT_STYLES,
        default=_config.DEBATE_PROMPT_STYLE,
        help="Debate prompt style for prompt ablations"
    )
    
    args = parser.parse_args()
    apply_experiment_settings(args.agent_profile, args.prompt_style)
    
    # Prepare dataset once
    print("\n" + "="*70)
    print(f"LOADING {args.dataset.upper()} DATASET")
    print(f"AGENT PROFILE: {args.agent_profile}")
    print(f"PROMPT STYLE: {args.prompt_style}")
    print("="*70)
    dataset, questions = prepare_dataset(args.dataset, args.num_samples)
    
    # Store results for comparison
    all_results = {}
    
    if args.mode == "all":
        modes = ["single", "sc", "debate", "mechanism"]
    else:
        modes = [args.mode]
    
    # Run experiments
    for mode in modes:
        results = run_experiment_mode(
            mode,
            questions,
            dataset_name=args.dataset,
            verbose=args.verbose,
        )
        all_results[mode] = results
        save_results(results, mode)
    
    # Comparison table if all modes run
    if args.mode == "all" and len(all_results) >= 3:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        compare_modes(all_results["single"], all_results["sc"], all_results["debate"])
        if "mechanism" in all_results:
            mech_acc = all_results["mechanism"]["metrics"]["overall_accuracy"]
            mech_metrics = all_results["mechanism"]["metrics"].get("mechanism_metrics", {})
            print(f"  Mechanism accuracy: {mech_acc:.2%}")
            print(f"  Mechanism avg rounds: {mech_metrics.get('average_actual_rounds', 0):.2f}")
            print(f"  Mechanism early exit rate: {mech_metrics.get('early_exit_rate', 0):.2%}")


if __name__ == "__main__":
    main()
