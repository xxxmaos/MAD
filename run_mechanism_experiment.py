"""
Experiment runner for option-level compression plus two-stage early exit.

This entry point is intentionally separate from run_experiment.py so baseline
experiments remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import config as _config
from config import (
    AGENT_CONFIGS,
    CONSECUTIVE_STABLE_NEEDED,
    DEADLOCK_PATIENCE,
    DEBATE_TEMP,
    DISPUTED_PARTIAL_MAX,
    EARLY_EXIT_R0,
    ENABLE_DEVILS_ADVOCATE,
    MAX_DEBATE_ROUNDS,
    MIN_DEBATE_ROUNDS,
    SEED,
    SLEEP_BETWEEN_QUESTIONS,
    SUMMARIZER_AGENT_ID,
    TEMPERATURE,
)
from data_loader import (
    format_choices,
    inspect_first_sample,
    load_mmlu_pro_dataset,
    sample_questions,
)
from debate_with_mechanism import run_debate_with_mechanism
from evaluate_mechanism import evaluate_mechanism_results, print_compare_table
from ollama_gpu_manager import configure_ollama_for_agents

DEFAULT_MECHANISM_SAMPLES = 300
DEFAULT_ABLATION_SAMPLES = 50
RESULT_DIR = "ablation_result"


def apply_agent_profile(agent_profile: str) -> None:
    """Apply an agent profile while preserving imported AGENT_CONFIGS refs."""
    if agent_profile not in _config.AGENT_PROFILES:
        raise ValueError(f"Unknown agent profile: {agent_profile}")
    selected_agents = [
        cfg.copy() for cfg in _config.AGENT_PROFILES[agent_profile]
    ]
    _config.AGENT_CONFIGS[:] = selected_agents
    _config.AGENT_PROFILE = agent_profile
    _config.NUM_AGENTS = len(selected_agents)


def prepare_dataset(num_samples: int = DEFAULT_MECHANISM_SAMPLES) -> List[Dict[str, Any]]:
    """
    Load and sample the MMLU-Pro dataset with the baseline seed.

    Args:
        num_samples: Number of questions to sample.

    Returns:
        Sampled question records.
    """
    dataset = load_mmlu_pro_dataset()
    inspect_first_sample(dataset)
    return sample_questions(dataset, num_samples=num_samples, seed=SEED)


def run_mechanism_mode(
    questions: List[Dict[str, Any]],
    verbose: bool = False,
    experiment_name: str = "mechanism",
    run_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the mechanism debate over sampled questions.

    Args:
        questions: Sampled question records.
        verbose: Whether to print detailed output for every question.
        experiment_name: Name recorded in config and progress labels.
        run_options: Runtime switches passed to run_debate_with_mechanism.

    Returns:
        Complete experiment result dict.
    """
    run_options = run_options or {}
    print("\n" + "=" * 70)
    print(f"Running {experiment_name} experiment")
    print(f"Mode: {experiment_name}, Samples: {len(questions)} Questions")
    print("=" * 70 + "\n")

    results_list: List[Dict[str, Any]] = []
    progress = tqdm(questions, desc=f"{experiment_name} Progress")

    for question_index, question_data in enumerate(progress):
        question = question_data.get("question", "")
        options = question_data.get("options", [])
        choices = format_choices(options)
        is_verbose = verbose or question_index < 3

        try:
            result = run_debate_with_mechanism(
                question,
                choices,
                verbose=is_verbose,
                **run_options,
            )
            result_item = {
                "question_id": question_index,
                "question": question,
                "choices": [
                    f"{chr(65 + idx)}. {option}"
                    for idx, option in enumerate(options)
                ],
                "ground_truth": question_data.get("answer", ""),
                "category": question_data.get("category", "unknown"),
                "final_answer": result["final_answer"],
                "answers_by_round": result["answers_by_round"],
                "token_usage": result["token_usage"],
                "mechanism": result["mechanism"],
            }
        except Exception as exc:
            print(f"\nError on question {question_index}: {exc}")
            result_item = {
                "question_id": question_index,
                "question": question,
                "choices": [
                    f"{chr(65 + idx)}. {option}"
                    for idx, option in enumerate(options)
                ],
                "ground_truth": question_data.get("answer", ""),
                "category": question_data.get("category", "unknown"),
                "final_answer": None,
                "answers_by_round": {},
                "token_usage": {
                    "by_round": {},
                    "total": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
                "mechanism": {
                    "actual_rounds": 0,
                    "early_exit": False,
                    "exit_reason": "error",
                    "mechanism_log": [],
                    "summarizer_token_usage": {
                        "by_round": {},
                        "total": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    },
                    "da_token_usage": {
                        "by_round": {},
                        "total": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    },
                    "total_token_usage_with_overhead": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
                "error": str(exc),
            }

        results_list.append(result_item)
        time.sleep(SLEEP_BETWEEN_QUESTIONS)

    config = _build_config(
        len(questions),
        mode=experiment_name,
        run_options=run_options,
    )
    results_data = {
        "config": config,
        "results": results_list,
    }
    baseline_debate = _load_latest_result("debate")
    metrics = evaluate_mechanism_results(
        results_data,
        baseline_debate=baseline_debate,
        verbose=True,
    )
    return {
        "config": config,
        "metrics": metrics,
        "results": results_list,
    }


def run_ablation_mode(
    questions: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run no-DA ablations over the same sampled question set.

    Args:
        questions: Sampled question records.
        verbose: Whether to print detailed output for every question.

    Returns:
        Aggregate ablation result dict keyed by variant name.
    """
    variants = [
        {
            "name": "no_da_default",
            "description": "No DA; keep LLM summarizer and all early-exit rules.",
            "run_options": {
                "use_summarizer": True,
                "enable_verdict_stable_exit": True,
            },
        },
        {
            "name": "no_da_capsule_only",
            "description": "No DA; replace LLM summarizer with zero-cost capsules.",
            "run_options": {
                "use_summarizer": False,
                "enable_verdict_stable_exit": True,
            },
        },
        {
            "name": "no_da_no_verdict_stable",
            "description": "No DA; disable verdicts_stable early exit.",
            "run_options": {
                "use_summarizer": True,
                "enable_verdict_stable_exit": False,
            },
        },
        {
            "name": "no_da_capsule_no_verdict_stable",
            "description": "No DA; capsule-only context and no verdicts_stable exit.",
            "run_options": {
                "use_summarizer": False,
                "enable_verdict_stable_exit": False,
            },
        },
    ]

    variant_results: Dict[str, Any] = {}
    for variant in variants:
        print("\n" + "#" * 70)
        print(f"ABLATION VARIANT: {variant['name']}")
        print(variant["description"])
        print("#" * 70)
        result = run_mechanism_mode(
            questions,
            verbose=verbose,
            experiment_name=variant["name"],
            run_options=variant["run_options"],
        )
        result["config"]["ablation_description"] = variant["description"]
        variant_results[variant["name"]] = result

    return {
        "config": {
            "mode": "ablation",
            "num_samples": len(questions),
            "seed": SEED,
            "dataset": "MMLU-Pro",
            "da_removed": not ENABLE_DEVILS_ADVOCATE,
            "variants": [
                {
                    "name": variant["name"],
                    "description": variant["description"],
                    "run_options": variant["run_options"],
                }
                for variant in variants
            ],
        },
        "variants": variant_results,
    }


def save_results(results: Dict[str, Any]) -> str:
    """
    Save mechanism results under ablation_result/.

    Args:
        results: Complete experiment result dict.

    Returns:
        Saved file path.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_samples = results["config"]["num_samples"]
    mode = results["config"].get("mode", "mechanism")
    filename = f"{RESULT_DIR}/{mode}_{num_samples}q_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filename}")
    return filename


def save_ablation_results(results: Dict[str, Any]) -> str:
    """
    Save aggregate ablation results under ablation_result/.

    Args:
        results: Aggregate ablation result dict.

    Returns:
        Saved file path.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_samples = results["config"]["num_samples"]
    filename = f"{RESULT_DIR}/ablation_{num_samples}q_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nAblation results saved to: {filename}")
    return filename


def compare_latest_results() -> None:
    """Load latest result JSONs for all four modes and print a comparison table."""
    single = _load_latest_result("single", required=True)
    sc = _load_latest_result("sc", required=True)
    debate = _load_latest_result("debate", required=True)
    mechanism = _load_latest_result("mechanism", required=True)

    if "mechanism_metrics" not in mechanism.get("metrics", {}):
        mechanism["metrics"] = evaluate_mechanism_results(
            mechanism,
            baseline_debate=debate,
            verbose=False,
        )

    print_compare_table(single, sc, debate, mechanism)


def _build_config(
    num_samples: int,
    mode: str = "mechanism",
    run_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the experiment config section."""
    run_options = run_options or {}
    summarizer_name = next(
        (
            cfg["name"]
            for cfg in AGENT_CONFIGS
            if cfg["agent_id"] == SUMMARIZER_AGENT_ID
        ),
        f"Agent {SUMMARIZER_AGENT_ID}",
    )
    return {
        "mode": mode,
        "agents": [
            {
                "id": cfg["agent_id"],
                "model": cfg["model"],
                "name": cfg["name"],
                "base_url": cfg.get("base_url"),
                "gpu_index": cfg.get("gpu_index"),
                "ollama_port": cfg.get("ollama_port"),
            }
            for cfg in AGENT_CONFIGS
        ],
        "num_agents": len(AGENT_CONFIGS),
        "num_rounds": MAX_DEBATE_ROUNDS,
        "topology": "chain",
        "num_samples": num_samples,
        "seed": SEED,
        "agent_profile": _config.AGENT_PROFILE,
        "temperature": TEMPERATURE,
        "debate_temp": DEBATE_TEMP,
        "ollama_gpu_assignment": _config.OLLAMA_GPU_ASSIGNMENT,
        "mechanism_config": {
            "disputed_partial_max": DISPUTED_PARTIAL_MAX,
            "early_exit_r0": EARLY_EXIT_R0,
            "min_debate_rounds": MIN_DEBATE_ROUNDS,
            "max_debate_rounds": MAX_DEBATE_ROUNDS,
            "consecutive_stable_needed": CONSECUTIVE_STABLE_NEEDED,
            "deadlock_patience": DEADLOCK_PATIENCE,
            "enable_da": ENABLE_DEVILS_ADVOCATE,
            "summarizer_agent": summarizer_name,
            "use_summarizer": run_options.get("use_summarizer", True),
            "enable_verdict_stable_exit": run_options.get(
                "enable_verdict_stable_exit",
                True,
            ),
        },
    }


def _load_latest_result(mode: str, required: bool = False) -> Optional[Dict[str, Any]]:
    """
    Load the newest result JSON for a mode from ablation_result/.

    Args:
        mode: Filename prefix, e.g. "single" or "mechanism".
        required: Whether to raise if no file is found.

    Returns:
        Parsed JSON dict, or None when optional and missing.
    """
    result_dir = Path(RESULT_DIR)
    files = sorted(
        result_dir.glob(f"{mode}_*q_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        if required:
            raise FileNotFoundError(
                f"No {RESULT_DIR}/{mode}_*q_*.json file found."
            )
        return None

    latest = files[0]
    print(f"Loaded latest {mode}: {latest}")
    with open(latest, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mechanism-enhanced multi-agent debate runner"
    )
    parser.add_argument(
        "--mode",
        choices=["mechanism", "compare", "ablation"],
        default="mechanism",
        help="Run mechanism experiment, no-DA ablation, or compare latest result files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of MMLU-Pro questions to sample.",
    )
    parser.add_argument(
        "--agent-profile",
        choices=list(_config.AGENT_PROFILES.keys()),
        default=_config.AGENT_PROFILE,
        help="Agent model profile.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for every question.",
    )
    parser.add_argument(
        "--no-auto-gpu",
        action="store_true",
        help="Disable automatic two-GPU Ollama routing.",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        compare_latest_results()
        return

    apply_agent_profile(args.agent_profile)
    configure_ollama_for_agents(
        _config.AGENT_CONFIGS,
        enabled=not args.no_auto_gpu,
    )

    print("\n" + "=" * 70)
    print("LOADING MMLU-PRO DATASET")
    print("=" * 70)
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = (
            DEFAULT_ABLATION_SAMPLES
            if args.mode == "ablation"
            else DEFAULT_MECHANISM_SAMPLES
        )
    questions = prepare_dataset(num_samples)
    if args.mode == "ablation":
        results = run_ablation_mode(questions, verbose=args.verbose)
        save_ablation_results(results)
    else:
        results = run_mechanism_mode(questions, verbose=args.verbose)
        save_results(results)


if __name__ == "__main__":
    main()
