"""
Chain debate implementation.
Supports three modes: single (E0), self-consistency (E1), and chain debate (E2).
"""

import time
from typing import Dict, List, Any, Optional
from collections import Counter
from agent import (
    agent_debate_response_multi_answer,
    agent_initial_response,
    agent_initial_response_multi_answer,
    agent_debate_response,
    extract_or_repair_answer_set,
    extract_or_repair_answer,
)
import config as _config
from config import NUM_AGENTS, NUM_ROUNDS, TEMPERATURE, DEBATE_TEMP, SLEEP_BETWEEN_AGENTS


_SINGLE_AGENT_MODEL_PRIORITY = {
    "mistral-small:24b": 100,
    "deepseek-r1:14b": 95,
    "phi4:14b": 90,
    "qwen3:14b": 88,
    "qwen3:8b": 78,
    "glm4:9b": 74,
    "llama3.1:8b": 70,
    "qwen2.5:7b": 66,
    "mistral:7b": 64,
    "gemma4:e4b-it-q4_K_M": 60,
    "command-r7b:7b": 58,
}


def _strongest_single_agent_id() -> int:
    """Select the strongest available agent in the active profile."""
    if not _config.AGENT_CONFIGS:
        return 1

    best_cfg = max(
        _config.AGENT_CONFIGS,
        key=lambda cfg: _SINGLE_AGENT_MODEL_PRIORITY.get(cfg.get("model", ""), 0),
    )
    return best_cfg.get("agent_id", 1)


def _build_debate_context(
    predecessor_response: str,
    agent_id: int,
    round_num: int,
    history: Dict[int, Dict[int, str]],
    answers: Dict[int, Dict[int, Optional[str]]],
) -> str:
    """
    Add own previous stance for commitment/evidence prompt ablations.

    The old prompt style returns the predecessor response unchanged, preserving
    the original baseline behavior.
    """
    if getattr(_config, "DEBATE_PROMPT_STYLE", "old") == "old":
        return predecessor_response

    own_previous_response = history.get(round_num - 1, {}).get(agent_id, "")
    own_previous_answer = answers.get(round_num - 1, {}).get(agent_id)

    return (
        f"[Your Previous Answer]\n{own_previous_answer or 'unknown'}\n\n"
        f"[Your Previous Reasoning]\n{own_previous_response or 'N/A'}\n\n"
        f"[Previous Speaker Response]\n{predecessor_response}"
    )


def _extract_answer_with_repair(
    text: str,
    choices: str,
    agent_id: int,
    usage: Dict[str, int],
) -> Optional[str]:
    """
    Extract an answer and add repair usage when format-only parsing fails.

    The repair call is only triggered if local extraction cannot find A-J.
    """
    answer, repair_usage, _ = extract_or_repair_answer(text, choices, agent_id)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        usage[key] = usage.get(key, 0) + repair_usage.get(key, 0)
    return answer


def _extract_answer_set_with_repair(
    text: str,
    choices: str,
    agent_id: int,
    usage: Dict[str, int],
) -> Optional[str]:
    """Extract a comma-separated answer set and add repair usage if enabled."""
    answer, repair_usage, _ = extract_or_repair_answer_set(text, choices, agent_id)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        usage[key] = usage.get(key, 0) + repair_usage.get(key, 0)
    return answer


def _canonical_answer_set_from_options(options: List[str]) -> Optional[str]:
    cleaned = sorted({opt.strip() for opt in options if opt and opt.strip()})
    return ",".join(cleaned) if cleaned else None


def _option_level_majority_answer_set(
    answer_sets: List[Optional[str]],
    num_agents: int,
) -> Optional[str]:
    threshold = (num_agents // 2) + 1
    counts: Counter[str] = Counter()
    for answer_set in answer_sets:
        if not answer_set:
            continue
        for opt in {part.strip() for part in answer_set.split(",") if part.strip()}:
            counts[opt] += 1

    included = [
        opt for opt, count in counts.items()
        if count >= threshold
    ]
    return _canonical_answer_set_from_options(included)


def run_single(question: str, choices: str, verbose: bool = False) -> Dict[str, Any]:
    """
    E0 Baseline: Single strongest agent in the active profile.
    
    Only the selected strongest agent answers once.
    
    Args:
        question: Question text
        choices: Formatted choices
        verbose: Print detailed output
        
    Returns:
        dict: Result with history, answers by round, final answer, and token usage
    """
    agent_ids = list(range(1, NUM_AGENTS + 1))
    single_agent_id = _strongest_single_agent_id()
    single_agent_name = next(
        (
            cfg.get("name", f"Agent {single_agent_id}")
            for cfg in _config.AGENT_CONFIGS
            if cfg.get("agent_id") == single_agent_id
        ),
        f"Agent {single_agent_id}",
    )
    
    # Only the strongest selected agent responds.
    history = {0: {}}
    usages = {0: {}}
    answers = {0: {}}
    
    if verbose:
        print(f"  [E0] Running single strongest agent ({single_agent_name})...")
    
    try:
        text, usage = agent_initial_response(question, choices, single_agent_id, TEMPERATURE)
        history[0][single_agent_id] = text
        usages[0][single_agent_id] = usage
        answers[0][single_agent_id] = _extract_answer_with_repair(
            text,
            choices,
            single_agent_id,
            usage,
        )
    except Exception as e:
        history[0][single_agent_id] = f"[ERROR] {str(e)}"
        usages[0][single_agent_id] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answers[0][single_agent_id] = None
    
    final_answer = answers[0][single_agent_id]
    
    # Token usage
    usage = usages[0][single_agent_id]
    token_usage_by_round = {
        "0": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"]
        }
    }
    
    total_usage = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"]
    }
    
    return {
        "history": history,
        "answers_by_round": {"0": {str(single_agent_id): answers[0][single_agent_id]}},
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_usage_by_round,
            "total": total_usage
        }
    }


def run_single_multi_answer(question: str, choices: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Multi-answer single baseline: one strongest agent answers once with an answer set.

    This is intended for MultiRC-style tasks. It does not change the original
    single-answer run_single baseline used by QuALITY or MMLU-Pro.
    """
    single_agent_id = _strongest_single_agent_id()
    single_agent_name = next(
        (
            cfg.get("name", f"Agent {single_agent_id}")
            for cfg in _config.AGENT_CONFIGS
            if cfg.get("agent_id") == single_agent_id
        ),
        f"Agent {single_agent_id}",
    )

    history = {0: {}}
    usages = {0: {}}
    answers = {0: {}}

    if verbose:
        print(f"  [E0M] Running single multi-answer agent ({single_agent_name})...")

    try:
        text, usage = agent_initial_response_multi_answer(
            question,
            choices,
            single_agent_id,
            TEMPERATURE,
        )
        history[0][single_agent_id] = text
        usages[0][single_agent_id] = usage
        answers[0][single_agent_id] = _extract_answer_set_with_repair(
            text,
            choices,
            single_agent_id,
            usage,
        )
    except Exception as e:
        history[0][single_agent_id] = f"[ERROR] {str(e)}"
        usages[0][single_agent_id] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answers[0][single_agent_id] = None

    final_answer = answers[0][single_agent_id]
    usage = usages[0][single_agent_id]
    token_usage_by_round = {
        "0": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    }
    total_usage = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }

    return {
        "history": history,
        "answers_by_round": {"0": {str(single_agent_id): final_answer}},
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_usage_by_round,
            "total": total_usage,
        },
    }


def run_sc(question: str, choices: str, verbose: bool = False) -> Dict[str, Any]:
    """
    E1 Self-Consistency: Three agents answer independently, majority vote.
    
    All three agents answer once without seeing each other's responses.
    Final answer is the majority vote.
    
    Args:
        question: Question text
        choices: Formatted choices
        verbose: Print detailed output
        
    Returns:
        dict: Result with history, answers by round, final answer, and token usage
    """
    agent_ids = list(range(1, NUM_AGENTS + 1))
    
    history = {0: {}}
    usages = {0: {}}
    answers = {0: {}}
    
    if verbose:
        print("  [E1] Running self-consistency (three agents, independent)...")
    
    for aid in agent_ids:
        try:
            text, usage = agent_initial_response(question, choices, aid, TEMPERATURE)
            history[0][aid] = text
            usages[0][aid] = usage
            answers[0][aid] = _extract_answer_with_repair(text, choices, aid, usage)
        except Exception as e:
            # Agent failed: record error but continue
            history[0][aid] = f"[ERROR] {str(e)}"
            usages[0][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            answers[0][aid] = None
        time.sleep(SLEEP_BETWEEN_AGENTS)
    
    # Majority vote
    final_answers_list = [answers[0][aid] for aid in agent_ids if answers[0][aid] is not None]
    if final_answers_list:
        final_answer = Counter(final_answers_list).most_common(1)[0][0]
    else:
        final_answer = None
    
    # Token usage
    token_usage_by_round = {"0": {
        "prompt_tokens": sum(usages[0][aid]["prompt_tokens"] for aid in agent_ids),
        "completion_tokens": sum(usages[0][aid]["completion_tokens"] for aid in agent_ids),
        "total_tokens": sum(usages[0][aid]["total_tokens"] for aid in agent_ids)
    }}
    
    total_usage = token_usage_by_round["0"].copy()
    
    return {
        "history": history,
        "answers_by_round": {"0": {str(aid): answers[0][aid] for aid in agent_ids}},
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_usage_by_round,
            "total": total_usage
        }
    }


def run_sc_multi_answer(question: str, choices: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Multi-answer self-consistency baseline for MultiRC-style tasks.

    Agents answer independently with answer sets. The final answer includes
    each option selected by a majority of agents.
    """
    agent_ids = list(range(1, NUM_AGENTS + 1))

    history = {0: {}}
    usages = {0: {}}
    answers = {0: {}}

    if verbose:
        print("  [E1-MA] Running multi-answer self-consistency...")

    for aid in agent_ids:
        try:
            text, usage = agent_initial_response_multi_answer(
                question,
                choices,
                aid,
                TEMPERATURE,
            )
            history[0][aid] = text
            usages[0][aid] = usage
            answers[0][aid] = _extract_answer_set_with_repair(
                text,
                choices,
                aid,
                usage,
            )
        except Exception as e:
            history[0][aid] = f"[ERROR] {str(e)}"
            usages[0][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            answers[0][aid] = None
        time.sleep(SLEEP_BETWEEN_AGENTS)

    final_answer = _option_level_majority_answer_set(
        [answers[0][aid] for aid in agent_ids],
        len(agent_ids),
    )

    token_usage_by_round = {"0": {
        "prompt_tokens": sum(usages[0][aid]["prompt_tokens"] for aid in agent_ids),
        "completion_tokens": sum(usages[0][aid]["completion_tokens"] for aid in agent_ids),
        "total_tokens": sum(usages[0][aid]["total_tokens"] for aid in agent_ids),
    }}
    total_usage = token_usage_by_round["0"].copy()

    return {
        "history": history,
        "answers_by_round": {"0": {str(aid): answers[0][aid] for aid in agent_ids}},
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_usage_by_round,
            "total": total_usage,
        },
    }


def run_chain_debate(question: str, choices: str, verbose: bool = False) -> Dict[str, Any]:
    """
    E2 Chain Debate: Three agents debate for NUM_ROUNDS with chain topology.
    
    Chain topology: Agent 1 <- Agent 3 (circular) <- Agent 2 <- Agent 1
    
    Round 0: All agents answer independently.
    Rounds 1+: Agents answer sequentially, each reading the previous agent's response.
    
    Args:
        question: Question text
        choices: Formatted choices
        verbose: Print detailed output
        
    Returns:
        dict: Result with history, answers by round, final answer, and token usage
    """
    agent_ids = list(range(1, NUM_AGENTS + 1))
    
    history: Dict[int, Dict[int, str]] = {}
    usages: Dict[int, Dict[int, Dict[str, int]]] = {}
    answers: Dict[int, Dict[int, Optional[str]]] = {}
    
    if verbose:
        print("  [E2] Running chain debate...")
    
    # Round 0: Independent responses
    if verbose:
        print(f"    Round 0 (independent)...")
    
    history[0] = {}
    usages[0] = {}
    answers[0] = {}
    for aid in agent_ids:
        try:
            text, usage = agent_initial_response(question, choices, aid, TEMPERATURE)
            history[0][aid] = text
            usages[0][aid] = usage
            answers[0][aid] = _extract_answer_with_repair(text, choices, aid, usage)
        except Exception as e:
            # Agent failed: record error but continue
            history[0][aid] = f"[ERROR] {str(e)}"
            usages[0][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            answers[0][aid] = None
        time.sleep(SLEEP_BETWEEN_AGENTS)
    
    # Rounds 1 to NUM_ROUNDS: Chain debate
    for round_num in range(1, NUM_ROUNDS + 1):
        if verbose:
            print(f"    Round {round_num} (chain)...")
        
        history[round_num] = {}
        usages[round_num] = {}
        answers[round_num] = {}
        
        for idx, aid in enumerate(agent_ids):
            if idx == 0:
                # Agent 1 reads Agent 3's previous round response (circular)
                pred_id = agent_ids[-1]
                if pred_id not in history[round_num - 1]:
                    # Predecessor had error, skip this agent
                    history[round_num][aid] = "[ERROR] Predecessor failed"
                    usages[round_num][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    answers[round_num][aid] = None
                    continue
                pred_response = history[round_num - 1][pred_id]
            else:
                # Other agents read current round's previous agent (chain)
                pred_id = agent_ids[idx - 1]
                if pred_id not in history[round_num]:
                    # Predecessor had error, skip this agent
                    history[round_num][aid] = "[ERROR] Predecessor failed"
                    usages[round_num][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    answers[round_num][aid] = None
                    continue
                pred_response = history[round_num][pred_id]
            
            debate_context = _build_debate_context(
                pred_response,
                aid,
                round_num,
                history,
                answers,
            )

            try:
                text, usage = agent_debate_response(
                    question, choices, aid,
                    pred_id, debate_context,
                    round_num, DEBATE_TEMP
                )
                history[round_num][aid] = text
                usages[round_num][aid] = usage
                answers[round_num][aid] = _extract_answer_with_repair(
                    text,
                    choices,
                    aid,
                    usage,
                )
            except Exception as e:
                # Agent failed: record error but continue
                history[round_num][aid] = f"[ERROR] {str(e)}"
                usages[round_num][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                answers[round_num][aid] = None
            time.sleep(SLEEP_BETWEEN_AGENTS)
    
    # Final majority vote (from last round)
    final_answers_list = [answers[NUM_ROUNDS][aid] for aid in agent_ids
                          if answers[NUM_ROUNDS][aid] is not None]
    if final_answers_list:
        final_answer = Counter(final_answers_list).most_common(1)[0][0]
    else:
        final_answer = None
    
    # Token usage summary
    token_usage_by_round: Dict[str, Dict[str, int]] = {}
    for rn in range(NUM_ROUNDS + 1):
        round_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        for aid in agent_ids:
            u = usages[rn][aid]
            round_total["prompt_tokens"] += u["prompt_tokens"]
            round_total["completion_tokens"] += u["completion_tokens"]
            round_total["total_tokens"] += u["total_tokens"]
        token_usage_by_round[str(rn)] = round_total
    
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    for rn_usage in token_usage_by_round.values():
        for k in total_usage:
            total_usage[k] += rn_usage[k]
    
    return {
        "history": history,
        "answers_by_round": {
            str(rn): {str(aid): answers[rn][aid] for aid in agent_ids}
            for rn in range(NUM_ROUNDS + 1)
        },
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_usage_by_round,
            "total": total_usage
        }
    }


def run_chain_debate_multi_answer(
    question: str,
    choices: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Multi-answer chain debate baseline for datasets like MultiRC.

    This intentionally mirrors the ordinary debate baseline: no compression,
    no resolver, no early exit, and fixed NUM_ROUNDS. The only difference is
    that each agent outputs a canonical answer set such as A,C.
    """
    agent_ids = list(range(1, NUM_AGENTS + 1))

    history: Dict[int, Dict[int, str]] = {}
    usages: Dict[int, Dict[int, Dict[str, int]]] = {}
    answers: Dict[int, Dict[int, Optional[str]]] = {}

    if verbose:
        print("  [E2-MA] Running multi-answer chain debate...")

    history[0] = {}
    usages[0] = {}
    answers[0] = {}
    if verbose:
        print("    Round 0 (independent, multi-answer)...")

    for aid in agent_ids:
        try:
            text, usage = agent_initial_response_multi_answer(
                question,
                choices,
                aid,
                TEMPERATURE,
            )
            history[0][aid] = text
            usages[0][aid] = usage
            answers[0][aid] = _extract_answer_set_with_repair(
                text,
                choices,
                aid,
                usage,
            )
        except Exception as e:
            history[0][aid] = f"[ERROR] {str(e)}"
            usages[0][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            answers[0][aid] = None
        time.sleep(SLEEP_BETWEEN_AGENTS)

    for round_num in range(1, NUM_ROUNDS + 1):
        if verbose:
            print(f"    Round {round_num} (chain, multi-answer)...")

        history[round_num] = {}
        usages[round_num] = {}
        answers[round_num] = {}

        for idx, aid in enumerate(agent_ids):
            if idx == 0:
                pred_id = agent_ids[-1]
                if pred_id not in history[round_num - 1]:
                    history[round_num][aid] = "[ERROR] Predecessor failed"
                    usages[round_num][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    answers[round_num][aid] = None
                    continue
                pred_response = history[round_num - 1][pred_id]
            else:
                pred_id = agent_ids[idx - 1]
                if pred_id not in history[round_num]:
                    history[round_num][aid] = "[ERROR] Predecessor failed"
                    usages[round_num][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    answers[round_num][aid] = None
                    continue
                pred_response = history[round_num][pred_id]

            debate_context = _build_debate_context(
                pred_response,
                aid,
                round_num,
                history,
                answers,
            )

            try:
                text, usage = agent_debate_response_multi_answer(
                    question,
                    choices,
                    aid,
                    pred_id,
                    debate_context,
                    round_num,
                    DEBATE_TEMP,
                )
                history[round_num][aid] = text
                usages[round_num][aid] = usage
                answers[round_num][aid] = _extract_answer_set_with_repair(
                    text,
                    choices,
                    aid,
                    usage,
                )
            except Exception as e:
                history[round_num][aid] = f"[ERROR] {str(e)}"
                usages[round_num][aid] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                answers[round_num][aid] = None
            time.sleep(SLEEP_BETWEEN_AGENTS)

    final_answers_list = [
        answers[NUM_ROUNDS][aid]
        for aid in agent_ids
        if answers[NUM_ROUNDS][aid] is not None
    ]
    final_answer = (
        Counter(final_answers_list).most_common(1)[0][0]
        if final_answers_list
        else None
    )

    token_usage_by_round: Dict[str, Dict[str, int]] = {}
    for rn in range(NUM_ROUNDS + 1):
        round_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for aid in agent_ids:
            u = usages[rn][aid]
            round_total["prompt_tokens"] += u["prompt_tokens"]
            round_total["completion_tokens"] += u["completion_tokens"]
            round_total["total_tokens"] += u["total_tokens"]
        token_usage_by_round[str(rn)] = round_total

    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for rn_usage in token_usage_by_round.values():
        for key in total_usage:
            total_usage[key] += rn_usage[key]

    return {
        "history": history,
        "answers_by_round": {
            str(rn): {str(aid): answers[rn][aid] for aid in agent_ids}
            for rn in range(NUM_ROUNDS + 1)
        },
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_usage_by_round,
            "total": total_usage,
        },
    }
