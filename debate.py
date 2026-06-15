"""
Chain debate implementation.
Supports three modes: single (E0), self-consistency (E1), and chain debate (E2).
"""

import time
from typing import Dict, List, Any, Optional
from collections import Counter
from agent import (
    agent_initial_response,
    agent_debate_response,
    extract_or_repair_answer,
)
import config as _config
from config import NUM_AGENTS, NUM_ROUNDS, TEMPERATURE, DEBATE_TEMP, SLEEP_BETWEEN_AGENTS


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


def run_single(question: str, choices: str, verbose: bool = False) -> Dict[str, Any]:
    """
    E0 Baseline: Single agent (Qwen2.5-7B only).
    
    Only Agent 1 answers once.
    
    Args:
        question: Question text
        choices: Formatted choices
        verbose: Print detailed output
        
    Returns:
        dict: Result with history, answers by round, final answer, and token usage
    """
    agent_ids = list(range(1, NUM_AGENTS + 1))
    
    # Only Agent 1 responds
    history = {0: {}}
    usages = {0: {}}
    answers = {0: {}}
    
    if verbose:
        print("  [E0] Running single agent (Qwen2.5-7B only)...")
    
    try:
        text, usage = agent_initial_response(question, choices, 1, TEMPERATURE)
        history[0][1] = text
        usages[0][1] = usage
        answers[0][1] = _extract_answer_with_repair(text, choices, 1, usage)
    except Exception as e:
        # Agent 1 failed
        history[0][1] = f"[ERROR] {str(e)}"
        usages[0][1] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answers[0][1] = None
    
    final_answer = answers[0][1]
    
    # Token usage
    usage = usages[0][1]
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
        "answers_by_round": {"0": {"1": answers[0][1]}},
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_usage_by_round,
            "total": total_usage
        }
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
