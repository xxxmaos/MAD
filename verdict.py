"""
Option-level verdict utilities for the compression and early-exit mechanism.

Each option is evaluated independently as include, exclude, or disputed based on
the current round's agent answers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from config import DISPUTED_PARTIAL_MAX


class OptionVerdict(TypedDict):
    """Per-option support and verdict information."""

    verdict: str
    support_count: int
    support_agents: List[int]
    oppose_count: int
    oppose_agents: List[int]


OptionLedger = Dict[str, OptionVerdict]


def compute_verdicts(
    answers_dict: Dict[int, Optional[str]],
    num_options: int = 10,
) -> OptionLedger:
    """
    Compute include/exclude/disputed verdicts for each answer option.

    Args:
        answers_dict: Mapping from agent id to that agent's answer letter.
        num_options: Number of options to evaluate, defaulting to MMLU-Pro's 10.

    Returns:
        Option ledger keyed by option label.
    """
    num_agents = len(answers_dict)
    option_labels = [chr(65 + i) for i in range(num_options)]

    support_counts = {opt: 0 for opt in option_labels}
    support_agents: Dict[str, List[int]] = {opt: [] for opt in option_labels}

    for agent_id, answer in answers_dict.items():
        if answer and answer in support_counts:
            support_counts[answer] += 1
            support_agents[answer].append(agent_id)

    option_ledger: OptionLedger = {}
    for opt in option_labels:
        support_count = support_counts[opt]
        if support_count == num_agents:
            verdict = "include"
        elif support_count == 0:
            verdict = "exclude"
        else:
            verdict = "disputed"

        option_ledger[opt] = {
            "verdict": verdict,
            "support_count": support_count,
            "support_agents": support_agents[opt],
            "oppose_count": num_agents - support_count,
            "oppose_agents": [
                agent_id
                for agent_id in answers_dict
                if agent_id not in support_agents[opt]
            ],
        }

    return option_ledger


def decide_compression_level(option_ledger: OptionLedger) -> str:
    """
    Decide compression level from the absolute count of disputed options.

    Returns:
        One of "full", "partial", or "none".
    """
    n_disputed = sum(
        1 for value in option_ledger.values() if value["verdict"] == "disputed"
    )

    if n_disputed == 0:
        return "full"
    if n_disputed <= DISPUTED_PARTIAL_MAX:
        return "partial"
    return "none"


def get_disputed_options(option_ledger: OptionLedger) -> List[str]:
    """
    Return option labels whose verdict is disputed.

    Args:
        option_ledger: Ledger returned by compute_verdicts.

    Returns:
        Sorted list of disputed option labels.
    """
    return [
        opt
        for opt, info in sorted(option_ledger.items())
        if info["verdict"] == "disputed"
    ]


def get_frozen_options(option_ledger: OptionLedger) -> List[str]:
    """
    Return option labels whose verdict is include or exclude.

    Args:
        option_ledger: Ledger returned by compute_verdicts.

    Returns:
        Sorted list of non-disputed option labels.
    """
    return [
        opt
        for opt, info in sorted(option_ledger.items())
        if info["verdict"] != "disputed"
    ]
