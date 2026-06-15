"""
Two-stage early-exit tracker for option-level debate compression.

Phase zero exits immediately after R0 if all agents agree. Phase one evaluates
stability, unchanged verdicts, and deadlock conditions during debate rounds.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from config import (
    CONSECUTIVE_STABLE_NEEDED,
    DEADLOCK_PATIENCE,
    ENABLE_DEVILS_ADVOCATE,
    MIN_DEBATE_ROUNDS,
)
from verdict import OptionLedger


class EarlyExitTracker:
    """Track verdict history and decide early exits."""

    def __init__(self, enable_verdict_stable_exit: bool = True) -> None:
        """
        Initialize empty tracker state.

        Args:
            enable_verdict_stable_exit: Whether unchanged verdict labels may
                trigger phase-one early exit.
        """
        self.verdict_history: List[OptionLedger] = []
        self.compression_history: List[str] = []
        self.consecutive_none = 0
        self.first_full_done = False
        self.enable_verdict_stable_exit = enable_verdict_stable_exit

    def check_phase_zero(
        self,
        answers_dict: Dict[int, Optional[str]],
    ) -> Tuple[str, Optional[str]]:
        """
        Exit after R0 only when every agent produced the same valid answer.

        Args:
            answers_dict: Current answers keyed by agent id.

        Returns:
            ("exit", final_answer) or ("continue", None).
        """
        answers = list(answers_dict.values())
        valid = [answer for answer in answers if answer is not None]
        if len(valid) == len(answers) and len(set(valid)) == 1:
            return ("exit", valid[0])
        return ("continue", None)

    def check_phase_one(
        self,
        current_round: int,
        option_ledger: OptionLedger,
        compression_level: str,
        delta: Dict[str, object],
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluate debate-round early-exit conditions.

        Args:
            current_round: Current debate round number, starting at 1.
            option_ledger: Current option ledger.
            compression_level: Current compression level.
            delta: Summary delta metadata for this round.

        Returns:
            ("exit", reason) or ("continue", None).
        """
        self.verdict_history.append(option_ledger)
        self.compression_history.append(compression_level)

        if compression_level == "none":
            self.consecutive_none += 1
        else:
            self.consecutive_none = 0

        if current_round < MIN_DEBATE_ROUNDS:
            return ("continue", None)

        n_disputed = sum(
            1 for value in option_ledger.values() if value["verdict"] == "disputed"
        )
        if n_disputed == 0:
            return ("exit", "all_options_stable")

        if self.enable_verdict_stable_exit and self._recent_verdicts_stable():
            return ("exit", "verdicts_stable")

        if self._recent_disputed_counts_stable(option_ledger):
            no_new_reasons = int(delta.get("new_reasons", 0) or 0) == 0
            if no_new_reasons:
                return ("exit", "deadlock")

        return ("continue", None)

    def should_trigger_dispute_summary(self) -> bool:
        """
        Return whether repeated NONE rounds should produce a dispute summary.

        Returns:
            True when consecutive NONE count reaches the configured patience.
        """
        return self.consecutive_none >= DEADLOCK_PATIENCE

    def should_trigger_da(self, compression_level: str) -> bool:
        """
        Return whether Devil's Advocate should run for this compression level.

        Args:
            compression_level: Current compression level.

        Returns:
            True only on the first FULL level if enabled.
        """
        return (
            ENABLE_DEVILS_ADVOCATE
            and compression_level == "full"
            and not self.first_full_done
        )

    def mark_da_done(self) -> None:
        """Mark the first-full Devil's Advocate opportunity as consumed."""
        self.first_full_done = True

    def _recent_verdicts_stable(self) -> bool:
        """Check whether recent verdict labels are unchanged."""
        needed = CONSECUTIVE_STABLE_NEEDED
        if len(self.verdict_history) < needed + 1:
            return False

        recent = self.verdict_history[-(needed + 1):]
        for prev, curr in zip(recent, recent[1:]):
            prev_verdicts = {key: value["verdict"] for key, value in prev.items()}
            curr_verdicts = {key: value["verdict"] for key, value in curr.items()}
            if prev_verdicts != curr_verdicts:
                return False
        return True

    def _recent_disputed_counts_stable(self, option_ledger: OptionLedger) -> bool:
        """Check whether disputed-option support counts are unchanged recently."""
        needed = CONSECUTIVE_STABLE_NEEDED
        if len(self.verdict_history) < needed + 1:
            return False

        disputed_options = [
            opt for opt, info in option_ledger.items() if info["verdict"] == "disputed"
        ]
        if not disputed_options:
            return False

        recent = self.verdict_history[-(needed + 1):]
        for prev, curr in zip(recent, recent[1:]):
            for opt in disputed_options:
                prev_count = prev.get(opt, {}).get("support_count", -1)
                curr_count = curr.get(opt, {}).get("support_count", -1)
                if prev_count != curr_count:
                    return False
        return True
