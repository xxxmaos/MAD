"""
Adaptive dispute-resolver debate loop.

This module keeps the baseline and prior mechanism loops intact. It implements
an accuracy-oriented variant:

- all_stable uses a V2 safety gate before another summarizer call
- disputed states do not use verdicts_stable or deadlock exits
- answer-distribution stability is measured with entropy, margin, and
  total-variation distance
- the first R2+ disputed state triggers a Dispute Resolver for the next round
- resolver-confirmed early exit is allowed only when the next round confirms
  the resolver answer with sufficient margin and low entropy
- unsafe all_stable states use a lightweight groupthink challenge
- max_round still uses final evidence review
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import config as _config
import requests
from agent import (
    agent_debate_response_multi_answer,
    agent_debate_response,
    agent_initial_response_multi_answer,
    agent_initial_response,
    extract_or_repair_answer_set,
    extract_or_repair_answer,
    get_client,
)
from config import (
    AGENT_CONFIGS,
    DEBATE_TEMP,
    MAX_DEBATE_ROUNDS,
    MAX_RETRIES,
    MIN_DEBATE_ROUNDS,
    NUM_OPTIONS,
    RETRY_DELAY,
    SUMMARIZER_AGENT_ID,
    SUMMARIZER_MAX_TOKENS,
    SUMMARIZER_TEMPERATURE,
    TEMPERATURE,
)
from debate_with_mechanism import (
    OptionLedger,
    Usage,
    _add_usage,
    _attach_own_previous_context,
    _build_answer_repair_entry,
    _count_disputed,
    _format_option_standings,
    _format_recent_responses,
    _format_recent_votes,
    _get_final_answer,
    _zero_usage,
    call_summarizer,
    compute_verdicts,
    decide_compression_level,
    format_summary_for_prompt,
    generate_summary,
    parse_summary_json,
)

ADAPTIVE_MIN_MARGIN = 2
ADAPTIVE_MAX_NORMALIZED_ENTROPY = 0.55
ADAPTIVE_MAX_TV_DISTANCE = 0.20
CALIBRATED_HIGH_THRESHOLD = 0.75
CALIBRATED_MEDIUM_THRESHOLD = 0.55
ADAPTIVE_RESOLVER_VARIANT = "V2 + all_stable safety gate"
ADAPTIVE_RESOLVER_V3_VARIANT = "V3 Resolver Influence Gate"
ADAPTIVE_RESOLVER_V4_VARIANT = "V4 Question-Aware Evidence Snippets"
ADAPTIVE_RESOLVER_V42_VARIANT = "V4.2 Embedding Top-K Snippets"
ADAPTIVE_RESOLVER_V43_VARIANT = "V4.3 Selective Embedding Snippets"
ADAPTIVE_RESOLVER_V5_VARIANT = "V5.1 Structured Evidence Snippets"
SELECTIVE_EVIDENCE_HIGH_ENTROPY = 0.70
V5_REPLACEMENT_MIN_SNIPPET_SCORE = 0.35
V5_STRUCTURED_CANDIDATE_TOP_K = 12
V5_STRUCTURED_FINAL_TOP_K = 4
V5_STRUCTURED_CONTRAST_TOP_K = 2
V5_MMR_LAMBDA = 0.70
_EMBEDDING_CACHE: Dict[Tuple[str, str], List[float]] = {}


def run_debate_with_adaptive_resolver(
    question: str,
    choices: str,
    num_options: Optional[int] = None,
    verbose: bool = False,
    enable_all_stable_safety_gate: bool = True,
    enable_resolver_influence_gate: bool = False,
    enable_question_aware_evidence_snippets: bool = False,
    evidence_snippet_method: str = "lexical",
    evidence_snippet_policy: str = "always",
    enable_selective_passage_replacement: bool = False,
    force_llm_summaries: bool = False,
    multi_answer: bool = False,
) -> Dict[str, Any]:
    """
    Run debate with adaptive disputed-state handling.

    The loop intentionally disables verdicts_stable and deadlock exits. The
    first R2+ disputed state calls a Dispute Resolver and injects that analysis
    into the next debate round. The resolver cannot decide directly; the next
    agent round must confirm its recommendation before early exit is allowed.
    """
    if num_options is None:
        num_options = NUM_OPTIONS

    if enable_selective_passage_replacement:
        variant_name = ADAPTIVE_RESOLVER_V5_VARIANT
    elif (
        enable_question_aware_evidence_snippets
        and evidence_snippet_method == "embedding"
        and evidence_snippet_policy == "selective"
    ):
        variant_name = ADAPTIVE_RESOLVER_V43_VARIANT
    elif enable_question_aware_evidence_snippets and evidence_snippet_method == "embedding":
        variant_name = ADAPTIVE_RESOLVER_V42_VARIANT
    elif enable_question_aware_evidence_snippets:
        variant_name = ADAPTIVE_RESOLVER_V4_VARIANT
    elif enable_resolver_influence_gate:
        variant_name = (
            "V3 Multi-Answer Resolver Influence Gate"
            if multi_answer
            else ADAPTIVE_RESOLVER_V3_VARIANT
        )
    elif enable_all_stable_safety_gate:
        variant_name = ADAPTIVE_RESOLVER_VARIANT
    else:
        variant_name = "adaptive_resolver_v1"
    agent_ids = [cfg["agent_id"] for cfg in AGENT_CONFIGS]
    history: Dict[int, Dict[int, str]] = {0: {}}
    usages: Dict[int, Dict[int, Usage]] = {0: {}}
    answers: Dict[int, Dict[int, Optional[str]]] = {0: {}}
    verdict_history: List[OptionLedger] = []
    mechanism_log: List[Dict[str, Any]] = []
    answer_repair_log: Dict[str, List[Dict[str, Any]]] = {}
    summarizer_total_usage = _zero_usage()
    dispute_resolver_total_usage = _zero_usage()

    if verbose:
        print("  [ADAPTIVE] Round 0 (independent, no early exit)...")

    for agent_id in agent_ids:
        try:
            if multi_answer:
                text, usage = agent_initial_response_multi_answer(
                    question,
                    choices,
                    agent_id,
                    TEMPERATURE,
                )
            else:
                text, usage = agent_initial_response(
                    question,
                    choices,
                    agent_id,
                    TEMPERATURE,
                )
        except Exception as exc:
            text = f"[ERROR] {exc}"
            usage = _zero_usage()
        history[0][agent_id] = text
        if multi_answer:
            answer, repair_usage, repair_text = extract_or_repair_answer_set(
                text,
                choices,
                agent_id,
            )
        else:
            answer, repair_usage, repair_text = extract_or_repair_answer(
                text,
                choices,
                agent_id,
            )
        if repair_usage.get("total_tokens", 0) > 0:
            _add_usage(usage, repair_usage)
            answer_repair_log.setdefault("0", []).append(
                _build_answer_repair_entry(
                    round_num=0,
                    agent_id=agent_id,
                    answer=answer,
                    original_response=text,
                    repair_response=repair_text,
                    repair_usage=repair_usage,
                )
            )
        usages[0][agent_id] = usage
        answers[0][agent_id] = answer
        time.sleep(0.5)

    if verbose:
        print(
            "  R0: {"
            + ", ".join(f"{agent_id}:{answers[0][agent_id]}" for agent_id in agent_ids)
            + "}"
        )

    current_ledger = compute_verdicts(
        {str(agent_id): answers[0][agent_id] for agent_id in agent_ids},
        num_options=num_options,
        multi_answer_majority=multi_answer,
    )
    verdict_history.append(current_ledger)

    prev_summary: Optional[Dict[str, Any]] = None
    next_resolver_context: Optional[str] = None
    next_resolver_analysis: Optional[Dict[str, Any]] = None
    next_resolver_trigger_snapshot: Optional[Dict[str, Any]] = None
    next_evidence_context: Optional[str] = None
    next_replacement_question: Optional[str] = None
    resolver_medium_confirmed_once = False
    resolver_trigger_count = 0
    option_labels = [chr(65 + index) for index in range(num_options)]

    for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
        active_resolver_context = next_resolver_context
        active_resolver_analysis = next_resolver_analysis
        active_resolver_trigger_snapshot = next_resolver_trigger_snapshot
        active_evidence_context = next_evidence_context
        active_replacement_question = next_replacement_question
        next_resolver_context = None
        next_resolver_analysis = None
        next_resolver_trigger_snapshot = None
        next_evidence_context = None
        next_replacement_question = None
        history[round_num] = {}
        usages[round_num] = {}
        answers[round_num] = {}

        for idx, agent_id in enumerate(agent_ids):
            if idx == 0:
                predecessor_id = agent_ids[-1]
                predecessor_response = history[round_num - 1][predecessor_id]
            else:
                predecessor_id = agent_ids[idx - 1]
                predecessor_response = history[round_num][predecessor_id]

            if prev_summary:
                enhanced_context = format_summary_for_prompt(prev_summary)
                if prev_summary.get("compression_level") == "none":
                    enhanced_context = (
                        f"{enhanced_context}\n\n"
                        f"[Previous Agent {predecessor_id} Response]\n"
                        f"{predecessor_response}"
                    )
            else:
                enhanced_context = predecessor_response

            if active_resolver_context:
                enhanced_context = (
                    f"{enhanced_context}\n\n"
                    "[Dispute Resolver Analysis]\n"
                    f"{active_resolver_context}"
                )

            if active_evidence_context:
                enhanced_context = (
                    f"{enhanced_context}\n\n"
                    "[Question-Aware Evidence Snippets]\n"
                    f"{active_evidence_context}"
                )

            enhanced_context = _attach_own_previous_context(
                enhanced_context,
                agent_id,
                round_num,
                history,
                answers,
            )

            try:
                agent_question = active_replacement_question or question
                if multi_answer:
                    text, usage = agent_debate_response_multi_answer(
                        agent_question,
                        choices,
                        agent_id,
                        predecessor_id,
                        enhanced_context,
                        round_num,
                        DEBATE_TEMP,
                    )
                else:
                    text, usage = agent_debate_response(
                        agent_question,
                        choices,
                        agent_id,
                        predecessor_id,
                        enhanced_context,
                        round_num,
                        DEBATE_TEMP,
                    )
            except Exception as exc:
                text = f"[ERROR] {exc}"
                usage = _zero_usage()
            history[round_num][agent_id] = text
            if multi_answer:
                answer, repair_usage, repair_text = extract_or_repair_answer_set(
                    text,
                    choices,
                    agent_id,
                )
            else:
                answer, repair_usage, repair_text = extract_or_repair_answer(
                    text,
                    choices,
                    agent_id,
                )
            if repair_usage.get("total_tokens", 0) > 0:
                _add_usage(usage, repair_usage)
                answer_repair_log.setdefault(str(round_num), []).append(
                    _build_answer_repair_entry(
                        round_num=round_num,
                        agent_id=agent_id,
                        answer=answer,
                        original_response=text,
                        repair_response=repair_text,
                        repair_usage=repair_usage,
                    )
                )
            usages[round_num][agent_id] = usage
            answers[round_num][agent_id] = answer
            time.sleep(0.5)

        if verbose:
            print(
                f"  R{round_num}: {{"
                + ", ".join(
                    f"{agent_id}:{answers[round_num][agent_id]}"
                    for agent_id in agent_ids
                )
                + "}"
            )

        current_ledger = compute_verdicts(
            {str(agent_id): answers[round_num][agent_id] for agent_id in agent_ids},
            num_options=num_options,
            multi_answer_majority=multi_answer,
        )
        compression_level = decide_compression_level(
            current_ledger,
            prev_summary is not None,
        )
        n_disputed = _count_disputed(current_ledger)
        adaptive_stats = _compute_adaptive_stability(
            answers,
            round_num,
            option_labels,
        )
        round_log: Dict[str, Any] = {
            "round": round_num,
            "compression_level": compression_level,
            "n_disputed": n_disputed,
            "disputed": [
                option for option, info in current_ledger.items()
                if info["verdict"] == "disputed"
            ],
            "adaptive_stability": adaptive_stats,
            "answer_repairs": answer_repair_log.get(str(round_num), []),
        }
        verdict_history.append(current_ledger)

        if (
            round_num >= MIN_DEBATE_ROUNDS
            and _all_stable_answer_unchanged_local(verdict_history)
        ):
            if not enable_all_stable_safety_gate:
                round_log["exit_check"] = "all_stable"
                mechanism_log.append(round_log)
                final_answer = _get_final_answer_multi(
                    answers,
                    round_num,
                    current_ledger,
                    multi_answer,
                )
                if verbose:
                    print(f"    -> Early exit: all_stable, answer={final_answer}")
                return _build_adaptive_result(
                    history,
                    usages,
                    answers,
                    agent_ids,
                    final_answer,
                    round_num,
                    "all_stable",
                    mechanism_log,
                    summarizer_total_usage,
                    dispute_resolver_total_usage,
                    answer_repair_log=answer_repair_log,
                    variant="adaptive_resolver_v1",
                )

            # V2 + all_stable safety gate:
            #   all_stable must pass temporal safety checks; unsafe cases go
            #   through a lightweight groupthink challenge instead of ordinary
            #   final review.
            mechanism_log.append(round_log)
            final_answer = _get_final_answer_multi(
                answers,
                round_num,
                current_ledger,
                multi_answer,
            )
            safety = _all_stable_safety_check(
                answers,
                round_num,
                final_answer,
                agent_ids,
            )
            round_log["all_stable_safety_gate"] = safety
            if safety["safe"]:
                round_log["exit_check"] = "all_stable_safety_accept"
                if verbose:
                    print(
                        "    -> Early exit: all_stable_safety_accept, "
                        f"answer={final_answer}"
                    )
                return _build_adaptive_result(
                    history,
                    usages,
                    answers,
                    agent_ids,
                    final_answer,
                    round_num,
                    "all_stable_safety_accept",
                    mechanism_log,
                    summarizer_total_usage,
                    dispute_resolver_total_usage,
                    answer_repair_log=answer_repair_log,
                    variant=variant_name,
                )

            challenge_evidence_context = None
            if (
                enable_question_aware_evidence_snippets
                and evidence_snippet_policy == "selective"
            ):
                focus_options = _all_stable_challenge_focus_options(
                    final_answer,
                    safety,
                )
                challenge_evidence_context, evidence_payload = (
                    _build_question_aware_evidence_context(
                        question,
                        choices,
                        current_ledger,
                        method=evidence_snippet_method,
                        focus_options=focus_options,
                    )
                )
                gate = _selective_evidence_snippet_gate(
                    current_ledger,
                    verdict_history,
                    adaptive_stats,
                    round_log,
                    agent_ids,
                    all_stable_safety=safety,
                )
                evidence_payload["gate"] = gate
                round_log["question_aware_evidence_snippets"] = evidence_payload

            if multi_answer:
                challenge, challenge_text, challenge_usage = _run_all_stable_challenge_multi_answer(
                    question,
                    choices,
                    final_answer,
                    answers,
                    history,
                    round_num,
                    safety,
                    evidence_context=challenge_evidence_context,
                )
            else:
                challenge, challenge_text, challenge_usage = _run_all_stable_challenge(
                    question,
                    choices,
                    final_answer,
                    answers,
                    history,
                    round_num,
                    safety,
                    evidence_context=challenge_evidence_context,
                )
            _add_usage(dispute_resolver_total_usage, challenge_usage)
            if multi_answer:
                challenged_answer = _choose_all_stable_challenge_answer_multi(
                    final_answer,
                    challenge,
                )
            else:
                challenged_answer = _choose_all_stable_challenge_answer(
                    final_answer,
                    challenge,
                )
            round_log["all_stable_challenge"] = {
                "triggered": True,
                "analysis": challenge,
                "response": challenge_text,
                "usage": challenge_usage,
                "used_answer": challenged_answer,
            }
            round_log["exit_check"] = (
                "all_stable_challenge"
                if challenged_answer != final_answer
                else "all_stable_challenge_accept"
            )
            if verbose:
                print(
                    f"    -> Early exit: {round_log['exit_check']}, "
                    f"answer={challenged_answer}"
                )
            return _build_adaptive_result(
                history,
                usages,
                answers,
                agent_ids,
                challenged_answer,
                round_num,
                round_log["exit_check"],
                mechanism_log,
                summarizer_total_usage,
                dispute_resolver_total_usage,
                answer_repair_log=answer_repair_log,
                variant=variant_name,
            )

        if active_resolver_analysis:
            if multi_answer:
                confirmed, confirmation = _resolver_confirmed_multi_answer(
                    active_resolver_analysis,
                    adaptive_stats,
                    resolver_medium_confirmed_once,
                )
            else:
                confirmed, confirmation = _resolver_confirmed(
                    active_resolver_analysis,
                    adaptive_stats,
                    resolver_medium_confirmed_once,
                )
            round_log["resolver_confirmation"] = confirmation
            if confirmed:
                if enable_resolver_influence_gate:
                    if multi_answer:
                        influence_gate = _resolver_influence_gate_multi_answer(
                            active_resolver_analysis,
                            confirmation,
                            active_resolver_trigger_snapshot,
                            answers,
                            round_num,
                            agent_ids,
                        )
                    else:
                        influence_gate = _resolver_influence_gate(
                            active_resolver_analysis,
                            confirmation,
                            active_resolver_trigger_snapshot,
                            answers,
                            round_num,
                            agent_ids,
                        )
                    round_log["resolver_influence_gate"] = influence_gate
                    if not influence_gate["safe"]:
                        next_resolver_context = _format_resolver_influence_block_for_prompt(
                            influence_gate
                        )
                        next_resolver_analysis = None
                        round_log["exit_check"] = "resolver_influence_blocked_continue"
                        confirmed = False
                    else:
                        round_log["exit_check"] = "resolver_confirmed_influence_accept"
                else:
                    round_log["exit_check"] = "resolver_confirmed"

            if confirmed:
                mechanism_log.append(round_log)
                final_answer = confirmation.get("resolver_answer")
                if verbose:
                    print(
                        f"    -> Early exit: {round_log['exit_check']}, "
                        f"answer={final_answer}"
                    )
                return _build_adaptive_result(
                    history,
                    usages,
                    answers,
                    agent_ids,
                    final_answer,
                    round_num,
                    round_log["exit_check"],
                    mechanism_log,
                    summarizer_total_usage,
                    dispute_resolver_total_usage,
                    answer_repair_log=answer_repair_log,
                    variant=variant_name,
                )
            if (
                confirmation.get("majority_matches_resolver")
                and confirmation.get("calibrated_level") == "medium"
                and confirmation.get("margin", 0) >= ADAPTIVE_MIN_MARGIN
                and _metric_float(confirmation, "normalized_entropy", 1.0)
                <= ADAPTIVE_MAX_NORMALIZED_ENTROPY
            ):
                resolver_medium_confirmed_once = True
                next_resolver_context = _format_resolver_for_prompt(
                    active_resolver_analysis,
                    "",
                    mode="full",
                )
                next_resolver_analysis = active_resolver_analysis
                round_log["exit_check"] = "resolver_medium_observe"
            elif confirmation.get("calibrated_level") == "low":
                round_log["exit_check"] = "resolver_low_continue"

        responses_for_summary = {
            str(agent_id): history[round_num][agent_id] for agent_id in agent_ids
        }
        if force_llm_summaries:
            summary = _generate_llm_summary_v5(
                compression_level,
                current_ledger,
                responses_for_summary,
                round_num,
                prev_summary,
            )
        else:
            summary = generate_summary(
                compression_level,
                current_ledger,
                responses_for_summary,
                round_num,
                prev_summary,
            )
        _add_usage(summarizer_total_usage, summary.get("usage", {}))
        prev_summary = summary

        if (
            round_num >= MIN_DEBATE_ROUNDS
            and n_disputed > 0
            and round_num < MAX_DEBATE_ROUNDS
            and resolver_trigger_count < 1
        ):
            if multi_answer:
                resolver, resolver_text, resolver_usage = _run_dispute_resolver_multi_answer(
                    question,
                    choices,
                    current_ledger,
                    answers,
                    history,
                    round_num,
                    adaptive_stats,
                )
            else:
                resolver, resolver_text, resolver_usage = _run_dispute_resolver(
                    question,
                    choices,
                    current_ledger,
                    answers,
                    history,
                    round_num,
                    adaptive_stats,
                )
            _add_usage(dispute_resolver_total_usage, resolver_usage)
            if multi_answer:
                calibrated = _compute_calibrated_confidence_multi_answer(
                    resolver,
                    adaptive_stats,
                )
            else:
                calibrated = _compute_calibrated_confidence(
                    resolver,
                    adaptive_stats,
                )
            resolver_trigger_count += 1
            if calibrated["level"] == "low":
                next_resolver_context = _format_resolver_for_prompt(
                    resolver,
                    resolver_text,
                    mode="weak",
                )
                next_resolver_analysis = None
                next_resolver_trigger_snapshot = None
            else:
                next_resolver_context = _format_resolver_for_prompt(
                    resolver,
                    resolver_text,
                    mode="full",
                )
                next_resolver_analysis = resolver
                if multi_answer:
                    next_resolver_trigger_snapshot = _resolver_trigger_snapshot_multi_answer(
                        resolver,
                        adaptive_stats,
                        answers,
                        round_num,
                    )
                else:
                    next_resolver_trigger_snapshot = _resolver_trigger_snapshot(
                        resolver,
                        adaptive_stats,
                        answers,
                        round_num,
                    )
            round_log["dispute_resolver"] = {
                "triggered": True,
                "trigger_policy": "first_r2_disputed",
                "analysis": resolver,
                "calibrated_confidence": calibrated,
                "trigger_snapshot": next_resolver_trigger_snapshot,
                "response": resolver_text,
                "usage": resolver_usage,
            }
            round_log["exit_check"] = (
                "resolver_weak_next_round"
                if calibrated["level"] == "low"
                else "resolver_next_round"
            )

        should_inject_evidence = (
            enable_question_aware_evidence_snippets
            and n_disputed > 0
            and round_num < MAX_DEBATE_ROUNDS
        )
        evidence_gate = None
        if (
            should_inject_evidence
            and evidence_snippet_policy == "selective"
            and not enable_selective_passage_replacement
        ):
            evidence_gate = _selective_evidence_snippet_gate(
                current_ledger,
                verdict_history,
                adaptive_stats,
                round_log,
                agent_ids,
            )
            should_inject_evidence = evidence_gate["inject"]
            round_log["evidence_snippet_gate"] = evidence_gate

        if should_inject_evidence:
            if enable_selective_passage_replacement:
                evidence_context, evidence_payload = _build_v5_structured_evidence_context(
                    question,
                    choices,
                    current_ledger,
                    method=evidence_snippet_method,
                )
            else:
                evidence_context, evidence_payload = _build_question_aware_evidence_context(
                    question,
                    choices,
                    current_ledger,
                    method=evidence_snippet_method,
                )
            if evidence_gate:
                evidence_payload["gate"] = evidence_gate
            if enable_selective_passage_replacement:
                replacement_gate = _v5_selective_replacement_gate(
                    verdict_history,
                    adaptive_stats,
                    round_log,
                    evidence_payload,
                )
                evidence_payload["replacement_gate"] = replacement_gate
                if replacement_gate["replace_full_passage"]:
                    next_replacement_question = _build_v5_replacement_question(
                        question,
                        evidence_payload,
                    )
                    evidence_payload["replaces_full_passage"] = True
                    round_log["passage_replacement"] = {
                        "enabled": True,
                        "applies_next_round": True,
                        "gate": replacement_gate,
                    }
                else:
                    next_evidence_context = evidence_context
                    round_log["passage_replacement"] = {
                        "enabled": True,
                        "applies_next_round": False,
                        "gate": replacement_gate,
                    }
            else:
                next_evidence_context = evidence_context
            round_log["question_aware_evidence_snippets"] = evidence_payload
        elif evidence_gate:
            round_log["question_aware_evidence_snippets"] = {
                "enabled": True,
                "requested_method": evidence_snippet_method,
                "policy": evidence_snippet_policy,
                "skipped": True,
                "gate": evidence_gate,
            }

        if "exit_check" not in round_log:
            round_log["exit_check"] = "continue"

        mechanism_log.append(round_log)

        if verbose:
            print(
                "    adaptive="
                f"top={adaptive_stats['top_answer']}, "
                f"margin={adaptive_stats['margin']}, "
                f"entropy={adaptive_stats['normalized_entropy']:.3f}, "
                f"tv={adaptive_stats['tv_distance']:.3f}, "
                f"stable={adaptive_stats['stable']}, "
                f"exit={round_log['exit_check']}"
            )

    fallback_answer = _get_final_answer_multi(
        answers,
        MAX_DEBATE_ROUNDS,
        current_ledger,
        multi_answer,
    )
    if multi_answer:
        review_answer, review_text, review_usage = _run_final_evidence_review_multi_answer(
            question,
            choices,
            current_ledger,
            answers,
            history,
            MAX_DEBATE_ROUNDS,
        )
    else:
        review_answer, review_text, review_usage = _run_final_evidence_review(
            question,
            choices,
            current_ledger,
            answers,
            history,
            MAX_DEBATE_ROUNDS,
        )
    final_answer = review_answer or fallback_answer
    if mechanism_log:
        mechanism_log[-1]["final_review"] = {
            "triggered": True,
            "answer": review_answer,
            "fallback_answer": fallback_answer,
            "used_answer": final_answer,
            "response": review_text,
        }
    if verbose:
        print(
            "    -> Max rounds final evidence review: "
            f"answer={review_answer}, fallback={fallback_answer}, used={final_answer}"
        )
    return _build_adaptive_result(
        history,
        usages,
        answers,
        agent_ids,
        final_answer,
        MAX_DEBATE_ROUNDS,
        "max_rounds",
        mechanism_log,
        summarizer_total_usage,
        dispute_resolver_total_usage,
        final_review_usage=review_usage,
        answer_repair_log=answer_repair_log,
        variant=variant_name,
    )


def _compute_adaptive_stability(
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
    option_labels: List[str],
) -> Dict[str, Any]:
    """Compute categorical stability metrics for the latest answer round."""
    current_answer_labels = [
        answer for answer in answers.get(current_round, {}).values() if answer
    ]
    previous_answer_labels = [
        answer for answer in answers.get(current_round - 1, {}).values() if answer
    ]
    distribution_labels = sorted(
        set(option_labels) | set(current_answer_labels) | set(previous_answer_labels)
    )
    current_dist = _answer_distribution(answers.get(current_round, {}), distribution_labels)
    previous_dist = _answer_distribution(
        answers.get(current_round - 1, {}),
        distribution_labels,
    )
    current_counts = Counter(
        answer for answer in answers.get(current_round, {}).values()
        if answer
    )
    previous_counts = Counter(
        answer for answer in answers.get(current_round - 1, {}).values()
        if answer
    )

    top_answer, top_count, runner_up_count = _top_answer_counts(current_counts)
    previous_top, _, _ = _top_answer_counts(previous_counts)
    margin = top_count - runner_up_count
    entropy = _entropy(current_dist)
    total_votes = sum(current_counts.values())
    max_entropy = math.log(max(2, total_votes))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    tv_distance = 0.5 * sum(
        abs(current_dist.get(option, 0.0) - previous_dist.get(option, 0.0))
        for option in distribution_labels
    )
    top_unchanged = top_answer is not None and top_answer == previous_top
    stable = (
        top_unchanged
        and margin >= ADAPTIVE_MIN_MARGIN
        and normalized_entropy <= ADAPTIVE_MAX_NORMALIZED_ENTROPY
        and tv_distance <= ADAPTIVE_MAX_TV_DISTANCE
    )

    return {
        "top_answer": top_answer,
        "top_count": top_count,
        "runner_up_count": runner_up_count,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "tv_distance": tv_distance,
        "top_unchanged": top_unchanged,
        "stable": stable,
        "distribution": current_dist,
        "thresholds": {
            "min_margin": ADAPTIVE_MIN_MARGIN,
            "max_normalized_entropy": ADAPTIVE_MAX_NORMALIZED_ENTROPY,
            "max_tv_distance": ADAPTIVE_MAX_TV_DISTANCE,
        },
    }


def _answer_distribution(
    answers_round: Dict[int, Optional[str]],
    option_labels: List[str],
) -> Dict[str, float]:
    """Return a probability distribution over option labels."""
    counts = Counter(
        answer for answer in answers_round.values()
        if answer in option_labels
    )
    total = sum(counts.values())
    if total <= 0:
        return {option: 0.0 for option in option_labels}
    return {option: counts.get(option, 0) / total for option in option_labels}


def _top_answer_counts(counts: Counter[str]) -> Tuple[Optional[str], int, int]:
    """Return top answer, top count, and runner-up count."""
    if not counts:
        return None, 0, 0
    ranked = counts.most_common()
    top_answer, top_count = ranked[0]
    runner_up_count = ranked[1][1] if len(ranked) > 1 else 0
    return top_answer, top_count, runner_up_count


def _entropy(distribution: Dict[str, float]) -> float:
    """Compute Shannon entropy for a categorical distribution."""
    return -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0
    )


def _run_dispute_resolver(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
    adaptive_stats: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Usage]:
    """
    Ask Agent 1 to diagnose stable-but-disputed cases.

    The resolver is not a final judge. Its structured analysis is passed to the
    next debate round so agents can inspect evidence, reason quality, option
    contrasts, and likely error sources.
    """
    client, model, name = get_client(SUMMARIZER_AGENT_ID)
    disputed = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "disputed"
    ]
    standings = _format_option_standings(option_ledger)
    recent_votes = _format_recent_votes(answers, current_round)
    recent_responses = _format_recent_responses(
        history,
        current_round,
        max_chars_per_response=900,
    )
    stats_text = json.dumps(adaptive_stats, ensure_ascii=False, indent=2)

    prompt = f"""You are an independent dispute resolver. The debate is stable by answer-distribution metrics, but some options remain disputed.

Your job is not to follow the majority. Diagnose the dispute so the next debate round can correct mistakes.

Question:
{question}

Options:
{choices}

Disputed options: {', '.join(disputed) if disputed else 'none'}

Option standings:
{standings}

Recent vote history:
{recent_votes}

Adaptive stability metrics:
{stats_text}

Recent agent reasoning:
{recent_responses}

Analyze exactly these four aspects:
1. Evidence grounding: what evidence supports or weakens each disputed option.
2. Reason quality: whether the reasons are direct, circular, irrelevant, or contradictory.
3. Option contrast: why the strongest candidate is better than the closest alternative.
4. Error diagnosis: the likely source of disagreement.

Output strict JSON only:
{{
  "recommended_answer": "<single option letter or null>",
  "confidence": "low/medium/high",
  "evidence_grounding": {{
    "<option>": {{
      "support": "specific evidence or argument supporting this option",
      "weakness": "specific weakness or missing evidence",
      "strength": "weak/medium/strong"
    }}
  }},
  "reason_quality": {{
    "<option>": {{"score": 1, "issue": "..."}}
  }},
  "option_contrast": "one concise paragraph",
  "error_diagnosis": {{
    "root_cause": "one sentence",
    "likely_error_type": "evidence_miss/question_misread/concept_error/reasoning_error/ambiguous"
  }},
  "next_round_instruction": "one sentence telling agents what to re-check"
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            usage = _zero_usage()
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
            text = response.choices[0].message.content or ""
            parsed = parse_summary_json(text)
            if not parsed:
                parsed = _fallback_resolver(disputed)
            return parsed, text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Dispute resolver failed: {exc}")

    return _fallback_resolver(disputed), "", _zero_usage()


def _run_dispute_resolver_multi_answer(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
    adaptive_stats: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Usage]:
    """Ask the resolver to recommend an answer set for MultiRC-style tasks."""
    client, model, name = get_client(SUMMARIZER_AGENT_ID)
    del name
    disputed = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "disputed"
    ]
    standings = _format_option_standings(option_ledger)
    recent_votes = _format_recent_votes(answers, current_round)
    recent_responses = _format_recent_responses(
        history,
        current_round,
        max_chars_per_response=900,
    )
    stats_text = json.dumps(adaptive_stats, ensure_ascii=False, indent=2)

    prompt = f"""You are an independent dispute resolver for a multiple-answer reading-comprehension debate.

Your job is not to follow the majority. Treat each option independently as include or exclude.

Question:
{question}

Options:
{choices}

Disputed options: {', '.join(disputed) if disputed else 'none'}

Option standings:
{standings}

Recent vote history:
{recent_votes}

Adaptive stability metrics:
{stats_text}

Recent agent reasoning:
{recent_responses}

Analyze evidence grounding, reason quality, option contrast, and error diagnosis.

Output strict JSON only:
{{
  "recommended_answer": "A,C",
  "recommended_answers": ["A", "C"],
  "confidence": "low/medium/high",
  "evidence_grounding": {{
    "<option>": {{
      "support": "specific evidence or argument supporting inclusion",
      "weakness": "specific weakness or missing evidence",
      "strength": "weak/medium/strong"
    }}
  }},
  "reason_quality": {{
    "<option>": {{"score": 1, "issue": "..."}}
  }},
  "option_contrast": "one concise paragraph",
  "error_diagnosis": {{
    "root_cause": "one sentence",
    "likely_error_type": "evidence_miss/question_misread/concept_error/reasoning_error/ambiguous"
  }},
  "next_round_instruction": "one sentence telling agents what to re-check"
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            usage = _zero_usage()
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
            text = response.choices[0].message.content or ""
            parsed = parse_summary_json(text)
            if not parsed:
                parsed = _fallback_resolver(disputed)
            parsed["recommended_answer"] = _normalize_answer_set_value(
                parsed.get("recommended_answer") or parsed.get("recommended_answers")
            )
            return parsed, text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Multi-answer dispute resolver failed: {exc}")

    return _fallback_resolver(disputed), "", _zero_usage()


def _fallback_resolver(disputed_options: List[str]) -> Dict[str, Any]:
    """Return a minimal resolver analysis when JSON parsing/calling fails."""
    return {
        "recommended_answer": None,
        "confidence": "low",
        "evidence_grounding": {
            option: {
                "support": "(resolver unavailable)",
                "weakness": "(resolver unavailable)",
                "strength": "weak",
            }
            for option in disputed_options
        },
        "reason_quality": {
            option: {"score": 1, "issue": "(resolver unavailable)"}
            for option in disputed_options
        },
        "option_contrast": "(resolver unavailable)",
        "error_diagnosis": {
            "root_cause": "(resolver unavailable)",
            "likely_error_type": "ambiguous",
        },
        "next_round_instruction": "Re-check the disputed options against the question wording.",
    }


def _run_all_stable_challenge(
    question: str,
    choices: str,
    current_answer: Optional[str],
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
    safety: Dict[str, Any],
    evidence_context: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, Usage]:
    """
    Lightweight all_stable challenge.

    This is not a full final review. It only checks whether the current
    unanimous answer may be groupthink after an unsafe temporal pattern.
    """
    client, model, _name = get_client(SUMMARIZER_AGENT_ID)
    recent_votes = _format_recent_votes(answers, current_round)
    recent_responses = _format_recent_responses(
        history,
        current_round,
        max_chars_per_response=700,
    )
    safety_text = json.dumps(safety, ensure_ascii=False, indent=2)

    prompt = f"""You are a lightweight all-stable challenge resolver.

All agents currently agree on answer: {current_answer}

The agreement failed a temporal safety gate, so your task is ONLY to check whether this consensus may be groupthink.
Do not perform a long final review. Look for signs that agents copied a wrong answer, ignored the question wording, or migrated together without new evidence.

Question:
{question}

Options:
{choices}

Safety gate diagnostics:
{safety_text}

Recent vote history:
{recent_votes}

Recent agent reasoning:
{recent_responses}

Question-aware evidence snippets:
{evidence_context or "None"}

Output strict JSON only:
{{
  "decision": "accept/challenge",
  "alternative_answer": "<single option letter or null>",
  "confidence": "low/medium/high",
  "groupthink_risk": "low/medium/high",
  "evidence_check": "one sentence on whether the consensus answer is directly supported",
  "reason": "one concise sentence"
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            usage = _zero_usage()
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
            text = response.choices[0].message.content or ""
            parsed = parse_summary_json(text)
            if not parsed:
                parsed = _fallback_all_stable_challenge(current_answer)
            return parsed, text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"All-stable challenge failed: {exc}")

    return _fallback_all_stable_challenge(current_answer), "", _zero_usage()


def _fallback_all_stable_challenge(
    current_answer: Optional[str],
) -> Dict[str, Any]:
    """Return a conservative challenge result if parsing/calling fails."""
    return {
        "decision": "accept",
        "alternative_answer": None,
        "confidence": "low",
        "groupthink_risk": "unknown",
        "evidence_check": "(challenge unavailable)",
        "reason": f"Keep current consensus answer {current_answer}.",
    }


def _run_all_stable_challenge_multi_answer(
    question: str,
    choices: str,
    current_answer: Optional[str],
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
    safety: Dict[str, Any],
    evidence_context: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, Usage]:
    """Lightweight challenge for an unsafe all-stable answer set."""
    client, model, _name = get_client(SUMMARIZER_AGENT_ID)
    recent_votes = _format_recent_votes(answers, current_round)
    recent_responses = _format_recent_responses(
        history,
        current_round,
        max_chars_per_response=700,
    )
    evidence_block = (
        f"\nRetrieved evidence snippets:\n{evidence_context}\n"
        if evidence_context
        else ""
    )
    prompt = f"""You are a lightweight safety challenger for a multiple-answer debate.

All agents currently agree on answer set: {current_answer}

Question:
{question}

Options:
{choices}

Recent vote history:
{recent_votes}

Safety diagnostics:
{json.dumps(safety, ensure_ascii=False, indent=2)}
{evidence_block}
Recent agent reasoning:
{recent_responses}

Task:
Check whether the consensus answer set is directly supported. If an option should be added or removed, propose the corrected answer set.
For multi-answer tasks, do not remove currently included options. Only propose additional options if they are strongly supported.

Output strict JSON only:
{{
  "safe": true,
  "alternative_answer": "A,C",
  "confidence": "low/medium/high",
  "reason": "one concise sentence",
  "evidence_check": "one sentence"
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            usage = _zero_usage()
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
            text = response.choices[0].message.content or ""
            parsed = parse_summary_json(text)
            if not parsed:
                parsed = _fallback_all_stable_challenge(current_answer)
            parsed["alternative_answer"] = _normalize_answer_set_value(
                parsed.get("alternative_answer")
            )
            return parsed, text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Multi-answer all-stable challenge failed: {exc}")

    return _fallback_all_stable_challenge(current_answer), "", _zero_usage()


def _choose_all_stable_challenge_answer_multi(
    current_answer: Optional[str],
    challenge: Dict[str, Any],
) -> Optional[str]:
    """Choose answer set from a multi-answer all-stable challenge.

    MultiRC's dominant error mode is omission, so the challenge is add-only:
    it may add strongly supported options but cannot remove current includes.
    """
    alternative = _normalize_answer_set_value(challenge.get("alternative_answer"))
    confidence = str(challenge.get("confidence", "")).strip().lower()
    if (
        alternative
        and confidence == "high"
        and challenge.get("safe") is False
    ):
        current_options = set(re.findall(r"[A-J]", str(current_answer or "").upper()))
        alternative_options = set(re.findall(r"[A-J]", alternative))
        merged = _canonical_answer_string(sorted(current_options | alternative_options))
        return merged or current_answer
    return current_answer


def _choose_all_stable_challenge_answer(
    current_answer: Optional[str],
    challenge: Dict[str, Any],
) -> Optional[str]:
    """Choose the final answer from a lightweight all-stable challenge."""
    decision = str(challenge.get("decision", "")).strip().lower()
    confidence = str(challenge.get("confidence", "")).strip().lower()
    alternative = _normalize_option(challenge.get("alternative_answer"))
    if (
        decision == "challenge"
        and confidence in {"medium", "high"}
        and alternative
        and alternative != current_answer
    ):
        return alternative
    return current_answer


def _all_stable_challenge_focus_options(
    current_answer: Optional[str],
    safety: Dict[str, Any],
) -> List[str]:
    """Return option targets for unsafe all-stable evidence retrieval."""
    candidates = [
        current_answer,
        safety.get("r0_majority", {}).get("majority_answer"),
        safety.get("history_weighted_vote", {}).get("weighted_answer"),
        safety.get("current_majority", {}).get("majority_answer"),
    ]
    focus: List[str] = []
    for candidate in candidates:
        option = _normalize_option(candidate)
        if option and option not in focus:
            focus.append(option)
    return focus


def _selective_evidence_snippet_gate(
    option_ledger: OptionLedger,
    verdict_history: List[OptionLedger],
    adaptive_stats: Dict[str, Any],
    round_log: Dict[str, Any],
    agent_ids: List[int],
    all_stable_safety: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Two-level V4.3 risk gate for evidence snippets.

    Strong risks always inject snippets. Weak risks inject only when the vote
    state is uncertain, avoiding the always-on V4.1/V4.2 behavior.
    """
    thresholds = _n_aware_thresholds(len(agent_ids))
    strong_reasons: List[str] = []
    weak_reasons: List[str] = []

    if all_stable_safety is not None and not all_stable_safety.get("safe", False):
        strong_reasons.append("all_stable_safety_failed")

    influence_gate = round_log.get("resolver_influence_gate", {})
    if (
        round_log.get("exit_check") == "resolver_influence_blocked_continue"
        or (
            isinstance(influence_gate, dict)
            and influence_gate
            and not influence_gate.get("safe", True)
        )
    ):
        strong_reasons.append("resolver_influence_blocked")

    stalled = _disputed_stalled_for_two_rounds(verdict_history, adaptive_stats)
    if stalled["stalled"]:
        weak_reasons.append("disputed_stalled_for_two_rounds")

    round_num = int(round_log.get("round", 0) or 0)
    if round_num == MAX_DEBATE_ROUNDS - 1 and _count_disputed(option_ledger) > 0:
        weak_reasons.append("pre_max_round_disputed")

    calibrated_level = _latest_calibrated_level(round_log)
    margin = int(adaptive_stats.get("margin", 0) or 0)
    normalized_entropy = _metric_float(adaptive_stats, "normalized_entropy", 1.0)
    weak_uncertain_reasons: List[str] = []
    if calibrated_level in ("low", "medium"):
        weak_uncertain_reasons.append(f"calibrated_{calibrated_level}")
    if margin <= thresholds["weak_margin_threshold"]:
        weak_uncertain_reasons.append("weak_margin")
    if normalized_entropy >= SELECTIVE_EVIDENCE_HIGH_ENTROPY:
        weak_uncertain_reasons.append("high_entropy")

    strong = bool(strong_reasons)
    weak = bool(weak_reasons)
    weak_uncertain = bool(weak_uncertain_reasons)
    inject = strong or (weak and weak_uncertain)

    return {
        "policy": "selective_two_level",
        "inject": inject,
        "risk_level": "strong" if strong else ("weak" if weak else "none"),
        "strong_reasons": strong_reasons,
        "weak_reasons": weak_reasons,
        "weak_uncertain_reasons": weak_uncertain_reasons,
        "stalled_dispute": stalled,
        "metrics": {
            "margin": margin,
            "normalized_entropy": normalized_entropy,
            "tv_distance": _metric_float(adaptive_stats, "tv_distance", 1.0),
            "calibrated_level": calibrated_level,
        },
        "thresholds": {
            **thresholds,
            "high_entropy": SELECTIVE_EVIDENCE_HIGH_ENTROPY,
            "weak_injection_requires": [
                "weak risk",
                "and calibrated low/medium OR weak margin OR high entropy",
            ],
        },
    }


def _disputed_stalled_for_two_rounds(
    verdict_history: List[OptionLedger],
    adaptive_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Check whether disputed option support counts stayed unchanged."""
    if len(verdict_history) < 2:
        return {"stalled": False, "reason": "insufficient_history"}

    previous = verdict_history[-2]
    current = verdict_history[-1]
    previous_disputed = {
        option for option, info in previous.items()
        if info.get("verdict") == "disputed"
    }
    current_disputed = {
        option for option, info in current.items()
        if info.get("verdict") == "disputed"
    }
    if not current_disputed:
        return {"stalled": False, "reason": "no_current_dispute"}
    if previous_disputed != current_disputed:
        return {
            "stalled": False,
            "reason": "disputed_set_changed",
            "previous_disputed": sorted(previous_disputed),
            "current_disputed": sorted(current_disputed),
        }

    changed_counts = []
    for option in sorted(current_disputed):
        prev_count = previous.get(option, {}).get("support_count", -1)
        curr_count = current.get(option, {}).get("support_count", -1)
        if prev_count != curr_count:
            changed_counts.append({
                "option": option,
                "previous": prev_count,
                "current": curr_count,
            })

    tv_distance = _metric_float(adaptive_stats, "tv_distance", 1.0)
    stalled = not changed_counts and tv_distance <= ADAPTIVE_MAX_TV_DISTANCE
    return {
        "stalled": stalled,
        "previous_disputed": sorted(previous_disputed),
        "current_disputed": sorted(current_disputed),
        "changed_counts": changed_counts,
        "tv_distance": tv_distance,
        "max_tv_distance": ADAPTIVE_MAX_TV_DISTANCE,
    }


def _latest_calibrated_level(round_log: Dict[str, Any]) -> Optional[str]:
    """Return the most recent resolver confidence level available in a log."""
    resolver = round_log.get("dispute_resolver", {})
    if isinstance(resolver, dict):
        calibrated = resolver.get("calibrated_confidence", {})
        if isinstance(calibrated, dict) and calibrated.get("level"):
            return str(calibrated.get("level"))
    confirmation = round_log.get("resolver_confirmation", {})
    if isinstance(confirmation, dict) and confirmation.get("calibrated_level"):
        return str(confirmation.get("calibrated_level"))
    return None


def _generate_llm_summary_v5(
    compression_level: str,
    option_ledger: OptionLedger,
    responses: Dict[str, str],
    round_num: int,
    prev_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate V5 summaries.

    V5 keeps the original V3 zero-cost rule capsule for NONE, while FULL and
    PARTIAL use the LLM summarizer.
    """
    if compression_level == "none":
        return generate_summary(
            compression_level,
            option_ledger,
            responses,
            round_num,
            prev_summary,
        )

    num_agents = len(responses)
    response_text = "\n\n".join(
        f"Agent {agent_id}:\n{text}" for agent_id, text in responses.items()
    )
    prev_text = (
        json.dumps(prev_summary, indent=2, ensure_ascii=False)
        if prev_summary
        else "None"
    )

    if compression_level == "full":
        prompt = f"""You are a debate summarizer. All {num_agents} agents agreed on every option this round.

Agent responses:
{response_text}

Previous round summary:
{prev_text}

Task:
1. For each option, write ONE sentence (max 30 words) explaining why it is included or excluded.
2. Record which agents changed stance from previous round.
3. Output strict JSON only, no other text.

JSON format:
{{
  "round": {round_num},
  "compression_level": "full",
  "option_ledger": {{
    "<option>": {{"verdict": "include or exclude", "reason": "one sentence"}}
  }},
  "final_answer": ["included options"],
  "delta_from_last_round": {{
    "stance_changes": [{{"agent": "<id>", "option": "<opt>", "from": "<old>", "to": "<new>"}}],
    "new_reasons": 0
  }}
}}"""
    else:
        frozen = [
            option for option, value in option_ledger.items()
            if value["verdict"] != "disputed"
        ]
        disputed = [
            option for option, value in option_ledger.items()
            if value["verdict"] == "disputed"
        ]
        frozen_info = ", ".join(
            f"{option}({option_ledger[option]['verdict']})"
            for option in frozen
        )
        disputed_info = ", ".join(
            f"{option}(support:{option_ledger[option]['support_count']}/{num_agents} "
            f"agents={option_ledger[option]['support_agents']})"
            for option in disputed
        )
        prompt = f"""You are a debate summarizer. Some options reached consensus, others are disputed.

Agent responses:
{response_text}

Previous round summary:
{prev_text}

Stable options: {frozen_info}
Disputed options: {disputed_info}

Task:
1. For each stable option, write ONE sentence explaining the reason.
2. For each disputed option, write the top-1 SUPPORT reason and top-1 OPPOSE reason separately. Do NOT merge them. Max 30 words each.
3. For disputed options, list supporting and opposing agent IDs.
4. Record stance changes from previous round.
5. Output strict JSON only, no other text.

JSON format:
{{
  "round": {round_num},
  "compression_level": "partial",
  "option_ledger": {{
    "<stable_opt>": {{"verdict": "include/exclude", "reason": "one sentence"}},
    "<disputed_opt>": {{
      "verdict": "disputed",
      "support": {{"count": 0, "agents": [], "top_reason": "..."}},
      "oppose": {{"count": 0, "agents": [], "top_reason": "..."}}
    }}
  }},
  "frozen_options": [],
  "disputed_options": [],
  "current_answer": [],
  "delta_from_last_round": {{
    "stance_changes": [],
    "new_reasons": 0
  }}
}}"""

    text, usage = call_summarizer(prompt)
    result = parse_summary_json(text)
    if not result:
        result = _fallback_llm_summary_v5(option_ledger, round_num, compression_level)
    result["compression_level"] = compression_level
    result["round"] = round_num
    result["usage"] = usage
    result["option_ledger_raw"] = option_ledger
    return result


def _fallback_llm_summary_v5(
    option_ledger: OptionLedger,
    round_num: int,
    level: str,
) -> Dict[str, Any]:
    """Build a schema-compatible deterministic fallback for V5 LLM summaries."""
    if level == "none":
        lines = [f"[Debate Status - Round {round_num}]", "Option standings:"]
        for option, info in sorted(option_ledger.items()):
            if info.get("verdict") == "disputed":
                lines.append(
                    f"  {option}: {info.get('support_count', 0)} support | DISPUTED"
                )
            else:
                lines.append(f"  {option}: {info.get('verdict', 'unknown').upper()}")
        return {
            "round": round_num,
            "compression_level": "none",
            "debate_capsule": "\n".join(lines),
            "disputed_options": [
                option for option, info in option_ledger.items()
                if info.get("verdict") == "disputed"
            ],
            "delta_from_last_round": {"stance_changes": [], "new_reasons": 0},
        }

    ledger_out: Dict[str, Dict[str, Any]] = {}
    for option, info in option_ledger.items():
        if info.get("verdict") == "disputed":
            ledger_out[option] = {
                "verdict": "disputed",
                "support": {
                    "count": info.get("support_count", 0),
                    "agents": info.get("support_agents", []),
                    "top_reason": "(summary unavailable)",
                },
                "oppose": {
                    "count": info.get("oppose_count", 0),
                    "agents": info.get("oppose_agents", []),
                    "top_reason": "(summary unavailable)",
                },
            }
        else:
            ledger_out[option] = {
                "verdict": info.get("verdict", "exclude"),
                "reason": "(summary unavailable)",
            }
    included = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "include"
    ]
    return {
        "round": round_num,
        "compression_level": level,
        "option_ledger": ledger_out,
        "final_answer": included,
        "current_answer": included,
        "frozen_options": [
            option for option, info in option_ledger.items()
            if info.get("verdict") != "disputed"
        ],
        "disputed_options": [
            option for option, info in option_ledger.items()
            if info.get("verdict") == "disputed"
        ],
        "delta_from_last_round": {"stance_changes": [], "new_reasons": 0},
    }


def _v5_selective_replacement_gate(
    verdict_history: List[OptionLedger],
    adaptive_stats: Dict[str, Any],
    round_log: Dict[str, Any],
    evidence_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether V5 may replace the full passage next round."""
    blocked_by_resolver = (
        round_log.get("exit_check") == "resolver_influence_blocked_continue"
        or (
            isinstance(round_log.get("resolver_influence_gate"), dict)
            and not round_log["resolver_influence_gate"].get("safe", True)
        )
    )
    stalled = _disputed_stalled_for_two_rounds(verdict_history, adaptive_stats)
    quality = _v5_snippet_quality(evidence_payload)
    replace = (
        not blocked_by_resolver
        and stalled.get("stalled") is True
        and quality["high_quality"] is True
    )
    reasons: List[str] = []
    if blocked_by_resolver:
        reasons.append("resolver_influence_blocked_keep_full_passage")
    if not stalled.get("stalled"):
        reasons.append(f"not_stalled:{stalled.get('reason')}")
    if not quality["high_quality"]:
        reasons.append("snippet_quality_below_threshold")
    if replace:
        reasons.append("disputed_stalled_with_high_quality_snippets")
    return {
        "policy": "v5_selective_replacement",
        "replace_full_passage": replace,
        "reasons": reasons,
        "stalled_dispute": stalled,
        "snippet_quality": quality,
        "criteria": [
            "all_stable safe exits before replacement",
            "resolver_influence_blocked keeps the full passage",
            "replace only when disputed support counts stall for two rounds",
            "replace only when every target option has verified support or contrast snippets",
            "replace only when every target option reaches the snippet score threshold",
        ],
    }


def _v5_snippet_quality(evidence_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Assess whether retrieved snippets are strong enough to replace passage."""
    snippets_by_option = evidence_payload.get("snippets_by_option", {})
    structured_by_option = evidence_payload.get("structured_snippets_by_option", {})
    targets = evidence_payload.get("evidence_target_options", [])
    if not isinstance(snippets_by_option, dict) or not targets:
        return {
            "high_quality": False,
            "min_top_score": 0.0,
            "threshold": V5_REPLACEMENT_MIN_SNIPPET_SCORE,
            "missing_options": list(targets) if isinstance(targets, list) else [],
            "missing_verified_options": list(targets) if isinstance(targets, list) else [],
        }

    top_scores: List[float] = []
    missing: List[str] = []
    missing_verified: List[str] = []
    for option in targets:
        snippets = snippets_by_option.get(option, [])
        if not snippets:
            missing.append(option)
            continue
        top_scores.append(float(snippets[0].get("score", 0.0) or 0.0))
        structured = structured_by_option.get(option, {})
        support = structured.get("support", []) if isinstance(structured, dict) else []
        contrast = structured.get("contrast", {}) if isinstance(structured, dict) else {}
        has_contrast = (
            isinstance(contrast, dict)
            and any(bool(items) for items in contrast.values())
        )
        if structured_by_option and not support and not has_contrast:
            missing_verified.append(option)
    min_top_score = min(top_scores) if top_scores else 0.0
    return {
        "high_quality": (
            not missing
            and not missing_verified
            and min_top_score >= V5_REPLACEMENT_MIN_SNIPPET_SCORE
        ),
        "min_top_score": round(min_top_score, 4),
        "threshold": V5_REPLACEMENT_MIN_SNIPPET_SCORE,
        "missing_options": missing,
        "missing_verified_options": missing_verified,
    }


def _build_v5_replacement_question(
    question: str,
    evidence_payload: Dict[str, Any],
) -> str:
    """Replace the full passage with retrieved snippets while keeping question text."""
    _, question_text = _split_passage_and_question(question)
    snippets_by_option = evidence_payload.get("snippets_by_option", {})
    structured = evidence_payload.get("structured_snippets_by_option", {})
    targets = evidence_payload.get("evidence_target_options", [])
    lines = [
        "Passage snippets:",
        "The full passage is intentionally replaced for this low-risk stalled dispute.",
        "Use only these retrieved snippets plus the debate summary/context.",
    ]
    for option in targets:
        lines.append(f"\nOption {option} evidence:")
        option_structured = structured.get(option, {})
        support = option_structured.get("support", [])
        oppose = option_structured.get("oppose", [])
        contrast = option_structured.get("contrast", {})
        if support:
            lines.append("  Support:")
            for idx, snippet in enumerate(support, start=1):
                lines.append(
                    f"    {idx}. (score={snippet.get('score')}) {snippet.get('text')}"
                )
        if oppose:
            lines.append("  Oppose:")
            for idx, snippet in enumerate(oppose, start=1):
                lines.append(
                    f"    {idx}. (score={snippet.get('score')}) {snippet.get('text')}"
                )
        if isinstance(contrast, dict) and contrast:
            lines.append("  Contrast:")
            for other_option, snippets in contrast.items():
                if not snippets:
                    continue
                lines.append(f"    vs {other_option}:")
                for idx, snippet in enumerate(snippets, start=1):
                    lines.append(
                        f"      {idx}. (score={snippet.get('score')}) {snippet.get('text')}"
                    )
        if not support and not oppose and not contrast:
            for idx, snippet in enumerate(snippets_by_option.get(option, []), start=1):
                lines.append(
                    f"  {idx}. (score={snippet.get('score')}) {snippet.get('text')}"
                )
    lines.append(f"\nQuestion:\n{question_text}")
    return "\n".join(lines)


def _build_v5_structured_evidence_context(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    method: str = "embedding",
    focus_options: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build V5.1 evidence with multi-query, contrast, MMR, and verification."""
    passage, question_text = _split_passage_and_question(question)
    disputed = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "disputed"
    ]
    evidence_targets = disputed
    if focus_options is not None:
        evidence_targets = [
            option for option in focus_options
            if option in option_ledger
        ]
    stable = [
        f"{option}:{info.get('verdict')}"
        for option, info in option_ledger.items()
        if info.get("verdict") != "disputed"
    ]
    option_texts = _parse_choice_texts(choices)
    chunks = _passage_chunks(passage)
    idf = _idf_for_chunks(chunks)
    actual_method = method
    retrieval_errors: List[str] = []

    snippets_by_option: Dict[str, List[Dict[str, Any]]] = {}
    structured_by_option: Dict[str, Dict[str, Any]] = {}
    queries_by_option: Dict[str, Dict[str, Any]] = {}

    for option in evidence_targets:
        option_text = option_texts.get(option, "")
        other_targets = [other for other in evidence_targets if other != option]
        direct_queries = _v5_multi_queries(question_text, option, option_text)
        contrast_queries = {
            other: _v5_contrast_queries(
                question_text,
                option,
                option_text,
                other,
                option_texts.get(other, ""),
            )
            for other in other_targets
        }
        queries_by_option[option] = {
            "direct": direct_queries,
            "contrast": contrast_queries,
        }

        candidates, used_method, error = _v5_retrieve_candidates(
            chunks,
            direct_queries,
            idf,
            method,
            query_type="direct",
        )
        actual_method, retrieval_errors = _v5_note_retrieval_status(
            actual_method,
            method,
            used_method,
            error,
            retrieval_errors,
        )

        contrast_candidates_by_other: Dict[str, List[Dict[str, Any]]] = {}
        for other, queries in contrast_queries.items():
            contrast_candidates, used_method, error = _v5_retrieve_candidates(
                chunks,
                queries,
                idf,
                method,
                query_type="contrast",
                contrast_with=other,
            )
            actual_method, retrieval_errors = _v5_note_retrieval_status(
                actual_method,
                method,
                used_method,
                error,
                retrieval_errors,
            )
            candidates.extend(contrast_candidates)
            contrast_candidates_by_other[other] = contrast_candidates

        base_query = f"{question_text}\nOption {option}: {option_text}"
        selected = _v5_mmr_select(
            base_query,
            _v5_merge_candidates(candidates),
            idf,
            method=actual_method,
            top_k=V5_STRUCTURED_FINAL_TOP_K,
        )

        verified: Dict[str, List[Dict[str, Any]]] = {
            "support": [],
            "oppose": [],
            "neutral": [],
        }
        flat_snippets: List[Dict[str, Any]] = []
        for candidate in selected:
            label, verification = _v5_verify_snippet(
                candidate.get("text", ""),
                question_text,
                option,
                option_text,
                method=actual_method,
            )
            snippet = {
                "score": round(float(candidate.get("score", 0.0) or 0.0), 4),
                "mmr_score": round(float(candidate.get("mmr_score", 0.0) or 0.0), 4),
                "label": label,
                "verification": verification,
                "query_type": candidate.get("query_type", "direct"),
                "contrast_with": candidate.get("contrast_with"),
                "text": candidate.get("text", ""),
            }
            verified[label].append(snippet)
            flat_snippets.append(snippet)

        contrast_structured: Dict[str, List[Dict[str, Any]]] = {}
        for other, contrast_candidates in contrast_candidates_by_other.items():
            contrast_query = " ".join(contrast_queries.get(other, []))
            selected_contrast = _v5_mmr_select(
                contrast_query,
                _v5_merge_candidates(contrast_candidates),
                idf,
                method=actual_method,
                top_k=V5_STRUCTURED_CONTRAST_TOP_K,
            )
            contrast_structured[other] = [
                {
                    "score": round(float(item.get("score", 0.0) or 0.0), 4),
                    "mmr_score": round(float(item.get("mmr_score", 0.0) or 0.0), 4),
                    "text": item.get("text", ""),
                }
                for item in selected_contrast
            ]

        structured_by_option[option] = {
            "support": verified["support"],
            "oppose": verified["oppose"],
            "neutral": verified["neutral"],
            "contrast": contrast_structured,
        }
        snippets_by_option[option] = flat_snippets

    lines = [
        "Use these structured evidence snippets as pointers, not as a replacement for the full passage unless explicitly stated.",
        f"Stable options: {', '.join(stable) if stable else 'none'}",
        f"Disputed options: {', '.join(disputed) if disputed else 'none'}",
        f"Evidence target options: {', '.join(evidence_targets) if evidence_targets else 'none'}",
    ]
    for option in evidence_targets:
        option_text = option_texts.get(option, "")
        option_structured = structured_by_option.get(option, {})
        lines.append(f"\nOption {option}: {option_text}")
        has_snippet = False
        for label in ("support", "oppose", "neutral"):
            snippets = option_structured.get(label, [])
            if not snippets:
                continue
            has_snippet = True
            lines.append(f"  {label.title()}:")
            for idx, snippet in enumerate(snippets, start=1):
                lines.append(
                    f"    {idx}. (score={snippet['score']}) {snippet['text']}"
                )
        contrast = option_structured.get("contrast", {})
        if isinstance(contrast, dict) and contrast:
            lines.append("  Contrast:")
            for other_option, snippets in contrast.items():
                if not snippets:
                    continue
                has_snippet = True
                lines.append(f"    vs {other_option}:")
                for idx, snippet in enumerate(snippets, start=1):
                    lines.append(
                        f"      {idx}. (score={snippet['score']}) {snippet['text']}"
                    )
        if not has_snippet:
            lines.append("  - No useful snippet retrieved; inspect the full passage.")

    payload = {
        "enabled": True,
        "method": actual_method,
        "requested_method": method,
        "strategy": "multi_query_option_contrast_mmr_verification",
        "embedding_model": getattr(_config, "EVIDENCE_EMBEDDING_MODEL", None)
        if method == "embedding"
        else None,
        "retrieval_error": "; ".join(sorted(set(retrieval_errors))) or None,
        "replaces_full_passage": False,
        "disputed_options": disputed,
        "evidence_target_options": evidence_targets,
        "stable_options": stable,
        "snippet_count": sum(len(v) for v in snippets_by_option.values()),
        "snippets_by_option": snippets_by_option,
        "structured_snippets_by_option": structured_by_option,
        "queries_by_option": queries_by_option,
        "mmr": {
            "lambda": V5_MMR_LAMBDA,
            "candidate_top_k": V5_STRUCTURED_CANDIDATE_TOP_K,
            "final_top_k": V5_STRUCTURED_FINAL_TOP_K,
            "contrast_top_k": V5_STRUCTURED_CONTRAST_TOP_K,
        },
    }
    return "\n".join(lines), payload


def _v5_note_retrieval_status(
    actual_method: str,
    requested_method: str,
    used_method: str,
    error: Optional[str],
    errors: List[str],
) -> Tuple[str, List[str]]:
    """Track embedding fallback status for a V5.1 retrieval batch."""
    if used_method != requested_method:
        actual_method = used_method
    if error:
        errors.append(error)
    return actual_method, errors


def _v5_multi_queries(question_text: str, option: str, option_text: str) -> List[str]:
    """Create multiple retrieval queries for one option."""
    return [
        f"{question_text}\nCandidate answer {option}: {option_text}",
        f"What passage evidence supports answer {option}: {option_text}?\nQuestion: {question_text}",
        f"What passage evidence contradicts answer {option}: {option_text}?\nQuestion: {question_text}",
        f"Key entities and events for answer {option}: {option_text}\nQuestion: {question_text}",
    ]


def _v5_contrast_queries(
    question_text: str,
    option: str,
    option_text: str,
    other_option: str,
    other_text: str,
) -> List[str]:
    """Create option-level contrast queries."""
    return [
        (
            f"{question_text}\nEvidence that {option}: {option_text} "
            f"is correct rather than {other_option}: {other_text}"
        ),
        (
            f"{question_text}\nEvidence distinguishing {option}: {option_text} "
            f"from {other_option}: {other_text}"
        ),
        (
            f"{question_text}\nWhy would {other_option}: {other_text} be wrong "
            f"if {option}: {option_text} is correct?"
        ),
    ]


def _v5_retrieve_candidates(
    chunks: List[str],
    queries: List[str],
    idf: Dict[str, float],
    method: str,
    query_type: str,
    contrast_with: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Retrieve candidate chunks for a list of V5.1 queries."""
    candidates: List[Dict[str, Any]] = []
    actual_method = method
    retrieval_error: Optional[str] = None
    for query in queries:
        if method == "embedding":
            try:
                ranked = _rank_evidence_chunks_embedding(
                    chunks,
                    query,
                    top_k=V5_STRUCTURED_CANDIDATE_TOP_K,
                )
            except Exception as exc:
                ranked = _rank_evidence_chunks(
                    chunks,
                    query,
                    idf,
                    top_k=V5_STRUCTURED_CANDIDATE_TOP_K,
                )
                actual_method = "embedding_fallback_lexical"
                retrieval_error = str(exc)
        else:
            ranked = _rank_evidence_chunks(
                chunks,
                query,
                idf,
                top_k=V5_STRUCTURED_CANDIDATE_TOP_K,
            )
        for score, text in ranked:
            candidates.append({
                "score": float(score or 0.0),
                "text": " ".join(text.split()),
                "query": query,
                "query_type": query_type,
                "contrast_with": contrast_with,
            })
    return candidates, actual_method, retrieval_error


def _v5_merge_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate candidates and keep the strongest metadata."""
    merged: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        text = " ".join(str(candidate.get("text", "")).split())
        if not text:
            continue
        existing = merged.get(text)
        if existing is None:
            merged[text] = {
                **candidate,
                "text": text,
                "source_queries": [candidate.get("query", "")],
                "query_types": [candidate.get("query_type", "")],
            }
            continue
        if float(candidate.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
            existing["score"] = candidate.get("score", 0.0)
            existing["query"] = candidate.get("query", "")
            existing["query_type"] = candidate.get("query_type", "")
        if candidate.get("query"):
            existing.setdefault("source_queries", []).append(candidate["query"])
        if candidate.get("query_type"):
            existing.setdefault("query_types", []).append(candidate["query_type"])
        if candidate.get("contrast_with") and not existing.get("contrast_with"):
            existing["contrast_with"] = candidate.get("contrast_with")
    return list(merged.values())


def _v5_mmr_select(
    query: str,
    candidates: List[Dict[str, Any]],
    idf: Dict[str, float],
    method: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Select relevant but non-redundant snippets with MMR."""
    if not candidates or top_k <= 0:
        return []
    if method == "embedding":
        try:
            query_embedding = _ollama_embedding(query)
            candidate_embeddings = [
                _ollama_embedding(candidate.get("text", ""))
                for candidate in candidates
            ]
            selected: List[int] = []
            remaining = list(range(len(candidates)))
            while remaining and len(selected) < top_k:
                best_idx = remaining[0]
                best_score = -1e9
                for idx in remaining:
                    relevance = _cosine_similarity(
                        query_embedding,
                        candidate_embeddings[idx],
                    )
                    redundancy = max(
                        _cosine_similarity(
                            candidate_embeddings[idx],
                            candidate_embeddings[selected_idx],
                        )
                        for selected_idx in selected
                    ) if selected else 0.0
                    mmr_score = (
                        V5_MMR_LAMBDA * relevance
                        - (1.0 - V5_MMR_LAMBDA) * redundancy
                    )
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                selected.append(best_idx)
                remaining.remove(best_idx)
                candidates[best_idx]["mmr_score"] = best_score
            return [candidates[idx] for idx in selected]
        except Exception:
            pass

    query_terms = set(_content_terms(query))
    selected_candidates: List[Dict[str, Any]] = []
    remaining = candidates[:]
    while remaining and len(selected_candidates) < top_k:
        best_idx = 0
        best_score = -1e9
        selected_terms = [
            set(_content_terms(item.get("text", "")))
            for item in selected_candidates
        ]
        for idx, candidate in enumerate(remaining):
            terms = set(_content_terms(candidate.get("text", "")))
            relevance = len(query_terms & terms) / max(len(query_terms), 1)
            redundancy = max(
                len(terms & prev_terms) / max(len(terms | prev_terms), 1)
                for prev_terms in selected_terms
            ) if selected_terms else 0.0
            score = V5_MMR_LAMBDA * relevance - (1.0 - V5_MMR_LAMBDA) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = best_score
        selected_candidates.append(chosen)
    return selected_candidates


def _v5_verify_snippet(
    snippet_text: str,
    question_text: str,
    option: str,
    option_text: str,
    method: str,
) -> Tuple[str, Dict[str, Any]]:
    """Classify a snippet as support, oppose, or neutral for one option."""
    if method == "embedding":
        try:
            snippet_embedding = _ollama_embedding(snippet_text)
            support_query = (
                f"{question_text}\nThis passage supports answer "
                f"{option}: {option_text}"
            )
            oppose_query = (
                f"{question_text}\nThis passage contradicts answer "
                f"{option}: {option_text}"
            )
            support_score = _cosine_similarity(
                snippet_embedding,
                _ollama_embedding(support_query),
            )
            oppose_score = _cosine_similarity(
                snippet_embedding,
                _ollama_embedding(oppose_query),
            )
            margin = support_score - oppose_score
            if support_score >= V5_REPLACEMENT_MIN_SNIPPET_SCORE and margin >= 0.03:
                label = "support"
            elif oppose_score >= V5_REPLACEMENT_MIN_SNIPPET_SCORE and margin <= -0.03:
                label = "oppose"
            else:
                label = "neutral"
            return label, {
                "method": "embedding_similarity",
                "support_score": round(support_score, 4),
                "oppose_score": round(oppose_score, 4),
                "margin": round(margin, 4),
            }
        except Exception as exc:
            lexical_label, lexical_info = _v5_verify_snippet_lexical(
                snippet_text,
                question_text,
                option_text,
            )
            lexical_info["embedding_error"] = str(exc)
            return lexical_label, lexical_info
    return _v5_verify_snippet_lexical(snippet_text, question_text, option_text)


def _v5_verify_snippet_lexical(
    snippet_text: str,
    question_text: str,
    option_text: str,
) -> Tuple[str, Dict[str, Any]]:
    """Low-cost lexical fallback for evidence verification."""
    snippet_terms = set(_content_terms(snippet_text))
    option_terms = set(_content_terms(option_text))
    question_terms = set(_content_terms(question_text))
    option_overlap = len(snippet_terms & option_terms)
    question_overlap = len(snippet_terms & question_terms)
    if option_overlap >= 2 or (option_overlap >= 1 and question_overlap >= 2):
        label = "support"
    else:
        label = "neutral"
    return label, {
        "method": "lexical_overlap",
        "option_overlap": option_overlap,
        "question_overlap": question_overlap,
    }


def _build_question_aware_evidence_context(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    method: str = "lexical",
    focus_options: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build deterministic question-aware evidence snippets for disputed options.

    First V4 version uses lexical retrieval rather than an LLM: the passage is
    split into sentence windows, then ranked by overlap with the question and
    each disputed option.
    """
    passage, question_text = _split_passage_and_question(question)
    disputed = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "disputed"
    ]
    evidence_targets = disputed
    if focus_options is not None:
        evidence_targets = [
            option for option in focus_options
            if option in option_ledger
        ]
    stable = [
        f"{option}:{info.get('verdict')}"
        for option, info in option_ledger.items()
        if info.get("verdict") != "disputed"
    ]
    option_texts = _parse_choice_texts(choices)
    chunks = _passage_chunks(passage)
    idf = _idf_for_chunks(chunks)
    actual_method = method
    retrieval_error: Optional[str] = None

    snippets_by_option: Dict[str, List[Dict[str, Any]]] = {}
    seen_texts = set()
    for option in evidence_targets:
        option_text = option_texts.get(option, "")
        query = f"{question_text} {option} {option_text}"
        if method == "embedding":
            try:
                ranked = _rank_evidence_chunks_embedding(chunks, query, top_k=2)
            except Exception as exc:
                ranked = _rank_evidence_chunks(chunks, query, idf, top_k=2)
                actual_method = "embedding_fallback_lexical"
                retrieval_error = str(exc)
        else:
            ranked = _rank_evidence_chunks(chunks, query, idf, top_k=2)
        snippets: List[Dict[str, Any]] = []
        for score, text in ranked:
            normalized_text = " ".join(text.split())
            if normalized_text in seen_texts and len(ranked) > 1:
                continue
            seen_texts.add(normalized_text)
            snippets.append({
                "score": round(score, 4),
                "text": normalized_text,
            })
        snippets_by_option[option] = snippets[:2]

    lines = [
        "Use these retrieved snippets as evidence pointers, not as a replacement for the full passage.",
        f"Stable options: {', '.join(stable) if stable else 'none'}",
        f"Disputed options: {', '.join(disputed) if disputed else 'none'}",
        f"Evidence target options: {', '.join(evidence_targets) if evidence_targets else 'none'}",
    ]
    for option in evidence_targets:
        option_text = option_texts.get(option, "")
        lines.append(f"\nOption {option}: {option_text}")
        snippets = snippets_by_option.get(option, [])
        if not snippets:
            lines.append("  - No high-overlap snippet retrieved; inspect the full passage.")
            continue
        for idx, snippet in enumerate(snippets, start=1):
            lines.append(
                f"  {idx}. (score={snippet['score']}) {snippet['text']}"
            )

    payload = {
        "enabled": True,
        "method": actual_method,
        "requested_method": method,
        "embedding_model": getattr(_config, "EVIDENCE_EMBEDDING_MODEL", None)
        if method == "embedding"
        else None,
        "retrieval_error": retrieval_error,
        "replaces_full_passage": False,
        "disputed_options": disputed,
        "evidence_target_options": evidence_targets,
        "stable_options": stable,
        "snippet_count": sum(len(v) for v in snippets_by_option.values()),
        "snippets_by_option": snippets_by_option,
    }
    return "\n".join(lines), payload


def _split_passage_and_question(question: str) -> Tuple[str, str]:
    """Split normalized QuALITY-style question into passage and question text."""
    passage_marker = "Passage:"
    question_marker = "\n\nQuestion:"
    if question.startswith(passage_marker) and question_marker in question:
        passage, question_text = question.split(question_marker, 1)
        return passage[len(passage_marker):].strip(), question_text.strip()
    return "", question.strip()


def _parse_choice_texts(choices: str) -> Dict[str, str]:
    """Parse formatted choices into option-letter text."""
    parsed: Dict[str, str] = {}
    for line in choices.splitlines():
        match = re.match(r"^\s*([A-J])\.\s*(.*)$", line.strip())
        if match:
            parsed[match.group(1)] = match.group(2).strip()
    return parsed


def _passage_chunks(passage: str, max_chars: int = 650) -> List[str]:
    """Split passage into compact sentence windows for retrieval."""
    if not passage:
        return []
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", passage)
        if sentence.strip()
    ]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _rank_evidence_chunks_embedding(
    chunks: List[str],
    query: str,
    top_k: int = 2,
) -> List[Tuple[float, str]]:
    """Rank chunks by Ollama embedding cosine similarity."""
    if not chunks or not query.strip():
        return []
    query_embedding = _ollama_embedding(query)
    chunk_embeddings = [_ollama_embedding(chunk) for chunk in chunks]
    ranked = [
        (_cosine_similarity(query_embedding, embedding), chunk)
        for embedding, chunk in zip(chunk_embeddings, chunks)
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def _ollama_embedding(text: str) -> List[float]:
    """Return an embedding vector from Ollama with a small text cache."""
    model = getattr(_config, "EVIDENCE_EMBEDDING_MODEL", "nomic-embed-text")
    cache_key = (model, text)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    base_url = getattr(
        _config,
        "EVIDENCE_EMBEDDING_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/")
    try:
        response = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    embeddings = payload.get("embeddings")
    if embeddings and isinstance(embeddings, list):
        vector = embeddings[0]
    else:
        vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError("Ollama embedding response did not include a vector")

    result = [float(value) for value in vector]
    _EMBEDDING_CACHE[cache_key] = result
    return result


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity for two embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rank_evidence_chunks(
    chunks: List[str],
    query: str,
    idf: Dict[str, float],
    top_k: int = 2,
) -> List[Tuple[float, str]]:
    """Rank chunks by deterministic lexical overlap with query terms."""
    query_terms = _content_terms(query)
    if not chunks or not query_terms:
        return []
    query_set = set(query_terms)
    ranked: List[Tuple[float, str]] = []
    for chunk in chunks:
        chunk_terms = _content_terms(chunk)
        if not chunk_terms:
            continue
        chunk_counts = Counter(chunk_terms)
        score = 0.0
        for term in query_set:
            if term in chunk_counts:
                score += idf.get(term, 1.0) * min(chunk_counts[term], 3)
        if score > 0:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def _idf_for_chunks(chunks: List[str]) -> Dict[str, float]:
    """Compute a small IDF table for passage chunks."""
    if not chunks:
        return {}
    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(set(_content_terms(chunk)))
    total = len(chunks)
    return {
        term: math.log((1 + total) / (1 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }


def _content_terms(text: str) -> List[str]:
    """Tokenize text and drop common stop words."""
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "with", "by", "as", "is", "are", "was", "were", "be", "been",
        "being", "it", "that", "this", "these", "those", "from", "at",
        "which", "what", "why", "who", "whom", "whose", "when", "where",
        "how", "did", "does", "do", "not", "no", "yes", "they", "their",
        "them", "he", "she", "his", "her", "him", "you", "your", "its",
        "about", "into", "than", "then", "so", "if", "because", "but",
        "option", "answer", "question", "passage",
    }
    terms = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9']+", text)
    ]
    return [
        term for term in terms
        if len(term) > 2 and term not in stopwords
    ]


def _run_final_evidence_review(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
) -> Tuple[Optional[str], str, Usage]:
    """
    Use the strongest configured agent as final evidence judge.

    This adaptive-local override keeps baseline/mechanism behavior unchanged
    while giving unresolved V3 max-round cases a stronger final reviewer.
    """
    reviewer_id, reviewer_name = _select_strongest_final_review_agent()
    client, model, name = get_client(reviewer_id)
    disputed = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "disputed"
    ]
    standings = _format_option_standings(option_ledger)
    recent_votes = _format_recent_votes(answers, current_round)
    recent_responses = _format_recent_responses(history, current_round)

    prompt = f"""You are Agent {reviewer_id} ({name}), acting as the strongest final evidence judge.

The debate reached the maximum number of rounds without a safe early exit.

Question:
{question}

Options:
{choices}

Option standings:
{standings}

Disputed options: {', '.join(disputed) if disputed else 'none'}

Recent vote history:
{recent_votes}

Recent agent reasoning:
{recent_responses}

Task:
1. Re-evaluate the question using the passage/question evidence and option wording.
2. Compare the strongest candidate against the closest alternative.
3. Identify if the latest majority may be groupthink or resolver-induced agreement.
4. Do not choose by vote count alone.

Output format:
First line: "My answer is: X" where X is a single letter from the available options.
Then give a concise evidence-based justification in 3-6 sentences."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            usage = _zero_usage()
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
            text = response.choices[0].message.content or ""
            answer, repair_usage, repair_text = extract_or_repair_answer(
                text,
                choices,
                reviewer_id,
            )
            if repair_usage.get("total_tokens", 0) > 0:
                _add_usage(usage, repair_usage)
                text = f"{text}\n\n[Answer repair]\n{repair_text}"
            review_header = (
                f"[Final review agent: {reviewer_id} {reviewer_name}]\n"
            )
            return answer, review_header + text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Strong final evidence review failed: {exc}")

    return None, "", _zero_usage()


def _run_final_evidence_review_multi_answer(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
) -> Tuple[Optional[str], str, Usage]:
    """Use the strongest configured agent as final judge for answer sets."""
    reviewer_id, reviewer_name = _select_strongest_final_review_agent()
    client, model, name = get_client(reviewer_id)
    disputed = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "disputed"
    ]
    standings = _format_option_standings(option_ledger)
    recent_votes = _format_recent_votes(answers, current_round)
    recent_responses = _format_recent_responses(history, current_round)

    prompt = f"""You are Agent {reviewer_id} ({name}), acting as the strongest final evidence judge.

The debate reached the maximum number of rounds on a multiple-answer question.

Question:
{question}

Options:
{choices}

Option standings:
{standings}

Disputed options: {', '.join(disputed) if disputed else 'none'}

Recent vote history:
{recent_votes}

Recent agent reasoning:
{recent_responses}

Task:
1. Treat each option independently as include or exclude.
2. Include every option directly supported by the passage/question.
3. Exclude unsupported, contradicted, or merely plausible options.
4. Do not choose by vote count alone.

Output format:
First line: "My answers are: A,C" using comma-separated option letters in alphabetical order.
Then give a concise evidence-based justification in 3-6 sentences."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            usage = _zero_usage()
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
            text = response.choices[0].message.content or ""
            answer, repair_usage, repair_text = extract_or_repair_answer_set(
                text,
                choices,
                reviewer_id,
            )
            if repair_usage.get("total_tokens", 0) > 0:
                _add_usage(usage, repair_usage)
                text = f"{text}\n\n[Answer repair]\n{repair_text}"
            review_header = (
                f"[Final review agent: {reviewer_id} {reviewer_name}]\n"
            )
            return answer, review_header + text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Strong multi-answer final evidence review failed: {exc}")

    return None, "", _zero_usage()


def _select_strongest_final_review_agent() -> Tuple[int, str]:
    """Select the strongest available configured agent for final review."""
    priority = [
        "deepseek-r1:14b",
        "phi4:14b",
        "qwen3:8b",
        "glm4:9b",
        "qwen2.5:7b",
        "llama3.1:8b",
        "mistral:7b",
        "gemma4:e4b-it-q4_k_m",
        "command-r7b:7b",
    ]
    priority_index = {model: idx for idx, model in enumerate(priority)}

    def score(cfg: Dict[str, Any]) -> int:
        model = str(cfg.get("model", "")).lower()
        return priority_index.get(model, len(priority) + int(cfg.get("agent_id", 99)))

    selected = min(AGENT_CONFIGS, key=score)
    return int(selected["agent_id"]), str(selected.get("name", selected["model"]))


def _compute_calibrated_confidence(
    resolver: Dict[str, Any],
    adaptive_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calibrate resolver confidence from structured analysis and vote dynamics.

    The LLM supplies structured evidence/reasoning fields. Code maps those
    fields into bounded scores, then combines them with majority, margin, and
    entropy signals.
    """
    resolver_answer = _normalize_option(resolver.get("recommended_answer"))
    evidence_score = _evidence_grounding_score(resolver, resolver_answer)
    reason_score = _reason_quality_score(resolver, resolver_answer)
    contrast_score = _option_contrast_score(resolver, resolver_answer)
    diagnosis_score = _error_diagnosis_score(resolver)

    resolver_quality_score = _clamp01(
        0.35 * evidence_score
        + 0.30 * reason_score
        + 0.20 * contrast_score
        + 0.15 * diagnosis_score
    )

    current_majority = adaptive_stats.get("top_answer")
    majority_match = 1.0 if resolver_answer and current_majority == resolver_answer else 0.0
    margin = int(adaptive_stats.get("margin", 0) or 0)
    normalized_entropy = _metric_float(adaptive_stats, "normalized_entropy", 1.0)
    margin_score = _clamp01(margin / 3.0)
    entropy_score = _clamp01(1.0 - normalized_entropy)
    vote_confirmation_score = _clamp01(
        0.40 * majority_match
        + 0.30 * margin_score
        + 0.30 * entropy_score
    )

    calibrated_score = _clamp01(
        0.40 * vote_confirmation_score
        + 0.60 * resolver_quality_score
    )
    if calibrated_score >= CALIBRATED_HIGH_THRESHOLD:
        level = "high"
    elif calibrated_score >= CALIBRATED_MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "low"

    return {
        "resolver_answer": resolver_answer,
        "score": calibrated_score,
        "level": level,
        "resolver_quality_score": resolver_quality_score,
        "vote_confirmation_score": vote_confirmation_score,
        "components": {
            "evidence_score": evidence_score,
            "reason_score": reason_score,
            "contrast_score": contrast_score,
            "diagnosis_score": diagnosis_score,
            "majority_match": majority_match,
            "margin_score": margin_score,
            "entropy_score": entropy_score,
        },
        "raw_llm_confidence": str(resolver.get("confidence", "")).strip().lower()
        or "unknown",
        "thresholds": {
            "high": CALIBRATED_HIGH_THRESHOLD,
            "medium": CALIBRATED_MEDIUM_THRESHOLD,
        },
    }


def _normalize_option(value: Any) -> Optional[str]:
    """Normalize an option-like value to a single uppercase A-J letter."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    if not cleaned:
        return None
    letter = cleaned[0]
    if "A" <= letter <= "J":
        return letter
    return None


def _normalize_answer_set_value(value: Any) -> Optional[str]:
    """Normalize resolver output into a canonical A,C answer set."""
    if value is None:
        return None
    if isinstance(value, list):
        letters = [
            _normalize_option(item)
            for item in value
        ]
    else:
        letters = re.findall(r"[A-J]", str(value).upper())
    return _canonical_answer_string([letter for letter in letters if letter])


def _compute_calibrated_confidence_multi_answer(
    resolver: Dict[str, Any],
    adaptive_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Calibrate resolver confidence for a recommended answer set."""
    resolver_answer = _normalize_answer_set_value(
        resolver.get("recommended_answer") or resolver.get("recommended_answers")
    )
    raw_confidence = str(resolver.get("confidence", "")).strip().lower()
    llm_score = {"high": 0.85, "medium": 0.60, "low": 0.35}.get(raw_confidence, 0.45)
    current_majority = adaptive_stats.get("top_answer")
    majority_match = 1.0 if resolver_answer and current_majority == resolver_answer else 0.0
    margin = int(adaptive_stats.get("margin", 0) or 0)
    normalized_entropy = _metric_float(adaptive_stats, "normalized_entropy", 1.0)
    margin_score = _clamp01(margin / 3.0)
    entropy_score = _clamp01(1.0 - normalized_entropy)
    vote_confirmation_score = _clamp01(
        0.45 * majority_match
        + 0.30 * margin_score
        + 0.25 * entropy_score
    )
    calibrated_score = _clamp01(0.50 * vote_confirmation_score + 0.50 * llm_score)
    if calibrated_score >= CALIBRATED_HIGH_THRESHOLD:
        level = "high"
    elif calibrated_score >= CALIBRATED_MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "low"
    return {
        "resolver_answer": resolver_answer,
        "score": calibrated_score,
        "level": level,
        "resolver_quality_score": llm_score,
        "vote_confirmation_score": vote_confirmation_score,
        "components": {
            "majority_match": majority_match,
            "margin_score": margin_score,
            "entropy_score": entropy_score,
            "llm_confidence_score": llm_score,
        },
        "raw_llm_confidence": raw_confidence or "unknown",
        "thresholds": {
            "high": CALIBRATED_HIGH_THRESHOLD,
            "medium": CALIBRATED_MEDIUM_THRESHOLD,
        },
    }


def _evidence_grounding_score(
    resolver: Dict[str, Any],
    resolver_answer: Optional[str],
) -> float:
    """Map evidence grounding strength into a bounded score.

    Preferred resolver JSON includes a weak/medium/strong ``strength`` field.
    Older or imperfect outputs may only include support/weakness text; those
    are still scored so a missing field does not zero out useful analysis.
    """
    if not resolver_answer:
        return 0.0
    grounding = resolver.get("evidence_grounding", {})
    if not isinstance(grounding, dict):
        return 0.0

    strengths: Dict[str, float] = {}
    for option, info in grounding.items():
        normalized = _normalize_option(option)
        if not normalized or not isinstance(info, dict):
            continue
        strengths[normalized] = _grounding_item_score(info)

    recommended_strength = strengths.get(resolver_answer, 0.0)
    best_alternative = max(
        [
            score
            for option, score in strengths.items()
            if option != resolver_answer
        ]
        or [0.0]
    )
    return _clamp01(
        0.70 * recommended_strength
        + 0.30 * max(recommended_strength - best_alternative, 0.0)
    )


def _strength_to_score(value: Any) -> float:
    """Map weak/medium/strong strings to a numeric evidence score."""
    text = str(value or "").strip().lower()
    if text == "strong":
        return 1.0
    if text == "medium":
        return 0.6
    if text == "weak":
        return 0.2
    return 0.0


def _grounding_item_score(info: Dict[str, Any]) -> float:
    """Score one evidence grounding item with graceful schema fallback."""
    explicit_strength = _strength_to_score(info.get("strength"))
    if explicit_strength > 0.0:
        return explicit_strength

    support = str(info.get("support", "") or "").strip()
    weakness = str(info.get("weakness", "") or "").strip()
    if not support and not weakness:
        return 0.0

    support_score = _text_specificity_score(support)
    weakness_score = _text_specificity_score(weakness)
    return _clamp01(0.80 * support_score + 0.20 * max(1.0 - weakness_score, 0.0))


def _text_specificity_score(text: str) -> float:
    """Estimate whether a text field contains concrete, useful evidence."""
    cleaned = text.strip()
    if not cleaned:
        return 0.0

    lowered = cleaned.lower()
    weak_markers = (
        "n/a",
        "none",
        "unknown",
        "unavailable",
        "parse failed",
        "resolver unavailable",
        "no evidence",
        "not enough",
    )
    if any(marker in lowered for marker in weak_markers):
        return 0.0

    words = cleaned.split()
    length_score = _clamp01(len(words) / 18.0)
    has_specific_signal = any(
        char.isdigit() or char in {'"', "'", ":"}
        for char in cleaned
    )
    return _clamp01(length_score + (0.15 if has_specific_signal else 0.0))


def _reason_quality_score(
    resolver: Dict[str, Any],
    resolver_answer: Optional[str],
) -> float:
    """Normalize resolver reason quality for the recommended answer."""
    if not resolver_answer:
        return 0.0
    reason_quality = resolver.get("reason_quality", {})
    if not isinstance(reason_quality, dict):
        return 0.0

    info = None
    for option, candidate in reason_quality.items():
        if _normalize_option(option) == resolver_answer and isinstance(candidate, dict):
            info = candidate
            break
    if not isinstance(info, dict):
        return 0.0

    raw_score = _safe_float(info.get("score"), default=0.0)
    score = _clamp01(raw_score / 5.0)
    issue = str(info.get("issue", "")).lower()
    if any(term in issue for term in ("circular", "irrelevant", "unsupported")):
        score -= 0.2
    if any(term in issue for term in ("contradict", "misread")):
        score -= 0.3
    return _clamp01(score)


def _option_contrast_score(
    resolver: Dict[str, Any],
    resolver_answer: Optional[str],
) -> float:
    """Score whether resolver provides an explicit option contrast."""
    text = str(resolver.get("option_contrast", "") or "").strip()
    if not text:
        return 0.0
    upper_text = text.upper()
    score = 0.3
    if resolver_answer and resolver_answer in upper_text:
        score += 0.3
    mentioned_options = {
        option for option in re.findall(r"\b[A-J]\b", upper_text)
        if option != resolver_answer
    }
    if mentioned_options:
        score += 0.2
    markers = (
        "because",
        "whereas",
        "while",
        "rather than",
        "better",
        "stronger",
        "rules out",
        "contradicts",
    )
    lowered = text.lower()
    if any(marker in lowered for marker in markers):
        score += 0.2
    return _clamp01(score)


def _error_diagnosis_score(resolver: Dict[str, Any]) -> float:
    """Score whether the resolver names a concrete disagreement source."""
    diagnosis = resolver.get("error_diagnosis", {})
    if not isinstance(diagnosis, dict):
        return 0.0
    error_type = str(diagnosis.get("likely_error_type", "")).strip().lower()
    root_cause = str(diagnosis.get("root_cause", "") or "").strip()
    if error_type in {
        "evidence_miss",
        "question_misread",
        "concept_error",
        "reasoning_error",
    }:
        score = 0.8
    elif error_type == "ambiguous":
        score = 0.4
    else:
        score = 0.0
    if len(root_cause) >= 20:
        score += 0.2
    return _clamp01(score)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with a fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    """Clamp a float to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _metric_float(source: Dict[str, Any], key: str, default: float) -> float:
    """Read a float metric without replacing valid zero values."""
    value = source.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolver_confirmed(
    resolver: Dict[str, Any],
    adaptive_stats: Dict[str, Any],
    medium_confirmed_once: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check whether the next round confirms the resolver recommendation.

    The resolver only proposes a hypothesis. Early exit is allowed when the
    next agent round confirms that answer with a clear majority and low
    distribution entropy.
    """
    calibrated = _compute_calibrated_confidence(resolver, adaptive_stats)
    resolver_answer = calibrated["resolver_answer"]
    calibrated_level = calibrated["level"]
    top_answer = adaptive_stats.get("top_answer")
    margin = int(adaptive_stats.get("margin", 0) or 0)
    normalized_entropy = _metric_float(adaptive_stats, "normalized_entropy", 1.0)
    majority_matches = top_answer == resolver_answer

    confirmed = (
        resolver_answer is not None
        and majority_matches
        and margin >= ADAPTIVE_MIN_MARGIN
        and normalized_entropy <= ADAPTIVE_MAX_NORMALIZED_ENTROPY
        and (
            calibrated_level == "high"
            or (calibrated_level == "medium" and medium_confirmed_once)
        )
    )

    confirmation = {
        "resolver_answer": resolver_answer,
        "resolver_confidence": str(resolver.get("confidence", "")).strip().lower()
        or "unknown",
        "calibrated_score": calibrated["score"],
        "calibrated_level": calibrated_level,
        "calibrated_confidence": calibrated,
        "current_majority": top_answer,
        "margin": margin,
        "normalized_entropy": normalized_entropy,
        "majority_matches_resolver": majority_matches,
        "medium_confirmed_once": medium_confirmed_once,
        "confirmed": confirmed,
        "criteria": {
            "high_can_exit_immediately": True,
            "medium_requires_two_confirmations": True,
            "min_margin": ADAPTIVE_MIN_MARGIN,
            "max_normalized_entropy": ADAPTIVE_MAX_NORMALIZED_ENTROPY,
            "high_threshold": CALIBRATED_HIGH_THRESHOLD,
            "medium_threshold": CALIBRATED_MEDIUM_THRESHOLD,
        },
    }
    return confirmed, confirmation


def _resolver_confirmed_multi_answer(
    resolver: Dict[str, Any],
    adaptive_stats: Dict[str, Any],
    medium_confirmed_once: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Check whether the next round confirms a resolver answer set."""
    calibrated = _compute_calibrated_confidence_multi_answer(resolver, adaptive_stats)
    resolver_answer = calibrated["resolver_answer"]
    calibrated_level = calibrated["level"]
    top_answer = adaptive_stats.get("top_answer")
    margin = int(adaptive_stats.get("margin", 0) or 0)
    normalized_entropy = _metric_float(adaptive_stats, "normalized_entropy", 1.0)
    majority_matches = top_answer == resolver_answer
    confirmed = (
        resolver_answer is not None
        and majority_matches
        and margin >= ADAPTIVE_MIN_MARGIN
        and normalized_entropy <= ADAPTIVE_MAX_NORMALIZED_ENTROPY
        and (
            calibrated_level == "high"
            or (calibrated_level == "medium" and medium_confirmed_once)
        )
    )
    confirmation = {
        "resolver_answer": resolver_answer,
        "resolver_confidence": str(resolver.get("confidence", "")).strip().lower()
        or "unknown",
        "calibrated_score": calibrated["score"],
        "calibrated_level": calibrated_level,
        "calibrated_confidence": calibrated,
        "current_majority": top_answer,
        "margin": margin,
        "normalized_entropy": normalized_entropy,
        "majority_matches_resolver": majority_matches,
        "medium_confirmed_once": medium_confirmed_once,
        "confirmed": confirmed,
        "answer_mode": "multi_answer_set",
        "criteria": {
            "high_can_exit_immediately": True,
            "medium_requires_two_confirmations": True,
            "min_margin": ADAPTIVE_MIN_MARGIN,
            "max_normalized_entropy": ADAPTIVE_MAX_NORMALIZED_ENTROPY,
        },
    }
    return confirmed, confirmation


def _resolver_trigger_snapshot(
    resolver: Dict[str, Any],
    adaptive_stats: Dict[str, Any],
    answers: Dict[int, Dict[int, Optional[str]]],
    trigger_round: int,
) -> Dict[str, Any]:
    """Capture pre-injection vote state for resolver influence detection."""
    resolver_answer = _normalize_option(resolver.get("recommended_answer"))
    trigger_answers = answers.get(trigger_round, {})
    trigger_counts = Counter(answer for answer in trigger_answers.values() if answer)
    r0_snapshot = _majority_snapshot(answers.get(0, {}))
    return {
        "trigger_round": trigger_round,
        "resolver_answer": resolver_answer,
        "trigger_top_answer": adaptive_stats.get("top_answer"),
        "trigger_margin": int(adaptive_stats.get("margin", 0) or 0),
        "trigger_support_count": trigger_counts.get(resolver_answer, 0)
        if resolver_answer
        else 0,
        "trigger_counts": dict(trigger_counts),
        "resolver_was_top": resolver_answer == adaptive_stats.get("top_answer"),
        "r0_majority_answer": r0_snapshot.get("majority_answer"),
        "r0_majority": r0_snapshot,
    }


def _resolver_trigger_snapshot_multi_answer(
    resolver: Dict[str, Any],
    adaptive_stats: Dict[str, Any],
    answers: Dict[int, Dict[int, Optional[str]]],
    trigger_round: int,
) -> Dict[str, Any]:
    """Capture pre-injection answer-set state for resolver influence detection."""
    resolver_answer = _normalize_answer_set_value(
        resolver.get("recommended_answer") or resolver.get("recommended_answers")
    )
    trigger_answers = answers.get(trigger_round, {})
    trigger_counts = Counter(answer for answer in trigger_answers.values() if answer)
    r0_snapshot = _majority_snapshot(answers.get(0, {}))
    return {
        "trigger_round": trigger_round,
        "resolver_answer": resolver_answer,
        "trigger_top_answer": adaptive_stats.get("top_answer"),
        "trigger_margin": int(adaptive_stats.get("margin", 0) or 0),
        "trigger_support_count": trigger_counts.get(resolver_answer, 0)
        if resolver_answer
        else 0,
        "trigger_counts": dict(trigger_counts),
        "resolver_was_top": resolver_answer == adaptive_stats.get("top_answer"),
        "r0_majority_answer": r0_snapshot.get("majority_answer"),
        "r0_majority": r0_snapshot,
    }


def _resolver_influence_gate(
    resolver: Dict[str, Any],
    confirmation: Dict[str, Any],
    trigger_snapshot: Optional[Dict[str, Any]],
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
    agent_ids: List[int],
) -> Dict[str, Any]:
    """
    Detect resolver-induced agreement before allowing resolver-confirmed exit.

    This gate is intentionally different from the all_stable temporal gate:
    it asks whether the resolver recommendation appears to have caused a rapid
    vote collapse rather than being independently supported before injection.
    """
    resolver_answer = _normalize_option(confirmation.get("resolver_answer"))
    if not trigger_snapshot:
        trigger_snapshot = _resolver_trigger_snapshot(
            resolver,
            {
                "top_answer": None,
                "margin": 0,
            },
            answers,
            max(0, current_round - 1),
        )

    current_counts = Counter(
        answer for answer in answers.get(current_round, {}).values() if answer
    )
    current_support_count = current_counts.get(resolver_answer, 0) if resolver_answer else 0
    trigger_support_count = int(trigger_snapshot.get("trigger_support_count", 0) or 0)
    support_jump = current_support_count - trigger_support_count
    trigger_margin = int(trigger_snapshot.get("trigger_margin", 0) or 0)
    r0_majority_answer = trigger_snapshot.get("r0_majority_answer")
    resolver_was_top = bool(trigger_snapshot.get("resolver_was_top"))
    thresholds = _n_aware_thresholds(len(agent_ids))

    risk_reasons: List[str] = []
    if trigger_margin < thresholds["weak_margin_threshold"]:
        risk_reasons.append("weak_pre_resolver_margin")
    if support_jump >= thresholds["large_jump_threshold"]:
        risk_reasons.append("large_post_resolver_support_jump")
    if resolver_answer and r0_majority_answer and resolver_answer != r0_majority_answer:
        risk_reasons.append("resolver_answer_differs_from_r0_majority")
    if not resolver_was_top:
        risk_reasons.append("resolver_answer_was_not_pre_resolver_top")

    risk_score = len(risk_reasons)
    safe = (
        resolver_answer is not None
        and confirmation.get("confirmed") is True
        and risk_score <= 1
    )

    return {
        "version": ADAPTIVE_RESOLVER_V3_VARIANT,
        "safe": safe,
        "risk_score": risk_score,
        "risk_reasons": risk_reasons,
        "resolver_answer": resolver_answer,
        "trigger_round": trigger_snapshot.get("trigger_round"),
        "trigger_top_answer": trigger_snapshot.get("trigger_top_answer"),
        "trigger_margin": trigger_margin,
        "trigger_support_count": trigger_support_count,
        "current_round": current_round,
        "current_support_count": current_support_count,
        "support_jump": support_jump,
        "r0_majority_answer": r0_majority_answer,
        "resolver_was_top": resolver_was_top,
        "thresholds": thresholds,
        "criteria": {
            "safe_requires": [
                "base resolver confirmation is true",
                "risk_score <= 1",
            ],
            "risk_reasons": {
                "weak_pre_resolver_margin": (
                    "trigger_margin < weak_margin_threshold"
                ),
                "large_post_resolver_support_jump": (
                    "support_jump >= large_jump_threshold"
                ),
                "resolver_answer_differs_from_r0_majority": "resolver_answer != R0 majority",
                "resolver_answer_was_not_pre_resolver_top": "resolver was not top before injection",
            },
        },
    }


def _resolver_influence_gate_multi_answer(
    resolver: Dict[str, Any],
    confirmation: Dict[str, Any],
    trigger_snapshot: Optional[Dict[str, Any]],
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
    agent_ids: List[int],
) -> Dict[str, Any]:
    """Resolver influence gate for answer-set recommendations."""
    resolver_answer = _normalize_answer_set_value(confirmation.get("resolver_answer"))
    if not trigger_snapshot:
        trigger_snapshot = _resolver_trigger_snapshot_multi_answer(
            resolver,
            {"top_answer": None, "margin": 0},
            answers,
            max(0, current_round - 1),
        )
    current_counts = Counter(
        answer for answer in answers.get(current_round, {}).values() if answer
    )
    current_support_count = current_counts.get(resolver_answer, 0) if resolver_answer else 0
    trigger_support_count = int(trigger_snapshot.get("trigger_support_count", 0) or 0)
    support_jump = current_support_count - trigger_support_count
    trigger_margin = int(trigger_snapshot.get("trigger_margin", 0) or 0)
    r0_majority_answer = trigger_snapshot.get("r0_majority_answer")
    resolver_was_top = bool(trigger_snapshot.get("resolver_was_top"))
    thresholds = _n_aware_thresholds(len(agent_ids))

    risk_reasons: List[str] = []
    if trigger_margin < thresholds["weak_margin_threshold"]:
        risk_reasons.append("weak_pre_resolver_margin")
    if support_jump >= thresholds["large_jump_threshold"]:
        risk_reasons.append("large_post_resolver_support_jump")
    if resolver_answer and r0_majority_answer and resolver_answer != r0_majority_answer:
        risk_reasons.append("resolver_answer_set_differs_from_r0_majority")
    if not resolver_was_top:
        risk_reasons.append("resolver_answer_set_was_not_pre_resolver_top")

    risk_score = len(risk_reasons)
    safe = (
        resolver_answer is not None
        and confirmation.get("confirmed") is True
        and risk_score <= 1
    )
    return {
        "version": "V3 Multi-Answer Resolver Influence Gate",
        "safe": safe,
        "risk_score": risk_score,
        "risk_reasons": risk_reasons,
        "resolver_answer": resolver_answer,
        "trigger_round": trigger_snapshot.get("trigger_round"),
        "trigger_top_answer": trigger_snapshot.get("trigger_top_answer"),
        "trigger_margin": trigger_margin,
        "trigger_support_count": trigger_support_count,
        "current_round": current_round,
        "current_support_count": current_support_count,
        "support_jump": support_jump,
        "r0_majority_answer": r0_majority_answer,
        "resolver_was_top": resolver_was_top,
        "thresholds": thresholds,
        "answer_mode": "multi_answer_set",
    }


def _format_resolver_influence_block_for_prompt(
    influence_gate: Dict[str, Any],
) -> str:
    """Weak prompt injected after a resolver-confirmed exit is blocked."""
    reasons = ", ".join(influence_gate.get("risk_reasons", [])) or "unknown"
    return (
        "The previous resolver recommendation was not accepted as a final "
        "answer because it may have induced agreement rather than independently "
        "resolved the dispute.\n"
        f"Blocked answer: {influence_gate.get('resolver_answer')}\n"
        f"Risk signals: {reasons}\n"
        "Re-check the disputed options independently against the question "
        "wording and evidence. Do not follow the resolver or majority blindly."
    )


def _format_resolver_for_prompt(
    resolver: Dict[str, Any],
    raw_text: str,
    mode: str = "full",
) -> str:
    """Format resolver output for the next debate round."""
    if mode == "weak":
        return (
            "Resolver could not confidently resolve the dispute.\n"
            "Re-check the disputed options independently against the question "
            "wording and evidence. Do not follow any prior majority blindly."
        )
    if resolver:
        return json.dumps(resolver, ensure_ascii=False, indent=2)
    return raw_text or "(no resolver analysis available)"


def _all_stable_answer_unchanged_local(verdict_history: List[OptionLedger]) -> bool:
    """Return whether the latest two ledgers are all-stable with same include."""
    if len(verdict_history) < 2:
        return False
    current = _included_answer_set(verdict_history[-1])
    previous = _included_answer_set(verdict_history[-2])
    return current is not None and current == previous


def _all_stable_safety_check(
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
    current_answer: Optional[str],
    agent_ids: List[int],
) -> Dict[str, Any]:
    """
    Check whether an all_stable consensus is temporally safe.

    Safety requires the current consensus to match the R0 majority, be
    supported by history-weighted voting, and avoid large answer migration.
    """
    r0_snapshot = _majority_snapshot(answers.get(0, {}))
    current_snapshot = _majority_snapshot(answers.get(current_round, {}))
    weighted_snapshot = _history_weighted_vote_snapshot(
        answers,
        current_round,
    )
    migration = _answer_migration_snapshot(
        answers.get(0, {}),
        answers.get(current_round, {}),
        current_answer,
        agent_ids,
    )

    r0_majority_consistent = (
        current_answer is not None
        and r0_snapshot["majority_answer"] == current_answer
        and r0_snapshot["majority_margin"] > 0
    )
    weighted_vote_supports_current = (
        current_answer is not None
        and weighted_snapshot["weighted_answer"] == current_answer
        and weighted_snapshot["weighted_margin"] > 0
    )
    large_answer_migration = migration["large_answer_migration"]
    safe = (
        r0_majority_consistent
        and weighted_vote_supports_current
        and not large_answer_migration
    )

    return {
        "version": ADAPTIVE_RESOLVER_VARIANT,
        "current_answer": current_answer,
        "safe": safe,
        "r0_majority_consistent": r0_majority_consistent,
        "weighted_vote_supports_current": weighted_vote_supports_current,
        "large_answer_migration": large_answer_migration,
        "r0_majority": r0_snapshot,
        "current_majority": current_snapshot,
        "history_weighted_vote": weighted_snapshot,
        "answer_migration": migration,
        "criteria": {
            "safe_requires": [
                "current_answer == R0 majority",
                "current_answer == history-weighted vote winner",
                "no large answer migration",
            ],
            "history_weights": _history_vote_weights(current_round),
        },
    }


def _majority_snapshot(
    answers_round: Dict[int, Optional[str]],
) -> Dict[str, Any]:
    """Return top-answer and margin diagnostics for one round."""
    valid = [answer for answer in answers_round.values() if answer]
    counts = Counter(valid)
    if not counts:
        return {
            "majority_answer": None,
            "majority_count": 0,
            "runner_up_count": 0,
            "majority_margin": 0,
            "counts": {},
        }
    top_answer, top_count, runner_up_count = _top_answer_counts(counts)
    return {
        "majority_answer": top_answer,
        "majority_count": top_count,
        "runner_up_count": runner_up_count,
        "majority_margin": top_count - runner_up_count,
        "counts": dict(counts),
    }


def _history_weighted_vote_snapshot(
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
) -> Dict[str, Any]:
    """Return history-weighted vote diagnostics up to the current round."""
    weights = _history_vote_weights(current_round)
    weighted_counts: Counter[str] = Counter()
    for round_num in range(0, current_round + 1):
        weight = weights.get(str(round_num), 1.0)
        for answer in answers.get(round_num, {}).values():
            if answer:
                weighted_counts[answer] += weight

    if not weighted_counts:
        return {
            "weighted_answer": None,
            "weighted_count": 0.0,
            "weighted_runner_up_count": 0.0,
            "weighted_margin": 0.0,
            "weighted_counts": {},
        }

    ranked = weighted_counts.most_common()
    top_answer, top_count = ranked[0]
    runner_up_count = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "weighted_answer": top_answer,
        "weighted_count": top_count,
        "weighted_runner_up_count": runner_up_count,
        "weighted_margin": top_count - runner_up_count,
        "weighted_counts": dict(weighted_counts),
    }


def _history_vote_weights(current_round: int) -> Dict[str, float]:
    """Weights used by the all_stable temporal consistency check."""
    weights: Dict[str, float] = {}
    for round_num in range(0, current_round + 1):
        if round_num <= 1:
            weight = 1.0
        elif round_num == 2:
            weight = 1.2
        else:
            weight = 1.5
        weights[str(round_num)] = weight
    return weights


def _answer_migration_snapshot(
    r0_answers: Dict[int, Optional[str]],
    current_answers: Dict[int, Optional[str]],
    current_answer: Optional[str],
    agent_ids: List[int],
) -> Dict[str, Any]:
    """Detect whether many agents migrated together to the final consensus."""
    changed_agents: List[int] = []
    migrated_to_current: List[int] = []
    for agent_id in agent_ids:
        r0_answer = r0_answers.get(agent_id)
        latest_answer = current_answers.get(agent_id)
        if r0_answer != latest_answer:
            changed_agents.append(agent_id)
        if (
            current_answer is not None
            and latest_answer == current_answer
            and r0_answer != current_answer
        ):
            migrated_to_current.append(agent_id)

    threshold = _n_aware_thresholds(len(agent_ids))["large_migration_threshold"]
    large_answer_migration = len(migrated_to_current) >= threshold
    return {
        "changed_agents": changed_agents,
        "migrated_to_current_agents": migrated_to_current,
        "changed_count": len(changed_agents),
        "migrated_to_current_count": len(migrated_to_current),
        "migration_threshold": threshold,
        "large_answer_migration": large_answer_migration,
    }


def _n_aware_thresholds(agent_count: int) -> Dict[str, int]:
    """Return thresholds scaled by the active number of agents."""
    return {
        "agent_count": agent_count,
        "weak_margin_threshold": math.ceil(0.30 * agent_count),
        "large_jump_threshold": math.ceil(0.40 * agent_count),
        "large_migration_threshold": math.ceil(0.60 * agent_count),
    }


def _included_answer(option_ledger: OptionLedger) -> Optional[str]:
    """Return a single included option if one exists."""
    included = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "include"
    ]
    if len(included) == 1:
        return included[0]
    return None


def _canonical_answer_string(options: List[str]) -> Optional[str]:
    """Return A,C style answer set or None."""
    normalized = sorted({option for option in options if re.fullmatch(r"[A-Z]", option)})
    return ",".join(normalized) if normalized else None


def _included_answer_set(option_ledger: OptionLedger) -> Optional[str]:
    """Return all included options from an option ledger."""
    included = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "include"
    ]
    return _canonical_answer_string(included)


def _get_final_answer_multi(
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
    option_ledger: OptionLedger,
    multi_answer: bool,
) -> Optional[str]:
    """Return final answer for single-answer or multi-answer mode."""
    if not multi_answer:
        return _get_final_answer(answers, current_round, option_ledger)
    included = _included_answer_set(option_ledger)
    if included:
        return included
    latest = [
        answer for answer in answers.get(current_round, {}).values()
        if answer
    ]
    if latest:
        return Counter(latest).most_common(1)[0][0]
    return None


def _build_adaptive_result(
    history: Dict[int, Dict[int, str]],
    usages: Dict[int, Dict[int, Usage]],
    answers: Dict[int, Dict[int, Optional[str]]],
    agent_ids: List[int],
    final_answer: Optional[str],
    actual_rounds: int,
    exit_reason: str,
    mechanism_log: List[Dict[str, Any]],
    summarizer_usage: Usage,
    dispute_resolver_usage: Usage,
    final_review_usage: Optional[Usage] = None,
    answer_repair_log: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    variant: str = ADAPTIVE_RESOLVER_VARIANT,
) -> Dict[str, Any]:
    """Build a baseline-compatible result with adaptive resolver metadata."""
    if final_review_usage is None:
        final_review_usage = _zero_usage()
    if answer_repair_log is None:
        answer_repair_log = {}

    token_by_round: Dict[str, Usage] = {}
    total_usage = _zero_usage()
    for round_num in sorted(history):
        round_total = _zero_usage()
        for agent_id in agent_ids:
            if agent_id in usages.get(round_num, {}):
                _add_usage(round_total, usages[round_num][agent_id])
        token_by_round[str(round_num)] = round_total
        _add_usage(total_usage, round_total)

    total_with_summarizer = total_usage.copy()
    _add_usage(total_with_summarizer, summarizer_usage)
    _add_usage(total_with_summarizer, dispute_resolver_usage)
    _add_usage(total_with_summarizer, final_review_usage)

    return {
        "history": history,
        "answers_by_round": {
            str(round_num): {
                str(agent_id): answers[round_num].get(agent_id)
                for agent_id in agent_ids
            }
            for round_num in sorted(answers)
        },
        "final_answer": final_answer,
        "token_usage": {
            "by_round": token_by_round,
            "total": total_usage,
            "summarizer": summarizer_usage,
            "dispute_resolver": dispute_resolver_usage,
            "final_review": final_review_usage,
            "total_with_summarizer": total_with_summarizer,
        },
        "mechanism": {
            "variant": variant,
            "actual_rounds": actual_rounds,
            "early_exit": exit_reason != "max_rounds",
            "exit_reason": exit_reason,
            "mechanism_log": mechanism_log,
            "answer_repair_log": answer_repair_log,
        },
    }
