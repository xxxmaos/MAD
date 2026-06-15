"""
Summarizer and Debate Capsule generation for mechanism-based debate.

FULL, PARTIAL, and DISPUTE summaries use Agent 1's Qwen model. NONE mode uses
pure Python string construction so it adds no LLM cost.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from agent import extract_usage, get_client
from config import MAX_RETRIES, RETRY_DELAY, SUMMARIZER_AGENT_ID
from verdict import OptionLedger, get_disputed_options, get_frozen_options

Usage = Dict[str, int]
SummaryDict = Dict[str, Any]


def call_summarizer(prompt_text: str) -> Tuple[str, Usage]:
    """
    Call the Qwen summarizer with deterministic decoding.

    Args:
        prompt_text: Prompt to send to the summarizer.

    Returns:
        Tuple of response text and usage dict.
    """
    client, model, _ = get_client(SUMMARIZER_AGENT_ID)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.1,
                max_tokens=800,
            )
            usage = extract_usage(response)
            return response.choices[0].message.content or "", usage
        except Exception as exc:
            print(f"Summarizer attempt {attempt + 1} failed: {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise


def generate_summary(
    compression_level: str,
    option_ledger: OptionLedger,
    responses: Dict[int, str],
    prev_summary: Optional[SummaryDict] = None,
) -> SummaryDict:
    """
    Generate the current round summary for the requested compression level.

    Args:
        compression_level: One of "full", "partial", "none", or "dispute".
        option_ledger: Current option-level verdict ledger.
        responses: Raw agent responses for the current round.
        prev_summary: Previous summary, if any.

    Returns:
        Summary dict with usage metadata.
    """
    if compression_level == "full":
        return _generate_full_summary(option_ledger, responses, prev_summary)
    if compression_level == "partial":
        return _generate_partial_summary(option_ledger, responses, prev_summary)
    if compression_level == "dispute":
        return _generate_dispute_summary(option_ledger, responses, prev_summary)
    return _generate_capsule(option_ledger, responses, prev_summary)


def _generate_full_summary(
    option_ledger: OptionLedger,
    responses: Dict[int, str],
    prev_summary: Optional[SummaryDict],
) -> SummaryDict:
    """Generate a FULL compression summary using the summarizer LLM."""
    round_num = _infer_round(prev_summary)
    prompt = f"""You are a debate summarizer. All {_num_agents(option_ledger)} agents agreed on every option this round.

Agent responses:
{_format_responses(responses)}

Previous round summary:
{_format_prev_summary(prev_summary)}

Task:
1. For each option, write ONE sentence explaining why it is included or excluded.
2. Note any changes from the previous round (which agents changed stance, on which options).
3. Output strict JSON only, no other text.

JSON format:
{{
  "round": {round_num},
  "compression_level": "full",
  "option_ledger": {{
    "<option>": {{"verdict": "include or exclude", "reason": "one sentence"}}
  }},
  "final_answer": ["list of included options"],
  "delta_from_last_round": {{
    "stance_changes": [{{"agent": <id>, "option": "<opt>", "from": "<old>", "to": "<new>"}}],
    "options_newly_stabilized": ["list"],
    "new_reasons": <count>
  }}
}}"""
    text, usage = call_summarizer(prompt)
    return _parse_or_fallback(
        text,
        usage,
        "full",
        option_ledger,
        responses,
        prev_summary,
        round_num,
    )


def _generate_partial_summary(
    option_ledger: OptionLedger,
    responses: Dict[int, str],
    prev_summary: Optional[SummaryDict],
) -> SummaryDict:
    """Generate a PARTIAL compression summary using the summarizer LLM."""
    round_num = _infer_round(prev_summary)
    frozen = get_frozen_options(option_ledger)
    disputed = get_disputed_options(option_ledger)
    frozen_with_verdicts = {
        opt: option_ledger[opt]["verdict"]
        for opt in frozen
    }

    prompt = f"""You are a debate summarizer. Some options have reached consensus, others are still disputed.

Agent responses:
{_format_responses(responses)}

Previous round summary:
{_format_prev_summary(prev_summary)}

Stable options: {json.dumps(frozen_with_verdicts)}
Disputed options: {json.dumps(disputed)}

Task:
1. For stable options, write ONE sentence each explaining the reason.
2. For disputed options, list the top-1 support reason AND top-1 oppose reason separately. Do NOT merge support and oppose reasons.
3. For disputed options, list which agents support and which oppose.
4. Note changes from previous round.
5. Output strict JSON only.

JSON format:
{{
  "round": {round_num},
  "compression_level": "partial",
  "option_ledger": {{
    "<stable_option>": {{"verdict": "include/exclude", "reason": "one sentence"}},
    "<disputed_option>": {{
      "verdict": "disputed",
      "support": {{"count": <n>, "agents": [<ids>], "top_reason": "..."}},
      "oppose": {{"count": <n>, "agents": [<ids>], "top_reason": "..."}}
    }}
  }},
  "frozen_options": ["list"],
  "disputed_options": ["list"],
  "current_answer": ["majority vote answer"],
  "delta_from_last_round": {{
    "stance_changes": [],
    "options_newly_stabilized": [],
    "new_reasons": <count>
  }}
}}"""
    text, usage = call_summarizer(prompt)
    return _parse_or_fallback(
        text,
        usage,
        "partial",
        option_ledger,
        responses,
        prev_summary,
        round_num,
    )


def _generate_capsule(
    option_ledger: OptionLedger,
    responses: Dict[int, str],
    prev_summary: Optional[SummaryDict],
) -> SummaryDict:
    """
    Generate a zero-cost Debate Capsule by formatting ledger statistics.

    Args:
        option_ledger: Current option-level verdict ledger.
        responses: Raw agent responses, unused except for round inference parity.
        prev_summary: Previous summary.

    Returns:
        Capsule summary dict with zero token usage.
    """
    del responses
    round_num = _infer_round(prev_summary)
    frozen = get_frozen_options(option_ledger)
    disputed = get_disputed_options(option_ledger)
    num_agents = _num_agents(option_ledger)

    lines = ["[Debate Status]", "Option standings:"]
    for opt, info in sorted(option_ledger.items()):
        if info["verdict"] == "disputed":
            lines.append(
                f"  {opt}: {info['support_count']}/{num_agents} include | DISPUTED"
            )
        else:
            lines.append(f"  {opt}: {info['verdict']} (all agents agree)")

    lines.append(f"\nStable options: {', '.join(frozen) if frozen else 'none'}")
    lines.append(
        f"Options under debate: {', '.join(disputed) if disputed else 'none'}"
    )

    prev_none = 0
    if prev_summary and prev_summary.get("compression_level") == "none":
        prev_none = int(prev_summary.get("consecutive_none_count", 0) or 0)

    return {
        "round": round_num,
        "compression_level": "none",
        "option_ledger": option_ledger,
        "_raw_option_ledger": option_ledger,
        "frozen_options": frozen,
        "disputed_options": disputed,
        "debate_capsule": "\n".join(lines),
        "delta_from_last_round": _compute_delta(option_ledger, prev_summary),
        "consecutive_none_count": prev_none + 1,
        "usage": _zero_usage(),
    }


def _generate_dispute_summary(
    option_ledger: OptionLedger,
    responses: Dict[int, str],
    prev_summary: Optional[SummaryDict],
) -> SummaryDict:
    """Generate a deadlock-oriented dispute summary using the summarizer LLM."""
    round_num = _infer_round(prev_summary)
    disputed = get_disputed_options(option_ledger)
    disputed_details = {
        opt: {
            "support_count": option_ledger[opt]["support_count"],
            "support_agents": option_ledger[opt]["support_agents"],
            "oppose_count": option_ledger[opt]["oppose_count"],
            "oppose_agents": option_ledger[opt]["oppose_agents"],
        }
        for opt in disputed
    }

    prompt = f"""You are a debate summarizer. Agents have debated for several rounds but multiple options remain disputed.

Agent responses:
{_format_responses(responses)}

Disputed options and their vote history:
{json.dumps(disputed_details, indent=2)}

Task:
1. For each disputed option, list the support and oppose core arguments (each <=50 tokens).
2. Identify the root cause of disagreement (one sentence).
3. Suggest which option to prioritize discussing next round.
4. Output strict JSON only.

JSON format:
{{
  "round": {round_num},
  "compression_level": "dispute",
  "disputed_analysis": {{
    "<option>": {{
      "verdict": "disputed",
      "support": {{"count": <n>, "agents": [<ids>], "top_reason": "..."}},
      "oppose": {{"count": <n>, "agents": [<ids>], "top_reason": "..."}},
      "vote_trend": "description of recent trend"
    }}
  }},
  "fundamental_disagreement": "one sentence",
  "priority_option": "<option>",
  "suggestion": "one sentence"
}}"""
    text, usage = call_summarizer(prompt)
    parsed = _parse_json(text)
    if parsed is None:
        parsed = _fallback_summary("dispute", option_ledger, prev_summary, round_num)

    return _finalize_summary(
        parsed,
        "dispute",
        option_ledger,
        prev_summary,
        round_num,
        usage,
    )


def _parse_or_fallback(
    text: str,
    usage: Usage,
    compression_level: str,
    option_ledger: OptionLedger,
    responses: Dict[int, str],
    prev_summary: Optional[SummaryDict],
    round_num: int,
) -> SummaryDict:
    """Parse summarizer JSON or build a deterministic fallback summary."""
    del responses
    parsed = _parse_json(text)
    if parsed is None:
        parsed = _fallback_summary(
            compression_level,
            option_ledger,
            prev_summary,
            round_num,
        )

    return _finalize_summary(
        parsed,
        compression_level,
        option_ledger,
        prev_summary,
        round_num,
        usage,
    )


def _finalize_summary(
    summary: SummaryDict,
    compression_level: str,
    option_ledger: OptionLedger,
    prev_summary: Optional[SummaryDict],
    round_num: int,
    usage: Usage,
) -> SummaryDict:
    """Attach required metadata that downstream code depends on."""
    summary["round"] = int(summary.get("round", round_num) or round_num)
    summary["compression_level"] = compression_level
    summary["_raw_option_ledger"] = option_ledger
    summary["frozen_options"] = summary.get("frozen_options", get_frozen_options(option_ledger))
    summary["disputed_options"] = summary.get(
        "disputed_options",
        get_disputed_options(option_ledger),
    )
    summary["delta_from_last_round"] = _normalize_delta(
        summary.get("delta_from_last_round"),
        option_ledger,
        prev_summary,
    )
    summary["usage"] = usage
    return summary


def _fallback_summary(
    compression_level: str,
    option_ledger: OptionLedger,
    prev_summary: Optional[SummaryDict],
    round_num: int,
) -> SummaryDict:
    """Build a minimal summary without relying on LLM JSON quality."""
    if compression_level == "dispute":
        disputed_analysis = {}
        for opt in get_disputed_options(option_ledger):
            info = option_ledger[opt]
            disputed_analysis[opt] = {
                "verdict": "disputed",
                "support": {
                    "count": info["support_count"],
                    "agents": info["support_agents"],
                    "top_reason": "Supporters selected this option in the current round.",
                },
                "oppose": {
                    "count": info["oppose_count"],
                    "agents": info["oppose_agents"],
                    "top_reason": "Opponents selected a different option in the current round.",
                },
                "vote_trend": "No parsed trend available.",
            }
        priority = next(iter(disputed_analysis), "")
        return {
            "round": round_num,
            "compression_level": "dispute",
            "disputed_analysis": disputed_analysis,
            "fundamental_disagreement": "Agents remain split across multiple options.",
            "priority_option": priority,
            "suggestion": "Focus on the option with the clearest support split next round.",
        }

    ledger_summary: Dict[str, Any] = {}
    for opt, info in sorted(option_ledger.items()):
        if info["verdict"] == "disputed":
            ledger_summary[opt] = {
                "verdict": "disputed",
                "support": {
                    "count": info["support_count"],
                    "agents": info["support_agents"],
                    "top_reason": "Supporters selected this option in the current round.",
                },
                "oppose": {
                    "count": info["oppose_count"],
                    "agents": info["oppose_agents"],
                    "top_reason": "Opponents selected a different option in the current round.",
                },
            }
        else:
            ledger_summary[opt] = {
                "verdict": info["verdict"],
                "reason": f"All agents {'selected' if info['verdict'] == 'include' else 'rejected'} option {opt}.",
            }

    included = [
        opt for opt, info in option_ledger.items() if info["verdict"] == "include"
    ]
    return {
        "round": round_num,
        "compression_level": compression_level,
        "option_ledger": ledger_summary,
        "final_answer": included,
        "current_answer": _majority_answers(option_ledger),
        "frozen_options": get_frozen_options(option_ledger),
        "disputed_options": get_disputed_options(option_ledger),
        "delta_from_last_round": _compute_delta(option_ledger, prev_summary),
    }


def _compute_delta(
    option_ledger: OptionLedger,
    prev_summary: Optional[SummaryDict],
) -> Dict[str, Any]:
    """Compute conservative delta metadata from the previous raw ledger."""
    previous = _previous_raw_ledger(prev_summary)
    if not previous:
        return {
            "stance_changes": [],
            "options_newly_stabilized": [],
            "new_reasons": 0,
        }

    stance_changes = []
    options_newly_stabilized = []
    for opt, info in option_ledger.items():
        prev_info = previous.get(opt)
        if not prev_info:
            continue
        if prev_info.get("verdict") != info["verdict"]:
            stance_changes.append(
                {
                    "agent": None,
                    "option": opt,
                    "from": prev_info.get("verdict"),
                    "to": info["verdict"],
                }
            )
            if info["verdict"] != "disputed":
                options_newly_stabilized.append(opt)

    return {
        "stance_changes": stance_changes,
        "options_newly_stabilized": options_newly_stabilized,
        "new_reasons": len(stance_changes),
    }


def _normalize_delta(
    candidate: Any,
    option_ledger: OptionLedger,
    prev_summary: Optional[SummaryDict],
) -> Dict[str, Any]:
    """Ensure delta has the expected keys and integer new_reasons."""
    fallback = _compute_delta(option_ledger, prev_summary)
    if not isinstance(candidate, dict):
        return fallback

    new_reasons = candidate.get("new_reasons", fallback["new_reasons"])
    try:
        new_reasons_int = int(new_reasons)
    except (TypeError, ValueError):
        new_reasons_int = fallback["new_reasons"]

    return {
        "stance_changes": candidate.get("stance_changes", fallback["stance_changes"]),
        "options_newly_stabilized": candidate.get(
            "options_newly_stabilized",
            fallback["options_newly_stabilized"],
        ),
        "new_reasons": new_reasons_int,
    }


def _parse_json(text: str) -> Optional[SummaryDict]:
    """Parse strict or fenced JSON from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _format_responses(responses: Dict[int, str]) -> str:
    """Format agent responses for summarizer prompts."""
    return "\n\n".join(
        f"Agent {agent_id}:\n{response}"
        for agent_id, response in sorted(responses.items())
    )


def _format_prev_summary(prev_summary: Optional[SummaryDict]) -> str:
    """Format previous summary compactly for prompt context."""
    if prev_summary is None:
        return "None"
    prompt_summary = {
        key: value
        for key, value in prev_summary.items()
        if key not in {"usage", "_raw_option_ledger"}
    }
    return json.dumps(prompt_summary, ensure_ascii=False, indent=2)


def _infer_round(prev_summary: Optional[SummaryDict]) -> int:
    """Infer current round from the previous summary."""
    if not prev_summary:
        return 1
    try:
        return int(prev_summary.get("round", 0)) + 1
    except (TypeError, ValueError):
        return 1


def _num_agents(option_ledger: OptionLedger) -> int:
    """Infer the number of agents represented in a ledger."""
    for info in option_ledger.values():
        return int(info["support_count"]) + int(info["oppose_count"])
    return 0


def _majority_answers(option_ledger: OptionLedger) -> List[str]:
    """Return option labels with the maximum current support count."""
    if not option_ledger:
        return []
    max_support = max(info["support_count"] for info in option_ledger.values())
    return [
        opt
        for opt, info in sorted(option_ledger.items())
        if info["support_count"] == max_support and max_support > 0
    ]


def _previous_raw_ledger(prev_summary: Optional[SummaryDict]) -> Optional[OptionLedger]:
    """Extract previous raw ledger metadata when available."""
    if not prev_summary:
        return None
    raw = prev_summary.get("_raw_option_ledger")
    if isinstance(raw, dict):
        return raw  # type: ignore[return-value]
    maybe = prev_summary.get("option_ledger")
    if isinstance(maybe, dict):
        has_counts = all(
            isinstance(value, dict) and "support_count" in value
            for value in maybe.values()
        )
        if has_counts:
            return maybe  # type: ignore[return-value]
    return None


def _zero_usage() -> Usage:
    """Return a zero token usage dict."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
