"""
Mechanism-enhanced chain debate loop.

This module keeps the baseline debate.py logic separate and implements
option-level compression plus debate-time early exit.
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
    agent_debate_response,
    agent_initial_response,
    extract_or_repair_answer,
    get_client,
)
from config import (
    AGENT_CONFIGS,
    CONSECUTIVE_STABLE_NEEDED,
    DEBATE_TEMP,
    MAX_DEBATE_ROUNDS,
    MIN_DEBATE_ROUNDS,
    NUM_OPTIONS,
    RETRY_DELAY,
    MAX_RETRIES,
    SUMMARIZER_AGENT_ID,
    SUMMARIZER_MAX_TOKENS,
    SUMMARIZER_TEMPERATURE,
    TEMPERATURE,
)

Usage = Dict[str, int]
OptionLedger = Dict[str, Dict[str, Any]]
_SUMMARY_EMBEDDING_CACHE: Dict[Tuple[str, str], List[float]] = {}


def compute_verdicts(
    answers_dict: Dict[str, Optional[str]],
    num_options: Optional[int] = None,
) -> OptionLedger:
    """
    Compute include/exclude/disputed verdicts for every option.

    Args:
        answers_dict: Mapping like {"1": "A", "2": "B"} for a round.
        num_options: Number of option labels to evaluate.

    Returns:
        Option ledger keyed by option letter.
    """
    if num_options is None:
        num_options = NUM_OPTIONS

    num_agents = len(answers_dict)
    option_labels = [chr(65 + index) for index in range(num_options)]
    support_counts = {option: 0 for option in option_labels}
    support_agents: Dict[str, List[str]] = {option: [] for option in option_labels}

    for agent_id, answer in answers_dict.items():
        if answer and answer in support_counts:
            support_counts[answer] += 1
            support_agents[answer].append(agent_id)

    all_agents = list(answers_dict.keys())
    ledger: OptionLedger = {}
    for option in option_labels:
        support_count = support_counts[option]
        if support_count == num_agents:
            verdict = "include"
        elif support_count == 0:
            verdict = "exclude"
        else:
            verdict = "disputed"

        ledger[option] = {
            "verdict": verdict,
            "support_count": support_count,
            "support_agents": support_agents[option],
            "oppose_count": num_agents - support_count,
            "oppose_agents": [
                agent_id
                for agent_id in all_agents
                if agent_id not in support_agents[option]
            ],
        }

    return ledger


def decide_compression_level(
    option_ledger: OptionLedger,
    has_prev_summary: bool,
) -> str:
    """
    Decide compression level.

    FULL: no disputed options.
    PARTIAL: disputed options exist and a previous summary exists.
    NONE: disputed options exist and this is the R1 transition.
    """
    n_disputed = sum(
        1 for value in option_ledger.values()
        if value["verdict"] == "disputed"
    )
    if n_disputed == 0:
        return "full"
    if has_prev_summary:
        return "partial"
    return "none"


def verdicts_unchanged(ledger_a: OptionLedger, ledger_b: OptionLedger) -> bool:
    """Return whether all option verdicts are unchanged across two ledgers."""
    for option in ledger_a:
        if option not in ledger_b:
            return False
        if ledger_a[option]["verdict"] != ledger_b[option]["verdict"]:
            return False
    return True


def support_counts_unchanged(ledger_a: OptionLedger, ledger_b: OptionLedger) -> bool:
    """Return whether disputed options have unchanged support counts."""
    for option in ledger_a:
        if (
            ledger_a[option]["verdict"] == "disputed"
            or ledger_b.get(option, {}).get("verdict") == "disputed"
        ):
            if ledger_a[option]["support_count"] != ledger_b.get(option, {}).get(
                "support_count",
                -1,
            ):
                return False
    return True


def call_summarizer(prompt_text: str) -> Tuple[str, Usage]:
    """
    Call Agent 1 as the summarizer.

    Args:
        prompt_text: Prompt sent to the summarizer.

    Returns:
        Response text and defensive usage dict.
    """
    client, model, _ = get_client(SUMMARIZER_AGENT_ID)
    zero_usage = _zero_usage()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            usage = _zero_usage()
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
            return response.choices[0].message.content or "", usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Summarizer failed: {exc}")
                return "", zero_usage

    return "", zero_usage


def parse_summary_json(text: str) -> Dict[str, Any]:
    """Parse JSON from strict, fenced, or noisy LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def generate_summary(
    compression_level: str,
    option_ledger: OptionLedger,
    responses: Dict[str, str],
    round_num: int,
    prev_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a summary for FULL/PARTIAL/NONE compression.

    FULL and PARTIAL use the summarizer LLM. NONE is a code-built capsule.
    """
    if compression_level == "full":
        return _gen_full_summary(option_ledger, responses, round_num, prev_summary)
    if compression_level == "partial":
        return _gen_partial_summary(option_ledger, responses, round_num, prev_summary)
    return _gen_capsule(option_ledger, responses, round_num)


def _gen_full_summary(
    option_ledger: OptionLedger,
    responses: Dict[str, str],
    round_num: int,
    prev_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate FULL summary with rule verdicts and inherited reasons."""
    del responses
    ledger_out = _build_inherited_full_ledger(option_ledger, prev_summary)
    included = [
        option for option, info in option_ledger.items()
        if info["verdict"] == "include"
    ]
    return {
        "round": round_num,
        "compression_level": "full",
        "option_ledger": ledger_out,
        "final_answer": included,
        "frozen_options": [
            option for option, info in option_ledger.items()
            if info["verdict"] != "disputed"
        ],
        "disputed_options": [],
        "delta_from_last_round": {"stance_changes": [], "new_reasons": 0},
        "usage": _zero_usage(),
        "option_ledger_raw": option_ledger,
    }


def _gen_partial_summary(
    option_ledger: OptionLedger,
    responses: Dict[str, str],
    round_num: int,
    prev_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a PARTIAL compression summary with the summarizer LLM."""
    num_agents = len(responses)
    response_text = "\n\n".join(
        f"Agent {agent_id}:\n{text}"
        for agent_id, text in responses.items()
    )
    prev_text = (
        json.dumps(prev_summary, indent=2, ensure_ascii=False)
        if prev_summary
        else "None"
    )
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
        result = _fallback_summary(option_ledger, round_num, "partial")
    result["option_ledger"] = _sanitize_llm_summary_ledger(
        option_ledger,
        result.get("option_ledger", {}),
    )
    result["compression_level"] = "partial"
    result["round"] = round_num
    result["usage"] = usage
    result["option_ledger_raw"] = option_ledger
    return result


def _gen_capsule(
    option_ledger: OptionLedger,
    responses: Dict[str, str],
    round_num: int,
) -> Dict[str, Any]:
    """Generate a zero-cost Debate Capsule for NONE mode."""
    num_agents = len(responses)
    frozen = [
        option for option, value in option_ledger.items()
        if value["verdict"] != "disputed"
    ]
    disputed = [
        option for option, value in option_ledger.items()
        if value["verdict"] == "disputed"
    ]

    lines = [f"[Debate Status - Round {round_num}]", "Option standings:"]
    for option in sorted(option_ledger):
        info = option_ledger[option]
        if info["verdict"] == "disputed":
            lines.append(
                f"  {option}: {info['support_count']}/{num_agents} include | DISPUTED"
            )
        elif info["verdict"] == "include":
            lines.append(f"  {option}: INCLUDED (all agree)")
        else:
            lines.append(f"  {option}: EXCLUDED (all agree)")
    if frozen:
        lines.append(f"Stable: {', '.join(frozen)}")
    if disputed:
        lines.append(f"Under debate: {', '.join(disputed)}")

    return {
        "round": round_num,
        "compression_level": "none",
        "option_ledger": option_ledger,
        "option_ledger_raw": option_ledger,
        "debate_capsule": "\n".join(lines),
        "delta_from_last_round": {"stance_changes": [], "new_reasons": 0},
        "usage": _zero_usage(),
    }


def _build_rule_reason_ledger(option_ledger: OptionLedger) -> Dict[str, Dict[str, Any]]:
    """Build deterministic include/exclude reasons from the raw ledger."""
    ledger_out: Dict[str, Dict[str, Any]] = {}
    for option, info in sorted(option_ledger.items()):
        verdict = info.get("verdict", "disputed")
        if verdict == "disputed":
            ledger_out[option] = {
                "verdict": "disputed",
                "support": {
                    "count": info.get("support_count", 0),
                    "agents": info.get("support_agents", []),
                    "top_reason": _option_fallback_reason(option, "support"),
                },
                "oppose": {
                    "count": info.get("oppose_count", 0),
                    "agents": info.get("oppose_agents", []),
                    "top_reason": _option_fallback_reason(option, "oppose"),
                },
            }
        else:
            ledger_out[option] = {
                "verdict": verdict,
                "reason": _option_fallback_reason(option, verdict),
            }
    return ledger_out


def _build_inherited_full_ledger(
    option_ledger: OptionLedger,
    prev_summary: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build FULL entries with deterministic verdicts and inherited reasons."""
    ledger_out: Dict[str, Dict[str, Any]] = {}
    prev_ledger = _safe_summary_ledger(prev_summary) if prev_summary else {}

    for option, info in sorted(option_ledger.items()):
        verdict = info.get("verdict", "exclude")
        if verdict == "disputed":
            ledger_out[option] = _build_rule_reason_ledger({option: info})[option]
            continue
        ledger_out[option] = {
            "verdict": verdict,
            "reason": _inherited_full_reason(
                option,
                verdict,
                prev_ledger.get(option),
            ),
        }
    return ledger_out


def _inherited_full_reason(
    option: str,
    verdict: str,
    previous_entry: Any,
) -> str:
    """Reuse the previous reason that matches the current stable verdict."""
    fallback = _option_fallback_reason(option, verdict)
    if not isinstance(previous_entry, dict):
        return fallback

    previous_verdict = previous_entry.get("verdict")
    if previous_verdict == verdict:
        return _safe_reason(previous_entry.get("reason"), fallback)

    if previous_verdict == "disputed":
        if verdict == "include":
            support = previous_entry.get("support", {})
            if isinstance(support, dict):
                return _safe_reason(support.get("top_reason"), fallback)
        if verdict == "exclude":
            oppose = previous_entry.get("oppose", {})
            if isinstance(oppose, dict):
                return _safe_reason(oppose.get("top_reason"), fallback)

    return fallback


def _sanitize_llm_summary_ledger(
    option_ledger: OptionLedger,
    candidate_ledger: Any,
) -> Dict[str, Dict[str, Any]]:
    """Keep LLM summaries schema-safe while preserving well-formed entries."""
    if not isinstance(candidate_ledger, dict):
        candidate_ledger = {}

    fallback = _build_rule_reason_ledger(option_ledger)
    safe_ledger: Dict[str, Dict[str, Any]] = {}
    for option, raw_info in sorted(option_ledger.items()):
        candidate = candidate_ledger.get(option)
        verdict = raw_info.get("verdict", "disputed")
        if verdict == "disputed":
            safe_ledger[option] = _sanitize_disputed_summary_entry(
                option,
                raw_info,
                candidate,
            )
            continue

        if isinstance(candidate, dict) and candidate.get("verdict") in (
            "include",
            "exclude",
        ):
            safe_ledger[option] = {
                "verdict": verdict,
                "reason": _safe_reason(
                    candidate.get("reason"),
                    fallback[option]["reason"],
                ),
            }
        else:
            safe_ledger[option] = fallback[option]
    return safe_ledger


def _build_embedding_stable_ledger(
    option_ledger: OptionLedger,
    responses: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Build stable include/exclude entries using embedding-selected sentences."""
    ledger_out: Dict[str, Dict[str, Any]] = {}
    for option, info in sorted(option_ledger.items()):
        verdict = info.get("verdict", "disputed")
        if verdict == "disputed":
            continue
        ledger_out[option] = {
            "verdict": verdict,
            "reason": _representative_sentence_reason(option, verdict, responses),
        }
    return ledger_out


def _sanitize_disputed_summary_entry(
    option: str,
    raw_info: Dict[str, Any],
    candidate: Any,
) -> Dict[str, Any]:
    """Return a schema-safe disputed summary entry."""
    fallback = _build_rule_reason_ledger({option: raw_info})[option]
    if not isinstance(candidate, dict) or candidate.get("verdict") != "disputed":
        return fallback

    support = candidate.get("support", {})
    oppose = candidate.get("oppose", {})
    if not isinstance(support, dict):
        support = {}
    if not isinstance(oppose, dict):
        oppose = {}

    return {
        "verdict": "disputed",
        "support": {
            "count": _safe_int(support.get("count"), raw_info.get("support_count", 0)),
            "agents": _safe_agent_list(
                support.get("agents"),
                raw_info.get("support_agents", []),
            ),
            "top_reason": _safe_reason(
                support.get("top_reason"),
                fallback["support"]["top_reason"],
            ),
        },
        "oppose": {
            "count": _safe_int(oppose.get("count"), raw_info.get("oppose_count", 0)),
            "agents": _safe_agent_list(
                oppose.get("agents"),
                raw_info.get("oppose_agents", []),
            ),
            "top_reason": _safe_reason(
                oppose.get("top_reason"),
                fallback["oppose"]["top_reason"],
            ),
        },
    }


def _safe_int(value: Any, fallback: Any = 0) -> int:
    """Parse an integer field without allowing malformed summaries to crash."""
    try:
        return int(value)
    except Exception:
        try:
            return int(fallback)
        except Exception:
            return 0


def _safe_agent_list(value: Any, fallback: Any) -> List[str]:
    """Normalize agent id lists from LLM JSON."""
    source = value if isinstance(value, list) else fallback
    if not isinstance(source, list):
        return []
    return [str(item) for item in source]


def _safe_reason(value: Any, fallback: str) -> str:
    """Normalize a reason field from LLM JSON."""
    if isinstance(value, str) and value.strip():
        return _truncate_reason(value)
    return fallback


def _representative_sentence_reason(
    option: str,
    verdict: str,
    responses: Dict[str, str],
) -> str:
    """Return the best embedding-matched sentence for a stable option."""
    fallback = _option_fallback_reason(option, verdict)
    candidates = _summary_sentence_candidates(option, responses)
    if not candidates:
        return fallback

    try:
        query = f"Reason to {verdict} option {option}."
        query_embedding = _summary_embedding(query)
        scored = [
            (_cosine_similarity(query_embedding, _summary_embedding(candidate)), candidate)
            for candidate in candidates
        ]
    except Exception:
        return fallback

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return fallback
    return _truncate_reason(scored[0][1])


def _summary_sentence_candidates(
    option: str,
    responses: Dict[str, str],
    max_candidates: int = 24,
) -> List[str]:
    """Split agent responses into compact candidate reasoning sentences."""
    option_patterns = [
        rf"\boption\s+{re.escape(option)}\b",
        rf"\b{re.escape(option)}\)",
        rf"\({re.escape(option)}\)",
        rf"\b{re.escape(option)}\b",
    ]
    option_regex = re.compile("|".join(option_patterns), re.IGNORECASE)
    all_sentences: List[str] = []
    option_sentences: List[str] = []

    for agent_id, text in responses.items():
        del agent_id
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            continue
        pieces = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        for piece in pieces:
            sentence = piece.strip()
            if len(sentence) < 20:
                continue
            sentence = _truncate_reason(sentence, max_words=36)
            all_sentences.append(sentence)
            if option_regex.search(sentence):
                option_sentences.append(sentence)

    candidates = option_sentences or all_sentences
    return candidates[:max_candidates]


def _option_fallback_reason(option: str, verdict: str) -> str:
    """Return a deterministic reason template for an option verdict."""
    if verdict == "include":
        return f"All agents agreed to include option {option}."
    if verdict == "exclude":
        return f"All agents agreed to exclude option {option}."
    if verdict == "support":
        return f"Some agents supported option {option}."
    if verdict == "oppose":
        return f"Some agents opposed option {option}."
    return f"Agents disagreed about option {option}."


def _summary_embedding(text: str) -> List[float]:
    """Return an Ollama embedding vector with an in-memory cache."""
    model = getattr(_config, "EVIDENCE_EMBEDDING_MODEL", "nomic-embed-text")
    cache_key = (model, text)
    if cache_key in _SUMMARY_EMBEDDING_CACHE:
        return _SUMMARY_EMBEDDING_CACHE[cache_key]

    base_url = getattr(
        _config,
        "EVIDENCE_EMBEDDING_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/")
    response = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings")
    if embeddings and isinstance(embeddings, list):
        vector = embeddings[0]
    else:
        vector = payload.get("embedding")
    if not isinstance(vector, list):
        raise ValueError("Ollama embedding response did not include a vector")

    result = [float(value) for value in vector]
    _SUMMARY_EMBEDDING_CACHE[cache_key] = result
    return result


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity for two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _truncate_reason(text: str, max_words: int = 30) -> str:
    """Keep representative reasons compact for downstream prompts."""
    words = re.sub(r"\s+", " ", str(text or "")).strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def _fallback_summary(
    option_ledger: OptionLedger,
    round_num: int,
    level: str,
) -> Dict[str, Any]:
    """Build a minimal deterministic summary when JSON parsing fails."""
    ledger_out: Dict[str, Dict[str, Any]] = {}
    for option, info in option_ledger.items():
        if info["verdict"] == "disputed":
            ledger_out[option] = {
                "verdict": "disputed",
                "support": {
                    "count": info["support_count"],
                    "agents": info["support_agents"],
                    "top_reason": "(parse failed)",
                },
                "oppose": {
                    "count": info["oppose_count"],
                    "agents": info["oppose_agents"],
                    "top_reason": "(parse failed)",
                },
            }
        else:
            ledger_out[option] = {
                "verdict": info["verdict"],
                "reason": "(parse failed)",
            }

    included = [
        option for option, value in option_ledger.items()
        if value["verdict"] == "include"
    ]
    return {
        "round": round_num,
        "compression_level": level,
        "option_ledger": ledger_out,
        "frozen_options": [
            option for option, value in option_ledger.items()
            if value["verdict"] != "disputed"
        ],
        "disputed_options": [
            option for option, value in option_ledger.items()
            if value["verdict"] == "disputed"
        ],
        "current_answer": included,
        "delta_from_last_round": {"stance_changes": [], "new_reasons": 0},
    }


def format_summary_for_prompt(summary: Optional[Dict[str, Any]]) -> str:
    """Format a summary as replacement context for the next agent prompt."""
    if not summary:
        return ""

    level = summary.get("compression_level", "none")
    if level == "none":
        return summary.get("debate_capsule", "")

    lines = [f"[Round {summary.get('round', '?')} Summary]"]
    ledger = _safe_summary_ledger(summary)

    stable = [
        (option, info)
        for option, info in ledger.items()
        if info.get("verdict") in ("include", "exclude")
    ]
    if stable:
        lines.append("Stable options:")
        for option, info in sorted(stable):
            lines.append(
                f"  {option}: {info.get('verdict', '?').upper()} - "
                f"{info.get('reason', 'N/A')}"
            )

    disputed = [
        (option, info)
        for option, info in ledger.items()
        if info.get("verdict") == "disputed"
    ]
    if disputed:
        lines.append("Disputed options (focus here):")
        for option, info in sorted(disputed):
            support = info.get("support", {})
            oppose = info.get("oppose", {})
            if not isinstance(support, dict):
                support = {}
            if not isinstance(oppose, dict):
                oppose = {}
            lines.append(
                f"  {option}: DISPUTED "
                f"({support.get('count', 0)} vs {oppose.get('count', 0)})"
            )
            lines.append(
                f"    Support (agents {support.get('agents', [])}): "
                f"{support.get('top_reason', 'N/A')}"
            )
            lines.append(
                f"    Oppose (agents {oppose.get('agents', [])}): "
                f"{oppose.get('top_reason', 'N/A')}"
            )

    delta = summary.get("delta_from_last_round", {})
    changes = delta.get("stance_changes", [])
    if changes:
        lines.append("Changes:")
        for change in changes[:5]:
            lines.append(
                f"  Agent {change.get('agent', '?')} on "
                f"{change.get('option', '?')}: "
                f"{change.get('from', '?')} -> {change.get('to', '?')}"
            )

    return "\n".join(lines)


def _attach_own_previous_context(
    context: str,
    agent_id: int,
    round_num: int,
    history: Dict[int, Dict[int, str]],
    answers: Dict[int, Dict[int, Optional[str]]],
) -> str:
    """
    Add own previous stance for commitment/evidence prompt ablations.

    With the old prompt style, return the context unchanged so the mechanism's
    compression behavior stays identical to the prior implementation.
    """
    if getattr(_config, "DEBATE_PROMPT_STYLE", "old") == "old":
        return context

    own_previous_response = history.get(round_num - 1, {}).get(agent_id, "")
    own_previous_answer = answers.get(round_num - 1, {}).get(agent_id)

    return (
        f"[Your Previous Answer]\n{own_previous_answer or 'unknown'}\n\n"
        f"[Your Previous Reasoning]\n{own_previous_response or 'N/A'}\n\n"
        f"[Current Debate Context]\n{context}"
    )


def _format_decompress_context(
    prev_summary: Dict[str, Any],
    predecessor_id: int,
    predecessor_response: str,
) -> str:
    """
    Build a one-round decompressed context from a capsule plus raw predecessor.

    This is used when verdicts are stable but the majority signal is not strong
    enough to safely exit. It avoids another summarizer call and temporarily
    restores raw local debate context.
    """
    capsule = _build_decompress_capsule(prev_summary)
    return (
        f"{capsule}\n\n"
        f"[Previous Agent {predecessor_id} Raw Response]\n"
        f"{predecessor_response}"
    )


def _build_decompress_capsule(prev_summary: Dict[str, Any]) -> str:
    """Build a deterministic Debate Capsule from the previous raw ledger."""
    raw_ledger = _safe_raw_ledger(prev_summary)
    if not raw_ledger:
        return format_summary_for_prompt(prev_summary)

    lines = [
        f"[Debate Capsule - Round {prev_summary.get('round', '?')}]",
        "Option standings:",
    ]
    frozen = []
    disputed = []
    for option, info in sorted(raw_ledger.items()):
        verdict = info.get("verdict", "unknown")
        if verdict == "disputed":
            disputed.append(option)
            support_count = info.get("support_count", 0)
            oppose_count = info.get("oppose_count", 0)
            total = int(support_count or 0) + int(oppose_count or 0)
            lines.append(
                f"  {option}: {support_count}/{total} include | DISPUTED"
            )
        else:
            frozen.append(option)
            lines.append(f"  {option}: {verdict.upper()}")

    lines.append(f"Stable options: {', '.join(frozen) if frozen else 'none'}")
    lines.append(
        f"Options under debate: {', '.join(disputed) if disputed else 'none'}"
    )
    return "\n".join(lines)


def run_debate_with_mechanism(
    question: str,
    choices: str,
    num_options: Optional[int] = None,
    verbose: bool = False,
    enable_early_exit: bool = True,
    enable_verdicts_stable_exit: bool = False,
    enable_deadlock_exit: bool = True,
) -> Dict[str, Any]:
    """
    Run chain debate with option-level compression and debate-time early exit.

    R0 never exits. When enabled, early exit is checked only after debate
    rounds, starting at R2. The current accuracy-first setting keeps
    verdicts_stable disabled by default; the implementation remains available
    behind enable_verdicts_stable_exit for ablation comparison.
    """
    if num_options is None:
        num_options = NUM_OPTIONS

    agent_ids = [cfg["agent_id"] for cfg in AGENT_CONFIGS]
    history: Dict[int, Dict[int, str]] = {0: {}}
    usages: Dict[int, Dict[int, Usage]] = {0: {}}
    answers: Dict[int, Dict[int, Optional[str]]] = {0: {}}
    verdict_history: List[OptionLedger] = []
    mechanism_log: List[Dict[str, Any]] = []
    answer_repair_log: Dict[str, List[Dict[str, Any]]] = {}
    summarizer_total_usage = _zero_usage()
    evidence_verification_total_usage = _zero_usage()

    if verbose:
        print("  [MECH] Round 0 (independent, no early exit)...")

    for agent_id in agent_ids:
        try:
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
        if answer_repair_log.get("0"):
            repaired = [
                (
                    f"A{item['agent']}->{item['answer']} "
                    f"(raw_len={item['original_response_length']}, "
                    f"empty={item['original_response_empty']})"
                )
                for item in answer_repair_log["0"]
            ]
            print(f"    answer repairs: {', '.join(repaired)}")

    r0_ledger = compute_verdicts(
        {str(agent_id): answers[0][agent_id] for agent_id in agent_ids},
        num_options=num_options,
    )
    verdict_history.append(r0_ledger)
    prev_summary: Optional[Dict[str, Any]] = None
    current_ledger = r0_ledger
    decompress_next_round = False
    decompress_used = False

    for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
        decompress_this_round = decompress_next_round
        decompress_next_round = False
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
                raw_ledger = _safe_raw_ledger(prev_summary)
                if decompress_this_round:
                    enhanced_context = _format_decompress_context(
                        prev_summary,
                        predecessor_id,
                        predecessor_response,
                    )
                else:
                    enhanced_context = format_summary_for_prompt(prev_summary)
                    if prev_summary.get("compression_level") == "none":
                        enhanced_context = (
                            f"{enhanced_context}\n\n"
                            f"[Previous Agent {predecessor_id} Response]\n"
                            f"{predecessor_response}"
                        )
                disputed_options = [
                    option for option, info in raw_ledger.items()
                    if isinstance(info, dict) and info.get("verdict") == "disputed"
                ]
                if disputed_options:
                    enhanced_context = (
                        f"{enhanced_context}\n\n"
                        "Focus on disputed options: "
                        + ", ".join(disputed_options)
                        + ". Do not re-discuss stable options."
                    )
            else:
                enhanced_context = predecessor_response

            enhanced_context = _attach_own_previous_context(
                enhanced_context,
                agent_id,
                round_num,
                history,
                answers,
            )

            try:
                text, usage = agent_debate_response(
                    question,
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
            if answer_repair_log.get(str(round_num)):
                repaired = [
                    (
                        f"A{item['agent']}->{item['answer']} "
                        f"(raw_len={item['original_response_length']}, "
                        f"empty={item['original_response_empty']})"
                    )
                    for item in answer_repair_log[str(round_num)]
                ]
                print(f"    answer repairs: {', '.join(repaired)}")

        current_ledger = compute_verdicts(
            {str(agent_id): answers[round_num][agent_id] for agent_id in agent_ids},
            num_options=num_options,
        )
        compression_level = decide_compression_level(
            current_ledger,
            prev_summary is not None,
        )

        n_disputed = _count_disputed(current_ledger)
        round_log = {
            "round": round_num,
            "compression_level": compression_level,
            "n_disputed": n_disputed,
            "disputed": [
                option for option, info in current_ledger.items()
                if info["verdict"] == "disputed"
            ],
            "decompress_context": decompress_this_round,
            "answer_repairs": answer_repair_log.get(str(round_num), []),
        }

        verdict_history.append(current_ledger)

        if (
            enable_early_exit
            and round_num >= MIN_DEBATE_ROUNDS
            and _all_stable_answer_unchanged(verdict_history)
        ):
            round_log["exit_check"] = "all_stable"
            mechanism_log.append(round_log)
            if verbose:
                verdict_dist = Counter(
                    info["verdict"] for info in current_ledger.values()
                )
                print(
                    f"    verdict={dict(verdict_dist)}, "
                    f"compression={compression_level}, "
                    f"n_disputed={n_disputed}, "
                    "exit=all_stable"
                )
                print(
                    "    -> Early exit before summary: "
                    f"all_stable, answer={_get_final_answer(answers, round_num, current_ledger)}"
                )
            final_answer = _get_final_answer(answers, round_num, current_ledger)
            return _build_mechanism_result(
                history,
                usages,
                answers,
                agent_ids,
                final_answer,
                round_num,
                "all_stable",
                mechanism_log,
                summarizer_total_usage,
                evidence_verification_usage=evidence_verification_total_usage,
                answer_repair_log=answer_repair_log,
            )

        summary = generate_summary(
            compression_level,
            current_ledger,
            {str(agent_id): history[round_num][agent_id] for agent_id in agent_ids},
            round_num,
            prev_summary,
        )
        summary_usage = summary.get("usage", {})
        _add_usage(summarizer_total_usage, summary_usage)
        prev_summary = summary

        exit_reason = _check_debate_early_exit(
            round_num,
            current_ledger,
            verdict_history,
            summary,
            answers,
            enable_early_exit=enable_early_exit,
            enable_verdicts_stable_exit=enable_verdicts_stable_exit,
            enable_deadlock_exit=enable_deadlock_exit,
            enable_all_stable_exit=False,
            allow_decompress=not decompress_used and round_num < MAX_DEBATE_ROUNDS,
        )
        round_log["exit_check"] = exit_reason or "continue"
        mechanism_log.append(round_log)

        if verbose:
            verdict_dist = Counter(info["verdict"] for info in current_ledger.values())
            print(
                f"    verdict={dict(verdict_dist)}, "
                f"compression={compression_level}, "
                f"n_disputed={n_disputed}, "
                f"exit={round_log['exit_check']}"
            )

        if exit_reason == "verdicts_stable_decompress":
            decompress_next_round = True
            decompress_used = True
            round_log["decompress_next_round"] = True
            if verbose:
                print(
                    "    -> Unsafe verdicts_stable: "
                    "decompress next round, no early exit"
                )
            continue

        if exit_reason == "verdicts_stable_verify":
            majority_answer, majority_margin = _majority_answer_and_margin(
                answers.get(round_num, {})
            )
            verification_answer, verification_text, verification_usage = (
                _run_evidence_verification(
                    question,
                    choices,
                    current_ledger,
                    answers,
                    history,
                    round_num,
                    majority_answer,
                )
            )
            _add_usage(evidence_verification_total_usage, verification_usage)
            verified = (
                majority_answer is not None
                and verification_answer == majority_answer
            )
            round_log["evidence_verification"] = {
                "triggered": True,
                "majority_answer": majority_answer,
                "majority_margin": majority_margin,
                "verifier_answer": verification_answer,
                "verified": verified,
                "response": verification_text,
                "usage": verification_usage,
            }
            if verified:
                round_log["exit_check"] = "verified_verdicts_stable"
                final_answer = verification_answer
                if verbose:
                    print(
                        "    -> Evidence verification passed: "
                        f"answer={final_answer}"
                    )
                return _build_mechanism_result(
                    history,
                    usages,
                    answers,
                    agent_ids,
                    final_answer,
                    round_num,
                    "verified_verdicts_stable",
                    mechanism_log,
                    summarizer_total_usage,
                    evidence_verification_usage=evidence_verification_total_usage,
                    answer_repair_log=answer_repair_log,
                )

            round_log["exit_check"] = "verdicts_stable_verification_failed"
            if verbose:
                print(
                    "    -> Evidence verification failed: "
                    f"majority={majority_answer}, verifier={verification_answer}; "
                    "continue"
                )
            continue

        if exit_reason:
            final_answer = _get_final_answer(answers, round_num, current_ledger)
            if verbose:
                print(f"    -> Early exit: {exit_reason}, answer={final_answer}")
            return _build_mechanism_result(
                history,
                usages,
                answers,
                agent_ids,
                final_answer,
                round_num,
                exit_reason,
                mechanism_log,
                summarizer_total_usage,
                evidence_verification_usage=evidence_verification_total_usage,
                answer_repair_log=answer_repair_log,
            )

    fallback_answer = _get_final_answer(answers, MAX_DEBATE_ROUNDS, current_ledger)
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
    return _build_mechanism_result(
        history,
        usages,
        answers,
        agent_ids,
        final_answer,
        MAX_DEBATE_ROUNDS,
        "max_rounds",
        mechanism_log,
        summarizer_total_usage,
        review_usage,
        evidence_verification_usage=evidence_verification_total_usage,
        answer_repair_log=answer_repair_log,
    )


def _run_final_evidence_review(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
) -> Tuple[Optional[str], str, Usage]:
    """
    Use Agent 1 as a final evidence judge for unresolved max-round cases.

    This is only called after debate reaches MAX_DEBATE_ROUNDS without an
    earlier exit. If the judge fails to provide a valid answer, the caller falls
    back to the deterministic latest-round majority vote.
    """
    client, model, name = get_client(SUMMARIZER_AGENT_ID)
    disputed = [
        option for option, info in option_ledger.items()
        if info.get("verdict") == "disputed"
    ]
    standings = _format_option_standings(option_ledger)
    recent_votes = _format_recent_votes(answers, current_round)
    recent_responses = _format_recent_responses(history, current_round)

    prompt = f"""You are Agent {SUMMARIZER_AGENT_ID} ({name}), acting as a final evidence judge.

The debate reached the maximum number of rounds without stable consensus.

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
2. Focus on disputed options and explain why the strongest option is better than the closest alternative.
3. Do not choose by vote count alone.

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
                SUMMARIZER_AGENT_ID,
            )
            if repair_usage.get("total_tokens", 0) > 0:
                _add_usage(usage, repair_usage)
                text = f"{text}\n\n[Answer repair]\n{repair_text}"
            return answer, text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Final evidence review failed: {exc}")

    return None, "", _zero_usage()


def _run_evidence_verification(
    question: str,
    choices: str,
    option_ledger: OptionLedger,
    answers: Dict[int, Dict[int, Optional[str]]],
    history: Dict[int, Dict[int, str]],
    current_round: int,
    majority_answer: Optional[str],
) -> Tuple[Optional[str], str, Usage]:
    """
    Independently verify a verdict-stable case before allowing early exit.

    The verifier does not see the current majority answer, vote counts, or the
    closest competitor. The caller compares this independent answer with the
    current majority and exits only when they match.
    """
    if majority_answer is None:
        return None, "", _zero_usage()

    client, model, name = get_client(SUMMARIZER_AGENT_ID)
    recent_responses = _format_recent_responses(
        history,
        current_round,
        max_chars_per_response=800,
    )

    # Previous majority-anchored verifier kept for comparison:
    # challenger = _runner_up_answer(answers.get(current_round, {}), majority_answer)
    # standings = _format_option_standings(option_ledger)
    # recent_votes = _format_recent_votes(answers, current_round)
    # prompt included:
    #   Current majority answer to verify: {majority_answer}
    #   Closest competing answer: {challenger or 'unknown'}
    #   Option standings: {standings}
    #   Recent vote history: {recent_votes}

    prompt = f"""You are Agent {SUMMARIZER_AGENT_ID} ({name}), acting as an evidence verifier.

The debate verdicts have been stable for multiple rounds, but stable agreement can still be wrong. You must answer independently.

Do not infer the answer from consensus, vote counts, or the number of agents supporting an option. Treat the recent reasoning below only as notes that may contain mistakes.

Question:
{question}

Options:
{choices}

Recent agent reasoning:
{recent_responses}

Task:
1. Re-read the passage/question evidence yourself.
2. Check each option against the evidence, including options that agents may have dismissed.
3. Choose the option best supported by textual evidence, not by consensus.

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
                SUMMARIZER_AGENT_ID,
            )
            if repair_usage.get("total_tokens", 0) > 0:
                _add_usage(usage, repair_usage)
                text = f"{text}\n\n[Answer repair]\n{repair_text}"
            return answer, text, usage
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Evidence verification failed: {exc}")

    return None, "", _zero_usage()


def _runner_up_answer(
    answers_round: Dict[int, Optional[str]],
    majority_answer: Optional[str],
) -> Optional[str]:
    """Return the runner-up answer in a round, excluding the majority answer."""
    valid_answers = [answer for answer in answers_round.values() if answer]
    if not valid_answers:
        return None

    counts = Counter(valid_answers).most_common()
    for answer, _ in counts:
        if answer != majority_answer:
            return answer
    return None


def _format_option_standings(option_ledger: OptionLedger) -> str:
    """Format deterministic option standings for the final judge."""
    lines = []
    for option, info in sorted(option_ledger.items()):
        lines.append(
            f"{option}: {info.get('verdict')} | "
            f"support={info.get('support_count', 0)} "
            f"agents={info.get('support_agents', [])} | "
            f"oppose={info.get('oppose_count', 0)} "
            f"agents={info.get('oppose_agents', [])}"
        )
    return "\n".join(lines)


def _format_recent_votes(
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
) -> str:
    """Format the latest two vote rounds for the final judge."""
    lines = []
    for round_id in (current_round - 1, current_round):
        if round_id not in answers:
            continue
        votes = ", ".join(
            f"Agent {agent_id}: {answer or 'None'}"
            for agent_id, answer in sorted(answers[round_id].items())
        )
        majority, margin = _majority_answer_and_margin(answers[round_id])
        lines.append(
            f"Round {round_id}: {votes} | majority={majority}, margin={margin}"
        )
    return "\n".join(lines) if lines else "No recent votes available."


def _format_recent_responses(
    history: Dict[int, Dict[int, str]],
    current_round: int,
    max_chars_per_response: int = 1200,
) -> str:
    """Format recent raw responses with a cap to keep judge prompts bounded."""
    lines = []
    for round_id in (current_round - 1, current_round):
        if round_id not in history:
            continue
        lines.append(f"[Round {round_id}]")
        for agent_id, response in sorted(history[round_id].items()):
            snippet = response[:max_chars_per_response]
            if len(response) > max_chars_per_response:
                snippet += "\n[truncated]"
            lines.append(f"Agent {agent_id}:\n{snippet}")
    return "\n\n".join(lines) if lines else "No recent responses available."


def _check_debate_early_exit(
    round_num: int,
    current_ledger: OptionLedger,
    verdict_history: List[OptionLedger],
    summary: Dict[str, Any],
    answers: Dict[int, Dict[int, Optional[str]]],
    enable_early_exit: bool = True,
    enable_verdicts_stable_exit: bool = True,
    enable_deadlock_exit: bool = True,
    enable_all_stable_exit: bool = True,
    allow_decompress: bool = True,
) -> Optional[str]:
    """
    Apply the three debate-time early-exit layers in strict order.

    Returns:
        all_stable, verdicts_stable_verify, verdicts_stable_decompress,
        deadlock, or None.
    """
    if not enable_early_exit:
        return None

    if round_num < MIN_DEBATE_ROUNDS:
        return None

    if enable_all_stable_exit and _all_stable_answer_unchanged(verdict_history):
        return "all_stable"

    # Fallback variant kept for future comparison, but disabled for the current
    # QuALITY mechanism setting requested by the experiment design.
    # if _all_stable_majority_fallback(verdict_history):
    #     return "all_stable_fallback"

    if (
        enable_verdicts_stable_exit
        and len(verdict_history) >= CONSECUTIVE_STABLE_NEEDED + 1
    ):
        verdicts_same = True
        for index in range(1, CONSECUTIVE_STABLE_NEEDED + 1):
            if not verdicts_unchanged(
                verdict_history[-index],
                verdict_history[-(index + 1)],
            ):
                verdicts_same = False
                break
        if verdicts_same and _safe_verdicts_stable_by_margin(
            answers,
            round_num,
            min_margin=2,
        ):
            # Previous safe direct-exit behavior kept for comparison:
            # return "verdicts_stable"
            return "verdicts_stable_verify"
        if verdicts_same and allow_decompress:
            return "verdicts_stable_decompress"
        # Previous direct-exit behavior kept for comparison:
        # if verdicts_same:
        #     return "verdicts_stable"

    if enable_deadlock_exit and len(verdict_history) >= CONSECUTIVE_STABLE_NEEDED + 1:
        counts_same = True
        for index in range(1, CONSECUTIVE_STABLE_NEEDED + 1):
            if not support_counts_unchanged(
                verdict_history[-index],
                verdict_history[-(index + 1)],
            ):
                counts_same = False
                break
        no_new_reason = (
            summary.get("delta_from_last_round", {}).get("new_reasons", 0) == 0
        )
        if counts_same and no_new_reason:
            return "deadlock"

    return None


def _all_stable_answer_unchanged(verdict_history: List[OptionLedger]) -> bool:
    """
    Return whether the latest two rounds are all-stable with the same answer.

    R0 may be all-stable, but early-exit checking starts at R2, so this still
    exits only during debate.
    """
    if len(verdict_history) < 2:
        return False

    current_answer = _included_answer_from_ledger(verdict_history[-1])
    previous_answer = _included_answer_from_ledger(verdict_history[-2])
    return (
        current_answer is not None
        and previous_answer is not None
        and current_answer == previous_answer
    )


def _all_stable_majority_fallback(verdict_history: List[OptionLedger]) -> bool:
    """
    Return whether a previously all-stable answer still holds a majority.

    This prevents one confirmed consensus round from drifting into long R4-R6
    summary chains when the next round only weakly reopens the dispute.
    """
    if len(verdict_history) < 2:
        return False

    previous_answer = _included_answer_from_ledger(verdict_history[-2])
    if previous_answer is None:
        return False

    current_ledger = verdict_history[-1]
    if _included_answer_from_ledger(current_ledger) is not None:
        return False

    previous_info = current_ledger.get(previous_answer)
    if not previous_info:
        return False

    num_agents = (
        int(previous_info.get("support_count", 0))
        + int(previous_info.get("oppose_count", 0))
    )
    majority_threshold = (num_agents // 2) + 1
    return int(previous_info.get("support_count", 0)) >= majority_threshold


def _safe_verdicts_stable_by_margin(
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
    min_margin: int = 2,
) -> bool:
    """
    Require the majority answer to stay unchanged with a sufficient margin.

    This makes verdicts_stable a stronger exit signal for larger agent groups.
    The older behavior, which exited on unchanged verdict labels alone, is kept
    commented in _check_debate_early_exit for ablation comparison.
    """
    if current_round < 1:
        return False

    current_majority, current_margin = _majority_answer_and_margin(
        answers.get(current_round, {})
    )
    previous_majority, previous_margin = _majority_answer_and_margin(
        answers.get(current_round - 1, {})
    )

    return (
        current_majority is not None
        and current_majority == previous_majority
        and current_margin >= min_margin
        and previous_margin >= min_margin
    )


def _majority_answer_and_margin(
    answers_round: Dict[int, Optional[str]],
) -> Tuple[Optional[str], int]:
    """Return the majority answer and its margin over the runner-up."""
    valid_answers = [answer for answer in answers_round.values() if answer]
    if not valid_answers:
        return None, 0

    counts = Counter(valid_answers).most_common()
    majority_answer, majority_count = counts[0]
    runner_up_count = counts[1][1] if len(counts) > 1 else 0
    return majority_answer, majority_count - runner_up_count


def _included_answer_from_ledger(option_ledger: OptionLedger) -> Optional[str]:
    """Return included option when the ledger has no disputed options."""
    if _count_disputed(option_ledger) != 0:
        return None

    included = [
        option for option, info in option_ledger.items()
        if info["verdict"] == "include"
    ]
    if len(included) == 1:
        return included[0]
    return None


def _get_final_answer(
    answers: Dict[int, Dict[int, Optional[str]]],
    current_round: int,
    option_ledger: OptionLedger,
) -> Optional[str]:
    """Return included option if stable, otherwise recent two-round vote."""
    included = [
        option for option, info in option_ledger.items()
        if info["verdict"] == "include"
    ]
    if included:
        return included[0]

    recent_answers: List[str] = []
    for round_id in (current_round - 1, current_round):
        if round_id in answers:
            recent_answers.extend(
                answer for answer in answers[round_id].values() if answer
            )
    if recent_answers:
        return Counter(recent_answers).most_common(1)[0][0]

    # Previous latest-round majority behavior kept for comparison:
    # valid_answers = [
    #     answer for answer in answers.get(current_round, {}).values() if answer
    # ]
    # if valid_answers:
    #     return Counter(valid_answers).most_common(1)[0][0]
    return None


def _build_mechanism_result(
    history: Dict[int, Dict[int, str]],
    usages: Dict[int, Dict[int, Usage]],
    answers: Dict[int, Dict[int, Optional[str]]],
    agent_ids: List[int],
    final_answer: Optional[str],
    actual_rounds: int,
    exit_reason: str,
    mechanism_log: List[Dict[str, Any]],
    summarizer_usage: Usage,
    final_review_usage: Optional[Usage] = None,
    evidence_verification_usage: Optional[Usage] = None,
    answer_repair_log: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Build a baseline-compatible result with mechanism metadata."""
    if final_review_usage is None:
        final_review_usage = _zero_usage()
    if evidence_verification_usage is None:
        evidence_verification_usage = _zero_usage()
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
    _add_usage(total_with_summarizer, final_review_usage)
    _add_usage(total_with_summarizer, evidence_verification_usage)

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
            "final_review": final_review_usage,
            "evidence_verification": evidence_verification_usage,
            "total_with_summarizer": total_with_summarizer,
        },
        "mechanism": {
            "actual_rounds": actual_rounds,
            "early_exit": exit_reason != "max_rounds",
            "exit_reason": exit_reason,
            "mechanism_log": mechanism_log,
            "answer_repair_log": answer_repair_log,
        },
    }


def _build_answer_repair_entry(
    round_num: int,
    agent_id: int,
    answer: Optional[str],
    original_response: str,
    repair_response: str,
    repair_usage: Usage,
    max_excerpt_chars: int = 2000,
) -> Dict[str, Any]:
    """Build a compact raw-response diagnostic entry for answer repair."""
    raw = original_response or ""
    excerpt = raw[:max_excerpt_chars]
    if len(raw) > max_excerpt_chars:
        excerpt += "\n[truncated]"

    return {
        "round": round_num,
        "agent": agent_id,
        "answer": answer,
        "original_response_length": len(raw),
        "original_response_empty": len(raw.strip()) == 0,
        "original_response_is_error": raw.startswith("[ERROR]"),
        "original_response_excerpt": excerpt,
        "repair_response": repair_response,
        "repair_usage": repair_usage,
    }


def _count_disputed(option_ledger: OptionLedger) -> int:
    """Count disputed options in a ledger."""
    return sum(1 for info in option_ledger.values() if info["verdict"] == "disputed")


def _safe_summary_ledger(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return a display-safe summary ledger.

    The summarizer occasionally emits a malformed option_ledger. In that case,
    fall back to the raw deterministic ledger so the debate can continue.
    """
    if not isinstance(summary, dict):
        return {}

    ledger = summary.get("option_ledger", {})
    raw_ledger = _safe_raw_ledger(summary)
    if raw_ledger:
        return _sanitize_llm_summary_ledger(raw_ledger, ledger)

    if isinstance(ledger, dict):
        safe_without_raw: Dict[str, Dict[str, Any]] = {}
        for option, info in ledger.items():
            if not isinstance(info, dict):
                continue
            verdict = info.get("verdict")
            if verdict in ("include", "exclude"):
                safe_without_raw[option] = {
                    "verdict": verdict,
                    "reason": _safe_reason(
                        info.get("reason"),
                        _option_fallback_reason(option, verdict),
                    ),
                }
            elif verdict == "disputed":
                safe_without_raw[option] = _sanitize_disputed_summary_entry(
                    option,
                    {
                        "verdict": "disputed",
                        "support_count": 0,
                        "support_agents": [],
                        "oppose_count": 0,
                        "oppose_agents": [],
                    },
                    info,
                )
        if safe_without_raw:
            return safe_without_raw

    fallback: Dict[str, Dict[str, Any]] = {}
    for option, info in raw_ledger.items():
        verdict = info.get("verdict", "unknown")
        if verdict == "disputed":
            fallback[option] = {
                "verdict": "disputed",
                "support": {
                    "count": info.get("support_count", 0),
                    "agents": info.get("support_agents", []),
                    "top_reason": "(malformed summary)",
                },
                "oppose": {
                    "count": info.get("oppose_count", 0),
                    "agents": info.get("oppose_agents", []),
                    "top_reason": "(malformed summary)",
                },
            }
        else:
            fallback[option] = {
                "verdict": verdict,
                "reason": "(malformed summary)",
            }
    return fallback


def _safe_raw_ledger(summary: Dict[str, Any]) -> OptionLedger:
    """Return option_ledger_raw if it is usable, otherwise an empty ledger."""
    raw_ledger = summary.get("option_ledger_raw", {})
    if isinstance(raw_ledger, dict) and all(
        isinstance(value, dict) for value in raw_ledger.values()
    ):
        return raw_ledger
    return {}


def _zero_usage() -> Usage:
    """Return a zero token usage dict."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_usage(target: Usage, source: Dict[str, Any]) -> None:
    """Add usage fields from source into target in place."""
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] += int(source.get(key, 0) or 0)
