"""
Agent implementation for multi-agent debate.
Each agent uses a configured Ollama model through the OpenAI-compatible API.
Handles API calls, token tracking, answer extraction, and retry logic.
"""

import re
import time
import unicodedata
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

import config as _config
from config import AGENT_CONFIGS, MAX_RETRIES, OLLAMA_REQUEST_TIMEOUT, RETRY_DELAY


Usage = Dict[str, int]
ENABLE_LLM_ANSWER_REPAIR = False


def get_client(agent_id: int) -> Tuple[OpenAI, str, str]:
    """
    Create an OpenAI client for a specific agent using Ollama.

    Args:
        agent_id: Agent ID, starting from 1.

    Returns:
        Tuple of client, model name, and display name.
    """
    if agent_id < 1 or agent_id > len(AGENT_CONFIGS):
        raise ValueError(
            f"Invalid agent_id: {agent_id}. Must be 1-{len(AGENT_CONFIGS)}"
        )

    cfg = AGENT_CONFIGS[agent_id - 1]
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=OLLAMA_REQUEST_TIMEOUT,
    )
    return client, cfg["model"], cfg["name"]


def extract_usage(response: Any) -> Usage:
    """Extract token usage from a response with defensive handling."""
    usage = _zero_usage()

    if response.usage is None:
        print("Warning: response.usage is None. Recording 0 tokens.")
        return usage

    try:
        usage["prompt_tokens"] = response.usage.prompt_tokens or 0
        usage["completion_tokens"] = response.usage.completion_tokens or 0
        usage["total_tokens"] = response.usage.total_tokens or 0
    except AttributeError as exc:
        print(f"Warning: Error extracting usage: {exc}. Recording 0 tokens.")

    return usage


def _zero_usage() -> Usage:
    """Return an empty token usage dict."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_usage(target: Usage, source: Usage) -> None:
    """Add usage values from source into target in place."""
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = target.get(key, 0) + source.get(key, 0)


def _normalize_answer_text(response_text: str) -> str:
    """Normalize full-width letters, punctuation, and invisible chars."""
    normalized = unicodedata.normalize("NFKC", response_text or "")
    normalized = normalized.replace("\u200b", "")
    normalized = normalized.replace("\ufeff", "")
    return normalized


def _number_to_letter(number_text: str) -> Optional[str]:
    """Map 1-10 style option references to A-J."""
    try:
        number = int(number_text)
    except ValueError:
        return None
    if 1 <= number <= 10:
        return chr(64 + number)
    return None


def extract_answer(response_text: str) -> Optional[str]:
    """
    Extract the answer letter (A-J) from an agent response.

    Handles answer-first English, common Chinese formats, full-width letters,
    option numbers, XML/markdown wrappers, and then preserves the previous
    last-A-J fallback behavior.
    """
    if not response_text:
        return None

    text = _normalize_answer_text(response_text)

    patterns = [
        r'\bMy\s+answer\s+is\s*:?\s*[\(\[\{]?\s*([A-J])\b',
        r'\bFinal\s+answer\s*(?:is)?\s*:?\s*[\(\[\{]?\s*([A-J])\b',
        r'\bAnswer\s*(?:is)?\s*:?\s*[\(\[\{]?\s*([A-J])\b',
        r'\bThe\s+answer\s+is\s*:?\s*[\(\[\{]?\s*([A-J])\b',
        r'\bOption\s*:?\s*[\(\[\{]?\s*([A-J])\b',
        r'\b(?:choose|select|pick|go\s+with)\s+(?:option\s+)?[\(\[\{]?\s*([A-J])\b',
        r'\bI\s+(?:choose|select|pick)\s+(?:option\s+)?[\(\[\{]?\s*([A-J])\b',
        r'(?:\u6211\u7684\u7b54\u6848|\u6700\u7ec8\u7b54\u6848|\u7b54\u6848|\u6b63\u786e\u7b54\u6848|\u9009\u62e9|\u9009\u9879|\u6211\u9009|\u5e94\u9009)\s*(?:\u662f|\u4e3a|:|\uff1a)?\s*[\(\[\{]?\s*([A-J])\b',
        r'(?:\u7b2c\s*)?([A-J])\s*(?:\u9879|\u9009\u9879)\b',
        r'<answer>\s*([A-J])\s*</answer>',
        r'\*\*\s*([A-J])\s*\*\*',
        r'^\s*[\(\[\{]?\s*([A-J])\s*[\)\]\}]?\s*(?:\.|\u3001|:|\uff1a|-|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    number_patterns = [
        r'\b(?:answer|option|choice|choose|select|pick)\s*(?:is|:)?\s*(10|[1-9])\b',
        r'(?:\u7b54\u6848|\u9009\u9879|\u9009\u62e9|\u6211\u9009|\u7b2c)\s*(?:\u662f|\u4e3a|:|\uff1a)?\s*(10|[1-9])\s*(?:\u9879|\u4e2a|\u53f7)?',
    ]
    for pattern in number_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = _number_to_letter(match.group(1))
            if letter:
                return letter

    chinese_ordinals = {
        "\u4e00": "A",
        "\u4e8c": "B",
        "\u4e09": "C",
        "\u56db": "D",
        "\u4e94": "E",
        "\u516d": "F",
        "\u4e03": "G",
        "\u516b": "H",
        "\u4e5d": "I",
        "\u5341": "J",
    }
    match = re.search(
        r'(?:\u7b54\u6848|\u9009\u9879|\u9009\u62e9|\u6211\u9009|\u7b2c)\s*(?:\u662f|\u4e3a|:|\uff1a)?\s*([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341])\s*(?:\u9879|\u4e2a|\u53f7)?',
        text,
    )
    if match:
        return chinese_ordinals.get(match.group(1))

    for line in text.splitlines()[:8]:
        cleaned = line.strip().strip("*`_ \t\r\n")
        match = re.match(
            r'^[\(\[\{]?\s*([A-J])\s*[\)\]\}]?\s*$',
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

    matches = re.findall(r'[A-J]', text)
    if matches:
        return matches[-1].upper()

    return None


def extract_or_repair_answer(
    response_text: str,
    choices: str,
    agent_id: Optional[int] = None,
) -> Tuple[Optional[str], Usage, str]:
    """
    Extract an answer locally, then optionally repair with a short LLM call.

    The repair path is only used when regex extraction returns None. It asks
    the same agent to map its own response to one option letter, preserving the
    original reasoning while avoiding dropped votes caused by formatting.
    """
    answer = extract_answer(response_text)
    if answer:
        return answer, _zero_usage(), ""
    if not ENABLE_LLM_ANSWER_REPAIR:
        return None, _zero_usage(), ""
    if agent_id is None:
        return None, _zero_usage(), ""

    repair_usage_total = _zero_usage()
    repair_transcript = []
    repair_agent_ids = [agent_id]
    if agent_id != 1 and len(AGENT_CONFIGS) >= 1:
        repair_agent_ids.append(1)

    for repair_agent_id in repair_agent_ids:
        client, model_name, agent_name = get_client(repair_agent_id)
        label = (
            f"Agent {agent_id}'s response"
            if repair_agent_id != agent_id
            else "your previous response"
        )
        role_line = (
            f"You are Agent {repair_agent_id} ({agent_name}). "
            "Extract the final multiple-choice answer."
        )
        ownership_line = (
            f"The response below was written by Agent {agent_id}."
            if repair_agent_id != agent_id
            else ""
        )

        prompt = f"""{role_line}
{ownership_line}

Options:
{choices}

Previous response ({label}):
\"\"\"
{response_text}
\"\"\"

Return exactly one letter from A to J. If the response implies an option without using a letter, infer the matching option from the options above. Return only the letter."""

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=20,
                )
                usage = extract_usage(response)
                _add_usage(repair_usage_total, usage)
                repair_text = response.choices[0].message.content or ""
                repair_transcript.append(
                    f"[repair_agent={repair_agent_id}] {repair_text}"
                )
                repaired_answer = extract_answer(repair_text)
                if repaired_answer:
                    return (
                        repaired_answer,
                        repair_usage_total,
                        "\n".join(repair_transcript),
                    )
            except Exception as exc:
                print(
                    f"Agent {repair_agent_id} answer repair attempt "
                    f"{attempt + 1} failed: {exc}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

    return None, repair_usage_total, "\n".join(repair_transcript)


def _build_debate_prompt(
    question: str,
    choices: str,
    agent_id: int,
    agent_name: str,
    predecessor_id: int,
    pred_agent_name: str,
    predecessor_response: str,
    round_num: int,
) -> str:
    """
    Build the debate prompt according to the configured ablation style.

    Styles:
        old: Original critical-evaluation prompt.
        commitment: Adds a stance-retention rule before changing answers.
        evidence: Adds stance-retention plus explicit evidence grounding.
    """
    style = getattr(_config, "DEBATE_PROMPT_STYLE", "old")

    if style == "commitment":
        return f"""You are Agent {agent_id} ({agent_name}) in round {round_num} of a chain debate about a multiple-choice question.

Question: {question}

Options:
{choices}

The debate context from Agent {predecessor_id} ({pred_agent_name}) is:
\"\"\"
{predecessor_response}
\"\"\"

Evaluate the context carefully before deciding:
- If the context includes your previous answer, treat it as your current commitment.
- Do not change your answer merely because another agent is confident or because a summary says there is consensus.
- Change your answer only if the other reasoning exposes a specific error, missing evidence, or stronger option comparison.
- If you keep your answer, briefly explain why the previous reasoning is not strong enough to overturn it.
- If you change your answer, explicitly state what evidence or logic changed your mind.

Output format:
First line: "My answer is: X" where X is a single letter from the available options.
Then give a concise justification in 3-6 sentences."""

    if style == "evidence":
        return f"""You are Agent {agent_id} ({agent_name}) in round {round_num} of a chain debate about a multiple-choice question.

Question: {question}

Options:
{choices}

The debate context from Agent {predecessor_id} ({pred_agent_name}) is:
\"\"\"
{predecessor_response}
\"\"\"

Use an evidence-grounded, commitment-aware decision process:
1. Identify the strongest evidence for your current/previous answer if it is available in the context.
2. Identify the strongest evidence for the predecessor's or summary's answer.
3. Compare the relevant options directly. Do not rely on confidence, consensus, or repeated claims alone.
4. Change your answer only when the new evidence clearly defeats your previous rationale or reveals a concrete mistake.
5. If the passage/question does not provide enough evidence to justify a change, keep your previous answer.
6. Briefly cite or paraphrase the key evidence from the question, passage, or option wording.

Output format:
First line: "My answer is: X" where X is a single letter from the available options.
Then give a concise evidence-based justification in 3-6 sentences."""

    return f"""You are Agent {agent_id} ({agent_name}) in round {round_num} of a chain debate about a multiple-choice question.

Question: {question}

Options:
{choices}

The previous speaker, Agent {predecessor_id} ({pred_agent_name}), said:
\"\"\"
{predecessor_response}
\"\"\"

Carefully evaluate the previous agent's reasoning:
- If you find errors or gaps in their logic, explain what is wrong and provide your own reasoning.
- If their reasoning is sound and convincing, you may agree but must explain why in your own words.
- Do not blindly agree. Think independently and critically.

Output format:
First line: "My answer is: X" where X is a single letter from the available options.
Then give a concise justification in 3-6 sentences."""


def agent_initial_response(
    question: str,
    choices: str,
    agent_id: int,
    temperature: float,
) -> Tuple[str, Usage]:
    """Get initial response from an agent in Round 0."""
    client, model_name, agent_name = get_client(agent_id)

    prompt = f"""You are Agent {agent_id} ({agent_name}), participating in a group discussion about a multiple-choice question.

Question: {question}

Options:
{choices}

Choose the best option carefully, but put the final answer first.

Output format:
First line: "My answer is: X" where X is a single letter from the available options.
Then give a concise justification in 3-6 sentences."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
            )
            usage = extract_usage(response)
            return response.choices[0].message.content, usage
        except Exception as exc:
            print(f"Agent {agent_id} attempt {attempt + 1} failed: {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise


def agent_debate_response(
    question: str,
    choices: str,
    agent_id: int,
    predecessor_id: int,
    predecessor_response: str,
    round_num: int,
    temperature: float,
) -> Tuple[str, Usage]:
    """Get debate response from an agent in Round 1+."""
    client, model_name, agent_name = get_client(agent_id)
    _, _, pred_agent_name = get_client(predecessor_id)

    prompt = _build_debate_prompt(
        question,
        choices,
        agent_id,
        agent_name,
        predecessor_id,
        pred_agent_name,
        predecessor_response,
        round_num,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
            )
            usage = extract_usage(response)
            return response.choices[0].message.content, usage
        except Exception as exc:
            print(
                f"Agent {agent_id} round {round_num} attempt {attempt + 1} failed: {exc}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
