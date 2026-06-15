"""
Devil's Advocate probe for first FULL compression events.

The probe asks one agent to argue against the current consensus. If it extracts
a different answer, the compression is cancelled for that round.
"""

from __future__ import annotations

import random
import time
from typing import Dict, Optional, Tuple

from agent import extract_answer, extract_usage, get_client
from config import AGENT_CONFIGS, DEBATE_TEMP, MAX_RETRIES, RETRY_DELAY


def run_devils_advocate(
    question: str,
    choices: str,
    majority_answer: str,
    agent_id: Optional[int] = None,
) -> Tuple[Optional[str], str, Dict[str, int]]:
    """
    Ask an agent to argue against the current majority answer.

    Args:
        question: Question text.
        choices: Formatted options.
        majority_answer: Current consensus or majority answer.
        agent_id: Optional fixed agent id; random if omitted.

    Returns:
        Tuple of extracted answer, rebuttal text, and token usage.
    """
    if agent_id is None:
        agent_id = random.choice([cfg["agent_id"] for cfg in AGENT_CONFIGS])

    client, model, name = get_client(agent_id)

    prompt = f"""You are Agent {agent_id} ({name}), playing the role of Devil's Advocate.

The current group consensus answer is: {majority_answer}

Question: {question}

Options:
{choices}

Your task: Argue AGAINST the consensus answer "{majority_answer}".
- Find weaknesses in the reasoning that supports {majority_answer}.
- Propose the strongest alternative answer with detailed justification.
- If you genuinely cannot find any flaw, you may agree, but you must explain why after thorough examination.

End your response with "My answer is: X" where X is a single letter from A to J."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=DEBATE_TEMP,
                max_tokens=1000,
            )
            usage = extract_usage(response)
            text = response.choices[0].message.content or ""
            return extract_answer(text), text, usage
        except Exception as exc:
            print(
                f"Devil's Advocate agent {agent_id} attempt {attempt + 1} failed: {exc}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
