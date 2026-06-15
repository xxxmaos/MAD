"""
Single question test script.
Validates that all three models can be called and demonstrates chain debate flow.
"""

import time
from typing import Dict, Any
from tqdm import tqdm

from config import NUM_ROUNDS, TEMPERATURE, DEBATE_TEMP, SLEEP_BETWEEN_AGENTS, AGENT_CONFIGS
from data_loader import load_mmlu_pro_dataset, format_choices
from agent import agent_initial_response, agent_debate_response, extract_answer


def get_agent_name(agent_id: int) -> str:
    """Get agent name from config."""
    return AGENT_CONFIGS[agent_id - 1]["name"]


def test_single_question():
    """
    Test script: Load one question and run through E0, then one round of E2.
    
    Output format:
    - Question and options
    - Ground truth
    - Round 0: Each agent answers independently
    - Round 1: Chain debate (1 <- 3 <- 2 <- 1)
    - Majority vote result
    """
    print("\n" + "="*70)
    print("TEST: Single Question with All Models")
    print("="*70 + "\n")
    
    # Load first question
    dataset = load_mmlu_pro_dataset()
    question_data = dataset[0]
    
    question = question_data["question"]
    options = question_data["options"]
    ground_truth = question_data["answer"]
    category = question_data.get("category", "unknown")
    
    # Format for display
    choices_formatted = format_choices(options)
    
    # Print question
    print(f"Question: {question[:100]}...")
    print(f"Category: {category}")
    print(f"\nOptions:")
    print(choices_formatted)
    print(f"\nGround truth: {ground_truth}\n")
    
    # ========================================================================
    # ROUND 0: Independent responses
    # ========================================================================
    print("-" * 70)
    print("ROUND 0: Independent Responses (E0 Baseline)")
    print("-" * 70 + "\n")
    
    round0_history = {}
    round0_answers = {}
    round0_tokens = {}
    
    for agent_id in [1, 2, 3]:
        print(f"  Calling Agent {agent_id} ({get_agent_name(agent_id)})...")
        try:
            text, usage = agent_initial_response(question, choices_formatted, agent_id, TEMPERATURE)
            answer = extract_answer(text)
            
            round0_history[agent_id] = text
            round0_answers[agent_id] = answer
            round0_tokens[agent_id] = usage
            
            # Print summary
            text_preview = text.replace("\n", " ")[:80]
            print(f"    ✓ Answer: {answer}  |  Tokens: {usage['total_tokens']}")
            print(f"      Response preview: {text_preview}...\n")
            
        except Exception as e:
            print(f"    ✗ Error: {e}\n")
            round0_history[agent_id] = f"[ERROR] {str(e)}"
            round0_answers[agent_id] = None
            round0_tokens[agent_id] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        time.sleep(SLEEP_BETWEEN_AGENTS)
    
    # ========================================================================
    # ROUND 1: Chain debate
    # ========================================================================
    print("\n" + "-" * 70)
    print("ROUND 1: Chain Debate (E2 Baseline)")
    print("-" * 70)
    print("Debate order: Agent 1 <- Agent 3 <- Agent 2 <- Agent 1 (circular)\n")
    
    round1_history = {}
    round1_answers = {}
    round1_tokens = {}
    
    agent_ids = [1, 2, 3]
    
    for idx, agent_id in enumerate(agent_ids):
        # Determine predecessor (circular for agent 1)
        if idx == 0:
            pred_id = agent_ids[-1]  # Agent 3
            if pred_id not in round0_history:
                print(f"  Agent {agent_id} ({get_agent_name(agent_id)}) ✗ Skipped (Agent {pred_id} had error)\n")
                round1_answers[agent_id] = None
                round1_tokens[agent_id] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                continue
            pred_response = round0_history[pred_id]
            pred_label = f"Agent {pred_id}'s R0 response"
        else:
            pred_id = agent_ids[idx - 1]  # Previous in chain
            if pred_id not in round1_history:
                print(f"  Agent {agent_id} ({get_agent_name(agent_id)}) ✗ Skipped (Agent {pred_id} had error)\n")
                round1_answers[agent_id] = None
                round1_tokens[agent_id] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                continue
            pred_response = round1_history[pred_id]  # This round's latest
            pred_label = f"Agent {pred_id}'s R1 response (just generated)"
        
        print(f"  Agent {agent_id} ({get_agent_name(agent_id)}) ← reads {pred_label}")
        
        try:
            text, usage = agent_debate_response(
                question, choices_formatted, agent_id,
                pred_id, pred_response, 1, DEBATE_TEMP
            )
            answer = extract_answer(text)
            
            round1_history[agent_id] = text
            round1_answers[agent_id] = answer
            round1_tokens[agent_id] = usage
            
            # Print summary
            text_preview = text.replace("\n", " ")[:80]
            print(f"    ✓ Answer: {answer}  |  Tokens: {usage['total_tokens']}")
            print(f"      Response preview: {text_preview}...\n")
            
        except Exception as e:
            print(f"    ✗ Error: {e}\n")
            round1_history[agent_id] = f"[ERROR] {str(e)}"
            round1_answers[agent_id] = None
            round1_tokens[agent_id] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        time.sleep(SLEEP_BETWEEN_AGENTS)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    # Majority vote
    from collections import Counter
    
    final_answers = [a for a in round1_answers.values() if a is not None]
    if final_answers:
        majority_answer = Counter(final_answers).most_common(1)[0][0]
        is_correct = majority_answer == ground_truth
        correct_symbol = "✓" if is_correct else "✗"
    else:
        majority_answer = None
        correct_symbol = "?"
    
    print(f"Round 0 answers: {', '.join(f'A{aid}={round0_answers[aid]}' for aid in [1,2,3])}")
    print(f"Round 1 answers: {', '.join(f'A{aid}={round1_answers[aid]}' for aid in [1,2,3])}")
    print(f"\nMajority vote (R1): {majority_answer}  {correct_symbol} (ground truth: {ground_truth})")
    
    # Token summary
    total_tokens = sum(t["total_tokens"] for t in round0_tokens.values()) + \
                   sum(t["total_tokens"] for t in round1_tokens.values())
    
    print(f"\nToken Usage:")
    print(f"  Round 0 total: {sum(t['total_tokens'] for t in round0_tokens.values()):,}")
    for aid in [1, 2, 3]:
        print(f"    Agent {aid}: {round0_tokens[aid]['total_tokens']}")
    
    print(f"  Round 1 total: {sum(t['total_tokens'] for t in round1_tokens.values()):,}")
    for aid in [1, 2, 3]:
        print(f"    Agent {aid}: {round1_tokens[aid]['total_tokens']}")
    
    print(f"  TOTAL: {total_tokens:,} tokens")
    
    print("\n" + "="*70)
    if final_answers and all(a == final_answers[0] for a in final_answers):
        print("✓ All agents agreed in Round 1")
    else:
        print("✗ Agents disagreed in Round 1 (debate is working)")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_single_question()
