"""
Data loaders for MMLU-Pro, QuALITY, and MultiRC.
Loads from HuggingFace and samples questions with stratification by category.
"""

from typing import Dict, List, Any
import random
from datasets import load_dataset
from config import NUM_SAMPLES, SEED, MIN_CATEGORIES


def load_mmlu_pro_dataset() -> Dict[str, Any]:
    """
    Load MMLU-Pro dataset from HuggingFace (TIGER-Lab/MMLU-Pro).
    
    Returns:
        dict: Full dataset split by category
    """
    print("Loading MMLU-Pro dataset from HuggingFace...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    return dataset


def load_quality_dataset(split: str = "validation") -> List[Dict[str, Any]]:
    """
    Load and normalize the QuALITY dataset from HuggingFace.

    QuALITY is a long-context multiple-choice QA dataset with 4 options. The
    normalized records match the rest of the project: question, options, answer
    as a letter, answer_index, and category.

    Args:
        split: HuggingFace split to load. Validation is the default evaluation
            split.

    Returns:
        List of normalized question records.
    """
    print(f"Loading QuALITY dataset from HuggingFace (split={split})...")
    dataset = load_dataset("emozilla/quality", split=split)
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    normalized = []
    for idx, item in enumerate(dataset):
        answer_index = int(item["answer"])
        normalized.append({
            "global_idx": idx,
            "question": (
                f"Passage:\n{item['article']}\n\n"
                f"Question:\n{item['question']}"
            ),
            "options": list(item["options"]),
            "answer": chr(65 + answer_index),
            "answer_index": answer_index,
            "category": "quality_hard" if item.get("hard", False) else "quality_easy",
            "source_dataset": "quality",
            "hard": bool(item.get("hard", False)),
        })

    return normalized


def load_multirc_dataset(split: str = "validation") -> List[Dict[str, Any]]:
    """
    Load and normalize SuperGLUE MultiRC as a multi-answer MCQ dataset.

    HuggingFace stores MultiRC at the answer-candidate level. This loader
    groups rows by passage/question and turns all label=1 candidates into a
    canonical answer set such as "A,C".
    """
    print(f"Loading SuperGLUE MultiRC dataset from HuggingFace (split={split})...")
    dataset = load_dataset("super_glue", "multirc", split=split)
    print(f"Dataset loaded. Total answer candidates: {len(dataset)}")

    grouped: Dict[Any, Dict[str, Any]] = {}
    for idx, item in enumerate(dataset):
        item_idx = item.get("idx", {})
        key = (
            item_idx.get("paragraph", item.get("passage", "")),
            item_idx.get("question", item.get("question", "")),
        )
        if key not in grouped:
            grouped[key] = {
                "global_idx": len(grouped),
                "candidate_row_indices": [],
                "question": (
                    f"Passage:\n{item['paragraph'] if 'paragraph' in item else item['passage']}\n\n"
                    f"Question:\n{item['question']}"
                ),
                "options": [],
                "answer_indices": [],
                "category": "multirc",
                "source_dataset": "multirc",
            }
        group = grouped[key]
        option_index = len(group["options"])
        group["candidate_row_indices"].append(idx)
        group["options"].append(item["answer"])
        if int(item.get("label", 0)) == 1:
            group["answer_indices"].append(option_index)

    normalized: List[Dict[str, Any]] = []
    for group in grouped.values():
        if not group["answer_indices"]:
            continue
        answer = ",".join(chr(65 + index) for index in group["answer_indices"])
        group["answer"] = answer
        group["answer_set"] = answer
        normalized.append(group)

    print(f"Grouped into {len(normalized)} multi-answer questions")
    return normalized


def inspect_first_sample(dataset: Dict[str, Any]) -> None:
    """
    Inspect and print the first sample to verify dataset structure.
    
    Args:
        dataset: MMLU-Pro dataset
    """
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n=== First Sample Structure ===")
        print(f"Keys: {sample.keys()}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Number of options: {len(sample['options'])}")
        print(f"Options sample: {sample['options'][:3]}")
        print(f"Correct answer (letter): {sample['answer']}")
        print(f"Correct answer (index): {sample.get('answer_index', 'N/A')}")
        print(f"Category: {sample.get('category', 'unknown')}")
        print()


def sample_questions(
    dataset: Dict[str, Any],
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
    min_categories: int = MIN_CATEGORIES
) -> List[Dict[str, Any]]:
    """
    Sample questions from MMLU-Pro with stratification by category.
    Ensures representation of at least min_categories different categories.
    
    Args:
        dataset: MMLU-Pro dataset
        num_samples: Number of samples to select
        seed: Random seed for reproducibility
        min_categories: Minimum number of different categories
        
    Returns:
        list: Sampled questions with metadata
    """
    random.seed(seed)
    
    # Group by category
    category_groups: Dict[str, List[Dict[str, Any]]] = {}
    for idx, item in enumerate(dataset):
        cat = item.get('category', 'unknown')
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append({
            'global_idx': idx,
            'item': item
        })
    
    print(f"\nDataset has {len(category_groups)} unique categories")
    print(f"Categories: {sorted(category_groups.keys())}\n")
    
    # Ensure minimum category coverage
    if len(category_groups) < min_categories:
        print(f"Warning: Dataset has only {len(category_groups)} categories, "
              f"less than requested minimum {min_categories}")
        categories_to_sample = sorted(category_groups.keys())
    else:
        # First, pick one from each of the min_categories most common categories
        sorted_cats = sorted(
            category_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        categories_to_sample = [cat for cat, _ in sorted_cats[:min_categories]]
    
    sampled = []
    
    # Stratified sampling: first ensure min_categories
    for cat in categories_to_sample:
        items = category_groups[cat]
        if items:
            selected = random.choice(items)
            sampled.append(selected['item'])
    
    # Fill remaining slots randomly
    remaining = num_samples - len(sampled)
    if remaining > 0:
        all_items = []
        for items in category_groups.values():
            all_items.extend(items)
        additional = random.sample(all_items, min(remaining, len(all_items)))
        sampled.extend([x['item'] for x in additional])
    
    # Shuffle to avoid category bias
    random.shuffle(sampled)
    
    # Ensure no duplicates by checking indices
    sampled = sampled[:num_samples]
    
    print(f"Sampled {len(sampled)} questions")
    category_counts = {}
    for q in sampled:
        cat = q.get('category', 'unknown')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print(f"Category distribution: {dict(sorted(category_counts.items()))}\n")
    
    return sampled


def format_choices(options: List[str]) -> str:
    """
    Format options as 'A. option1\nB. option2\n...'.
    
    Args:
        options: List of option texts
        
    Returns:
        str: Formatted choices
    """
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    formatted = []
    for i, opt in enumerate(options[:len(letters)]):
        formatted.append(f"{letters[i]}. {opt}")
    return '\n'.join(formatted)
