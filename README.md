# Multi-Agent Debate Experiment

This project implements three baseline experiments for multi-agent debate on the MMLU-Pro dataset:

- **E0 (Single Agent)**: Only Qwen2.5-7B answers once
- **E1 (Self-Consistency)**: Three heterogeneous models answer independently, majority vote
- **E2 (Chain Debate - MAD-Fixed)**: Three heterogeneous models debate for 3 rounds in chain topology, majority vote

## Models

All models run locally via Ollama:
- **Agent 1**: `qwen2.5:7b` (Alibaba, strong at math reasoning)
- **Agent 2**: `llama3.1:8b` (Meta, strong at English understanding)
- **Agent 3**: `mistral:7b` (Mistral AI, fast inference)

Ollama service: `http://localhost:11434`

## Dataset

**MMLU-Pro** from HuggingFace (`TIGER-Lab/MMLU-Pro`)
- 10 multiple-choice options (A-J), harder than original MMLU
- Initial sample: 50 questions across multiple categories
- Can be expanded to 300 questions

## Installation

1. **Setup Python environment**:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure Ollama is running**:
```bash
ollama serve
```

In another terminal, verify models are available:
```bash
ollama list
```

If models are missing, pull them:
```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

## Project Structure

```
.
├── config.py                  # Configuration: models, rounds, data, timeouts
├── data_loader.py            # MMLU-Pro dataset loading and sampling
├── agent.py                  # Agent API calls, token tracking, answer extraction
├── debate.py                 # Three debate modes: single, sc, chain_debate
├── evaluate.py               # Metrics calculation and comparison
├── run_experiment.py         # Main entry point with argparse
├── test_single_question.py   # Single question validation script
├── results/                  # Experiment results (JSON)
└── venv/                     # Python virtual environment
```

## Usage

### Step 1: Verify Models are Available

```bash
python test_single_question.py
```

This loads one question and tests:
1. All three models can answer independently (Round 0)
2. Chain debate works for one round (Round 1)
3. Token usage is correctly tracked

**Expected output**:
- Question displayed with 10 options
- Each agent's answer and token count
- Majority vote result
- Total tokens consumed

### Step 2: Run E0 (Single Agent Baseline)

```bash
python run_experiment.py --mode single
```

Only Qwen2.5-7B answers. Fastest baseline.

**Output**: `results/single_50q_YYYYMMDD_HHMMSS.json`

### Step 3: Run E1 (Self-Consistency)

```bash
python run_experiment.py --mode sc
```

Three agents answer independently, majority vote. ~3x slower than E0.

**Output**: `results/sc_50q_YYYYMMDD_HHMMSS.json`

### Step 4: Run E2 (Chain Debate)

```bash
python run_experiment.py --mode debate
```

Three agents debate for 3 rounds in chain topology. ~12x slower than E0 (3 agents × 4 rounds).

**Output**: `results/debate_50q_YYYYMMDD_HHMMSS.json`

### Run All Three Modes at Once

```bash
python run_experiment.py --mode all
```

Runs E0 → E1 → E2 with the same 50 questions, then prints comparison table:
```
              Single    SC        Debate
准确率:        xx.x%     xx.x%     xx.x%
总Token:      xx,xxx    xx,xxx    xx,xxx
每题Token:    xxx       xxx       xxx
```

## Configuration

Edit `config.py` to adjust:

**Model & Debate**:
- `AGENT_CONFIGS`: Model names, base_url, API key
- `NUM_AGENTS`: Number of debating agents (default: 3)
- `NUM_ROUNDS`: Total rounds = 1 + NUM_ROUNDS debate rounds (default: 3)
- `TEMPERATURE`: Sampling temperature for Round 0 (default: 0.7)
- `DEBATE_TEMP`: Sampling temperature for debate rounds (default: 0.3)

**Data**:
- `NUM_SAMPLES`: Questions to sample (default: 50, can expand to 300)
- `SEED`: Random seed for reproducibility (default: 42)
- `MIN_CATEGORIES`: Minimum unique categories (default: 5)

**Timeouts & Retries**:
- `MAX_RETRIES`: API call retries (default: 3)
- `RETRY_DELAY`: Seconds between retries (default: 5)
- `OLLAMA_REQUEST_TIMEOUT`: Seconds per request (default: 120)
- `SLEEP_BETWEEN_QUESTIONS`: Seconds between questions (default: 1)
- `SLEEP_BETWEEN_AGENTS`: Seconds between agent calls (default: 0)

## Results Format

Each JSON contains:

**config**: Experiment settings
```json
{
  "mode": "debate",
  "num_rounds": 3,
  "num_samples": 50,
  "temperature": 0.7,
  "agents": [...]
}
```

**metrics**: Overall statistics
```json
{
  "overall_accuracy": 0.75,
  "rounds": {
    "0": {"majority_vote_accuracy": 0.74, ...},
    "1": {"majority_vote_accuracy": 0.75, ...}
  },
  "by_category": {
    "math": {"accuracy": 0.80, "count": 10},
    ...
  },
  "token_usage": {
    "total_tokens": 50000,
    "avg_tokens_per_question": 1000,
    "by_round": {...}
  }
}
```

**results**: Per-question data
```json
[
  {
    "question_id": 0,
    "question": "...",
    "choices": ["A. ...", "B. ...", ...],
    "ground_truth": "B",
    "category": "math",
    "final_answer": "B",
    "answers_by_round": {
      "0": {"1": "B", "2": "A", "3": "B"},
      "1": {"1": "B", "2": "B", "3": "B"}
    },
    "token_usage": {...}
  },
  ...
]
```

## Chain Debate Topology

In chain topology, agents pass their responses in a circle:

**Round 0** (independent):
```
Agent 1 ---> Answer 1
Agent 2 ---> Answer 2
Agent 3 ---> Answer 3
```

**Round 1 onwards** (chain, circular):
```
Agent 1 reads Agent 3's previous response
Agent 2 reads Agent 1's current response (just generated)
Agent 3 reads Agent 2's current response (just generated)
```

This creates a circular information flow where each agent sees exactly one predecessor's response.

## Token Usage Tracking

Token counts are extracted from each API response:
- **prompt_tokens**: Tokens in input prompt
- **completion_tokens**: Tokens in model response
- **total_tokens**: Sum of above

For Ollama, if usage data is not available, zeros are recorded with a warning.

Per-question breakdown:
```
Round 0: 3 agents × 1 call = 3 API calls
Round 1-3: 3 agents × 3 rounds = 9 API calls
Total: 12 API calls per question
```

## Troubleshooting

### Models not loading
```bash
# Check Ollama is running
ollama list

# If models missing, pull them
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

### Out of memory (OOM) errors
- Increase `SLEEP_BETWEEN_AGENTS` in config.py to give Ollama time to unload previous model
- Reduce `OLLAMA_REQUEST_TIMEOUT` if models are timing out (unlikely locally)
- Try running on a machine with more VRAM

### Timeout errors
- Increase `OLLAMA_REQUEST_TIMEOUT` (currently 120 seconds)
- Check Ollama service is responsive: `curl http://localhost:11434/api/tags`

### Token count is 0
- Ollama may not return usage information; this is expected
- All zeros in usage dict will be recorded for that call
- Total token counts will reflect actual API usage but may be incomplete

### HuggingFace dataset loading slow
- First load downloads dataset cache (~2-3 GB)
- Subsequent runs use cached data
- Check internet connection if download fails

## Performance Notes

**Estimated runtimes** (for 50 questions):
- E0 (Single): ~5-10 minutes
- E1 (Self-Consistency): ~15-30 minutes
- E2 (Chain Debate): ~60-120 minutes (longest due to multiple rounds)

Models load to memory on first call (~10-30 seconds). Subsequent calls are faster.

## Citation

MMLU-Pro dataset:
```
TIGER-Lab/MMLU-Pro on HuggingFace
https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
```

## License

This project is for research purposes.
