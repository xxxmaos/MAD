"""
Configuration file for Multi-Agent Debate Experiment.
Defines model configs, debate settings, data loading params, and Ollama connection details.
"""

# ============================================================================
# AGENT CONFIGURATIONS - Three Heterogeneous Models via Ollama
# ============================================================================
AGENT_CONFIGS = [
    {
        "agent_id": 1,
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Qwen2.5-7B"
    },
    {
        "agent_id": 2,
        "model": "llama3.1:8b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Llama3.1-8B"
    },
    {
        "agent_id": 3,
        "model": "mistral:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Mistral-7B"
    },
]

# Alternative 3-agent profile for stronger QuALITY/debate experiments.
# Keep the number of agents at 3 so results remain comparable to the
# existing baseline topology.
BASE_AGENT_CONFIGS = [cfg.copy() for cfg in AGENT_CONFIGS]
STRONG_AGENT_CONFIGS = [
    {
        "agent_id": 1,
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Qwen3-8B"
    },
    {
        "agent_id": 2,
        "model": "deepseek-r1:14b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "DeepSeek-R1-14B"
    },
    {
        "agent_id": 3,
        "model": "phi4:14b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Phi-4-14B"
    },
]
FIVE_AGENT_CONFIGS = [
    {
        "agent_id": 1,
        "model": "deepseek-r1:14b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "DeepSeek-R1-14B"
    },
    {
        "agent_id": 2,
        "model": "gemma4:e4b-it-q4_K_M",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Gemma4-E4B-Instruct"
    },
    {
        "agent_id": 3,
        "model": "phi4:14b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Phi-4-14B"
    },
    {
        "agent_id": 4,
        "model": "glm4:9b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "GLM4-9B"
    },
    {
        "agent_id": 5,
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Qwen3-8B"
    },
]
SEVEN_AGENT_CONFIGS = [
    *[cfg.copy() for cfg in FIVE_AGENT_CONFIGS],
    {
        "agent_id": 6,
        "model": "mistral:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Mistral-7B"
    },
    {
        "agent_id": 7,
        "model": "command-r7b:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "name": "Command-R7B"
    },
]
AGENT_PROFILES = {
    "default": BASE_AGENT_CONFIGS,
    "strong": STRONG_AGENT_CONFIGS,
    "five": FIVE_AGENT_CONFIGS,
    "seven": SEVEN_AGENT_CONFIGS,
}
AGENT_PROFILE = "default"

# Debate prompt style for prompt ablations:
# old: original debate prompt
# commitment: requires caution before changing stance
# evidence: commitment-aware plus explicit evidence grounding
DEBATE_PROMPT_STYLE = "old"
VALID_DEBATE_PROMPT_STYLES = ["old", "commitment", "evidence"]

# ============================================================================
# DEBATE CONFIGURATION
# ============================================================================
NUM_AGENTS = 3
NUM_ROUNDS = 3  # Round 0 (independent) + 3 debate rounds
TEMPERATURE = 0.7  # Round 0: higher temperature for diversity
DEBATE_TEMP = 0.3  # Rounds 1+: lower temperature for consistency
TOPOLOGY = "chain"  # Chain debate topology (Agent 1 <- Agent 3 <- Agent 2 <- Agent 1)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
NUM_SAMPLES = 50  # Full dataset for comprehensive evaluation
SEED = 42
MIN_CATEGORIES = 5  # Ensure at least 5 different categories in sample

# ============================================================================
# RETRY AND TIMEOUT CONFIGURATION
# ============================================================================
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds between retries (Ollama models load slowly)
OLLAMA_REQUEST_TIMEOUT = 120  # Seconds (local models are slower than API)

# ============================================================================
# OLLAMA GPU CONFIGURATION
# ============================================================================
# The experiment runners can automatically pick up to four available RTX 4090 D
# cards, start one Ollama server per GPU, and route agents across those servers.
OLLAMA_AUTO_GPU_ENABLED = True
OLLAMA_MIN_GPU_COUNT = 1
OLLAMA_MAX_GPU_COUNT = 4
OLLAMA_GPU_NAME_FILTER = "4090 D"
OLLAMA_MIN_FREE_MEMORY_MB = 12000
OLLAMA_IDLE_UTILIZATION = 10
OLLAMA_HOST = "127.0.0.1"
OLLAMA_BASE_PORT = 11440
OLLAMA_LOG_DIR = "ablation_result/ollama_logs"
OLLAMA_NUM_PARALLEL = 1
OLLAMA_MAX_LOADED_MODELS = 1
OLLAMA_KEEP_ALIVE = "-1"
OLLAMA_GPU_ASSIGNMENT = None

# Approximate resident VRAM footprints. These are used only for balancing
# models across the two selected GPUs; Ollama still owns the actual loading.
MODEL_MEMORY_ESTIMATES_GB = {
    "qwen2.5:7b": 4.7,
    "llama3.1:8b": 4.9,
    "mistral:7b": 4.4,
    "qwen3:8b": 5.2,
    "deepseek-r1:14b": 9.0,
    "phi4:14b": 9.1,
    "gemma4:e4b-it-q4_k_m": 3.0,
    "glm4:9b": 6.0,
    "command-r7b:7b": 4.5,
    "nomic-embed-text": 0.3,
}

# ============================================================================
# SLEEP CONFIGURATION FOR OLLAMA MODEL SWITCHING
# ============================================================================
SLEEP_BETWEEN_QUESTIONS = 1  # Seconds, give Ollama time to switch models
SLEEP_BETWEEN_AGENTS = 0  # Seconds, can be increased if OOM issues occur

# ============================================================================
# COMPRESSION + EARLY EXIT MECHANISM CONFIGURATION
# ============================================================================
# Summarizer is Agent 1 (Qwen).
SUMMARIZER_AGENT_ID = 1

# Compression thresholds use absolute disputed-option counts.
# FULL: n_disputed = 0
# PARTIAL: 1 <= n_disputed <= DISPUTED_PARTIAL_MAX
# NONE: n_disputed > DISPUTED_PARTIAL_MAX
DISPUTED_PARTIAL_MAX = 2  # n_disputed >= 3 is NONE for three agents.

# Early exit.
EARLY_EXIT_R0 = False
MIN_DEBATE_ROUNDS = 2
MAX_DEBATE_ROUNDS = 6
CONSECUTIVE_STABLE_NEEDED = 2
DEADLOCK_PATIENCE = 4

# Devil's Advocate.
ENABLE_DEVILS_ADVOCATE = False

# Question-aware evidence retrieval
EVIDENCE_EMBEDDING_MODEL = "nomic-embed-text"
EVIDENCE_EMBEDDING_BASE_URL = "http://localhost:11434"

# ============================================================================
# MECHANISM MODE CONFIGURATION (run_experiment.py --mode mechanism)
# ============================================================================
# Summarizer is Agent 1 (Qwen).
SUMMARIZER_AGENT_ID = 1
SUMMARIZER_TEMPERATURE = 0.1
SUMMARIZER_MAX_TOKENS = 800

# Debate-only early exit. No phase-zero exit and no Devil's Advocate.
MIN_DEBATE_ROUNDS = 2
MAX_DEBATE_ROUNDS = 6
CONSECUTIVE_STABLE_NEEDED = 2

# MMLU-Pro option count.
NUM_OPTIONS = 10
