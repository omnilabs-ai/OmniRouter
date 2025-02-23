from serverRouter.core.datamodels import ModelInfo, ModelProvider

# Primary chat models registry
CHAT_MODELS = {
    "gpt-4": ModelInfo(
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        description=(
            "OpenAI's flagship model excelling in complex reasoning and coding. "
            "Strengths: Technical explanations, API integrations, and multi-step problem solving. "
            "Weaknesses: Higher cost, slower response times."
        ),
        max_tokens=8192,
        benchmarks={
            "MMLU": 0.864,
            "GPQA": 0.414,
            "HumanEval": 0.866,
            "MATH": 0.645,
            "BFCL": 0.883,
            "MGSM": 0.86
        },
        tokenCost=0.045,  # Average of input ($0.03) and output ($0.06) costs
        latency=1.0  # ~60 tokens/sec, with 0.6-0.7s initial latency
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        description=(
            "GPT-3.5 (ChatGPT) is a versatile conversational model adept at general Q&A, basic reasoning, text completion, and casual dialogue. "
            "Strengths: Fast, cost-efficient, and well-suited for everyday tasks. "
            "Weaknesses: Weaker complex reasoning and limited knowledge compared to GPT-4."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.700,
            "GPQA": 0.308,
            "HumanEval": 0.680,
            "MATH": 0.341,
            "BFCL": 0.644,
            "MGSM": 0.563
        },
        tokenCost=0.00175,  # Average of input ($0.0015) and output ($0.002) costs
        latency=0.5  # ~70 tokens/sec, with <0.5s initial latency
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        description=(
            "GPT-4o mini enables a broad range of tasks with its low cost and latency. "
            "Strengths: Efficient for chaining multiple model calls and handling large context. "
            "Weaknesses: Lower performance on complex reasoning tasks."
        ),
        max_tokens=128000,
        benchmarks={
            "MMLU": 0.820,
            "GPQA": 0.308,
            "HumanEval": 0.402,
            "MATH": 0.702,
            "BFCL": 0.641,
            "MGSM": 0.870
        },
        tokenCost=0.000375,  # Average of input ($0.00015) and output ($0.0006) costs
        latency=0.3  # Estimated >100 tokens/sec
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        description=(
            "GPT-4o strikes a balance between quality and efficiency, offering moderate benchmark performance. "
            "Strengths: Fast response times and improved multilingual capabilities. "
            "Weaknesses: Less capable than full GPT-4 on complex tasks."
        ),
        max_tokens=128000,
        benchmarks={
            "MMLU": 0.887,
            "GPQA": 0.536,
            "HumanEval": 0.902,
            "MATH": 0.776,
            "BFCL": 0.805,
            "MGSM": 0.905
        },
        tokenCost=0.00625,  # Average of input ($0.0025) and output ($0.01) costs
        latency=0.8  # ~63 tokens/sec
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Claude 3 Opus is designed for extremely complex tasks with an extended context. "
            "Strengths: High knowledge, advanced reasoning, and excellent multilingual performance. "
            "Weaknesses: May be less creative than GPT-4 and has higher latency."
        ),
        max_tokens=200000,
        benchmarks={
            "MMLU": 0.857,
            "GPQA": 0.504,
            "HumanEval": 0.849,
            "MATH": 0.601,
            "BFCL": 0.884,
            "MGSM": 0.907
        },
        tokenCost=0.045,  # Average of input ($0.015) and output ($0.075) costs
        latency=2.1  # ~25-27 tokens/sec, 2.1s to first token
    ),
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Anthropic's top performer for complex Q&A and multilingual tasks. "
            "Strengths: Research-grade answers, non-English queries, and precise tool usage. "
            "Weaknesses: Less creative compared to GPT-4."
        ),
        max_tokens=200000,
        benchmarks={
            "MMLU": 0.883,
            "GPQA": 0.594,
            "HumanEval": 0.920,
            "MATH": 0.711,
            "BFCL": 0.902,
            "MGSM": 0.916
        },
        tokenCost=0.009,  # Average of input ($0.003) and output ($0.015) costs
        latency=0.9  # ~72 tokens/sec, 0.9-1.0s to first token
    ),
    "deepseek-v3": ModelInfo(
        name="deepseek-chat",
        provider=ModelProvider.DEEPSEEK,
        description=(
            "DeepSeek Chat is an open‑source conversational model known for cost‑effective, high‐quality dialogue. "
            "Strengths: Excellent general conversation and Q&A. "
            "Weaknesses: May lag behind in complex coding or reasoning tasks."
        ),
        max_tokens=64000,
        benchmarks={
            "MMLU": 0.880,
            "GPQA": 0.590,
            "HumanEval": 0.830,
            "MATH": 0.900,
            "BFCL": 0.0,
            "MGSM": 0.850
        },
        tokenCost=0.0007,  # Average of input ($0.00027) and output ($0.0011) costs
        latency=2.8  # 2.8s to first token
    ),
    "deepseek-r1": ModelInfo(
        name="deepseek-reasoner",
        provider=ModelProvider.DEEPSEEK,
        description=(
            "DeepSeek Reasoner (R1) is specialized for in-depth logical reasoning and multi-step problem solving. "
            "Strengths: Exceptional reasoning and math problem-solving; competitive with top-tier models on complex tasks. "
            "Weaknesses: May be slower and less polished in casual conversation."
        ),
        max_tokens=64000,
        benchmarks={
            "MMLU": 0.908,
            "GPQA": 0.715,
            "HumanEval": 0.830,
            "MATH": 0.840,
            "BFCL": 0.0,
            "MGSM": 0.900
        },
        tokenCost=0.00137,  # Average of input ($0.00055) and output ($0.00219) costs
        latency=3.5  # Estimated based on additional reasoning overhead
    ),
    "gemini-2.0-flash-lite": ModelInfo(
        name="gemini-2.0-flash-lite-preview-02-05",
        provider=ModelProvider.GEMINI,
        description=(
            "Gemini 2.0 Flash Lite is a lightweight, high-speed variant optimized for low latency. "
            "Strengths: Fast response and efficient for general queries. "
            "Weaknesses: May provide less detailed responses than Gemini Pro."
        ),
        max_tokens=1048576,  # 1M tokens
        benchmarks={
            "MMLU": 0.800,
            "GPQA": 0.580,
            "HumanEval": 0.900,
            "MATH": 0.890,
            "BFCL": 0.870,
            "MGSM": 0.880
        },
        tokenCost=0.00026,  # Average of input ($0.00013) and output ($0.00038) costs
        latency=0.3  # ~185 tokens/sec
    ),
    "gemini-2.0-pro": ModelInfo(
        name="gemini-2.0-pro-exp-02-05",
        provider=ModelProvider.GEMINI,
        description=(
            "Google's most advanced model for technical and scientific tasks. "
            "Strengths: STEM subjects, code generation, and multimodal reasoning. "
            "Weaknesses: Less conversational."
        ),
        max_tokens=32768,
        benchmarks={
            "MMLU": 0.899,
            "GPQA": 0.624,
            "HumanEval": 0.929,
            "MATH": 0.897,
            "BFCL": 0.891,
            "MGSM": 0.887
        },
        tokenCost=0.00525,  # Average of input ($0.0035) and output ($0.007) costs based on 1.5 Pro
        latency=1.2  # ~49 tokens/sec
    ),
    "learnlm-1.5-pro-experimental": ModelInfo(
        name="learnlm-1.5-pro-experimental",
        provider=ModelProvider.GEMINI,
        description=(
            "LearnLM 1.5 Pro Experimental is a versatile multimodal model. "
            "Strengths: Audio, image, video processing and natural language understanding. "
            "Weaknesses: Experimental features may be unstable."
        ),
        max_tokens=2097152,  # 2M tokens
        benchmarks={
            "MMLU": 0.850,
            "GPQA": 0.550,
            "HumanEval": 0.880,
            "MATH": 0.820,
            "BFCL": 0.830,
            "MGSM": 0.860
        },
        tokenCost=0.00525,  # Using Gemini 1.5 Pro pricing as reference
        latency=1.5  # Estimated based on large context and multimodal processing
    )
}

# Primary image models registry
IMAGE_MODELS = {
    "dall-e-3": ModelInfo(
        name="dall-e-3",
        provider=ModelProvider.OPENAI,
        description=(
            "OpenAI's most advanced image generation model. "
            "Strengths: Highly detailed and photorealistic images, excellent prompt following. "
            "Weaknesses: Higher cost and longer generation time."
        ),
    ),
    "dall-e-2": ModelInfo(
        name="dall-e-2",
        provider=ModelProvider.OPENAI,
        description=(
            "OpenAI's efficient image generation model. "
            "Strengths: Fast generation and good quality for common use cases. "
            "Weaknesses: Less detailed and accurate compared to DALL-E 3."
        ),
    )
}

# Combined models dictionary
MODELS = {**CHAT_MODELS, **IMAGE_MODELS}