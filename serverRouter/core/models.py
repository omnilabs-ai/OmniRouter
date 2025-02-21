from serverRouter.core.datamodels import ModelInfo, ModelProvider

# Primary chat models registry
CHAT_MODELS = {
    "gpt-4": ModelInfo(
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        description="OpenAI's most capable model for both language understanding and generation",
        max_tokens=8192,
        benchmarks={
            "MMLU": 0.864,
            "GPQA": 0.414,
            "HumanEval": 0.866,
            "MATH": 0.645,
            "BFCL": 0.883,
            "MGSM": 0.86
        },
        tokenCost=0.03,  # mock
        latency=2.5  # mock
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        description="OpenAI's fast and efficient model with good capabilities",
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.700,
            "GPQA": 0.308,
            "HumanEval": 0.680,
            "MATH": 0.341,
            "BFCL": 0.644,
            "MGSM": 0.563
        },
        tokenCost=0.002,  # mock
        latency=0.8  # mock
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        description="OpenAI's GPT-4o variant (mini) for lightweight generation (requires reasoning access, Tier 5)",
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.820,
            "GPQA": 0.308,
            "HumanEval": 0.402,
            "MATH": 0.702,
            "BFCL": 0.641,
            "MGSM": 0.870
        },
        tokenCost=0.015,  # mock
        latency=1.5  # mock
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        description="OpenAI's GPT-4o model for generation without heavy reasoning (suitable for non–Tier 5 accounts)",
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.887,
            "GPQA": 0.536,
            "HumanEval": 0.902,
            "MATH": 0.776,
            "BFCL": 0.805,
            "MGSM": 0.905
        },
        tokenCost=0.025,  # mock
        latency=2.0  # mock
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        description="Anthropic's most capable model",
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.857,
            "GPQA": 0.504,
            "HumanEval": 0.849,
            "MATH": 0.601,
            "BFCL": 0.884,
            "MGSM": 0.907
        },
        tokenCost=0.028,  # mock
        latency=2.2  # mock
    ),
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        description="Anthropic's balanced model for performance and efficiency",
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.883,
            "GPQA": 0.594,
            "HumanEval": 0.920,
            "MATH": 0.711,
            "BFCL": 0.902,
            "MGSM": 0.916
        },
        tokenCost=0.018,  # mock
        latency=1.8  # mock
    ),
    "deepseek-v3": ModelInfo(
        name="deepseek-chat",   # Use the exact string expected by the API.
        provider=ModelProvider.DEEPSEEK,
        description="DeepSeek's general-purpose chat model.",
        max_tokens=4096
    ),
    "deepseek-r1": ModelInfo(
        name="deepseek-reasoner",  # Must match API's exact model name
        provider=ModelProvider.DEEPSEEK,
        description="DeepSeek-R1 model for tool/API calling",
        max_tokens=4096
    ),
    # --- New Gemini Chat Models ---
    "gemini-2.0-flash-lite": ModelInfo(
        name="gemini-2.0-flash-lite-preview-02-05",
        provider=ModelProvider.GEMINI,
        description=(
            "Google's most advanced model for technical and scientific tasks. "
            "Strengths: STEM subjects, code generation, and multimodal reasoning. "
            "Weaknesses: Less conversational."
        ),
        max_tokens=1048576,
        benchmarks={
            "MMLU": 0.899,
            "GPQA": 0.624,
            "HumanEval": 0.929,
            "MATH": 0.897,
            "BFCL": 0.891,
            "MGSM": 0.887
        },
        tokenCost=0.02,  # mock
        latency=1.7  # mock
    ),
    "gemini-2.0-flash-lite": ModelInfo(
        name="gemini-2.0-flash-lite-preview-02-05",
        provider=ModelProvider.GEMINI,
        description="Gemini 2.0 Flash Lite",
        max_tokens=8192,  # Adjust as needed
    ),
    "gemini-2.0-pro": ModelInfo(
        name="gemini-2.0-pro-exp-02-05",
        provider=ModelProvider.GEMINI,
        description="Gemini 2.0 Pro",
        max_tokens=8192,  # Adjust as needed
    ),
    "learnlm-1.5-pro-experimental": ModelInfo(
        name="learnlm-1.5-pro-experimental",
        provider=ModelProvider.GEMINI,
        description="LearnLM 1.5 Pro Experimental (Audio, images, videos, and text input)",
        max_tokens=8192,  # Adjust as needed
    )
}

# Primary image models registry
IMAGE_MODELS = {
    "dall-e-3": ModelInfo(
        name="dall-e-3",
        provider=ModelProvider.OPENAI,
        description="OpenAI's most advanced image generation model"
    ),
    "dall-e-2": ModelInfo(
        name="dall-e-2",
        provider=ModelProvider.OPENAI,
        description="OpenAI's efficient image generation model",
    ),
}

# Combined models dictionary
MODELS = {**CHAT_MODELS, **IMAGE_MODELS}