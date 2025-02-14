from serverRouter.core.datamodels import ModelInfo, ModelProvider

# Primary chat models registry
CHAT_MODELS = {
    "gpt-4": ModelInfo(
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        description="OpenAI's most capable model for both language understanding and generation",
        max_tokens=8192
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        description="OpenAI's fast and efficient model with good capabilities",
        max_tokens=4096
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        description="OpenAI's GPT-4o variant (mini) for lightweight generation (requires reasoning access, Tier 5)",
        max_tokens=4096
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        description="OpenAI's GPT-4o model for generation without heavy reasoning (suitable for non–Tier 5 accounts)",
        max_tokens=4096
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        description="Anthropic's most capable model",
        max_tokens=4096,
    ),
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        description="Anthropic's balanced model for performance and efficiency",
        max_tokens=4096,
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
    "gemini-2.0-flash": ModelInfo(
        name="gemini-2.0-flash",
        provider=ModelProvider.GEMINI,
        description="Google Gemini 2.0 Flash for chat completions",
        max_tokens=4096,
    ),
    "gemini-2.0-pro-exp-02-05": ModelInfo(
        name="gemini-2.0-pro-exp-02-05",
        provider=ModelProvider.GEMINI,
        description="Gemini 2.0 Pro (Improved quality)",
        max_tokens=8192,  # Adjust as needed
    ),
    "gemini-2.0-flash-thinking-exp-01-21": ModelInfo(
        name="gemini-2.0-flash-thinking-exp-01-21",
        provider=ModelProvider.GEMINI,
        description="Gemini 2.0 Flash Thinking (Reasoning)",
        max_tokens=8192,  # Adjust as needed
    ),
    "gemini-exp-1206": ModelInfo(
        name="gemini-exp-1206",
        provider=ModelProvider.GEMINI,
        description="Gemini (Quality improvements)",
        max_tokens=8192, # Adjust as needed
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