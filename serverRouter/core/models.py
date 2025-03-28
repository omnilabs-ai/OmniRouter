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
            "MATH": 0.529,
            "BFCL": 0.883,
            "MGSM": 0.86
        },
        tokenCost=60.0,  
        latency=2.17  # ~60 tokens/sec, with 0.6-0.7s initial latency
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
        tokenCost=0.60,  # Average of input ($0.0015) and output ($0.002) costs
        latency=0.080  # ~70 tokens/sec, with <0.5s initial latency
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        description=(
            "OpenAI's GPT-4o variant (mini) for lightweight generation. GPT-4o mini enables a broad range of tasks with its low cost and latency. "
            "Strengths: It is over 60% cheaper than GPT-3.5 Turbo.  Despite its smaller size, GPT-4o Mini exhibits sophisticated language understanding and generation capabilities, ensuring high-quality performance across different tasks "
            "Weaknesses: Sometimes it prioritizes speed over thoughtfulness, giving quick but often shallow responses."
        ),
        max_tokens=128000,
        benchmarks={
            "MMLU": 0.820,
            "GPQA": 0.402,
            "HumanEval": 0.872,
            "MATH": 0.702,
            "BFCL": 0.641,
            "MGSM": 0.870
        },
        tokenCost=0.60, 
        latency=0.43  # Estimated >100 tokens/sec
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        description=(
            "GPT-4o strikes a balance between quality and efficiency, offering moderate benchmark performance. "
            "Strengths: GPT-4o demonstrates improved ability to understand complex language nuances and context, leading to more accurate and relevant responses. GPT-4o performs well in multiple languages, facilitating communication across diverse language barriers. "
            "Weaknesses: It may struggle with complex logical problems or tasks requiring deep analytical thinking. The amount of information GPT-4o can consider in a single prompt is restricted, potentially impacting its ability to understand long or complex contexts."
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
        tokenCost=10,  
        latency=0.41  
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Claude 3 Opus is designed for extremely complex tasks with an extended context. "
            "Strengths: High knowledge, advanced reasoning, and excellent multilingual performance. Excels at solving complex problems, performing multi-step calculations, and providing logical reasoning in responses. Can process large amounts of information and maintain context throughout long conversations. Performs well on tasks like generating detailed research reports, analyzing complex data, and creating high-quality content. "
            "Weaknesses: Compared to other models like Claude Sonnet, Opus can be more expensive to use due to its higher processing power. May require more carefully crafted prompts to achieve optimal results, especially for nuanced tasks."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.868,
            "GPQA": 0.504,
            "HumanEval": 0.849,
            "MATH": 0.601,
            "BFCL": 0.884,
            "MGSM": 0.907
        },
        tokenCost=75,  
        latency=1.29 
    ),
    "claude-3-7-sonnet": ModelInfo(
        name="claude-3-7-sonnet-20250219",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Claude 3.7 Sonnet is Anthropic's most intelligent model, with toggleable extended thinking. "
            "Strengths: Highest level of intelligence and capability, exceptional reasoning for complex tasks, "
            "top-tier performance across benchmarks including reasoning, coding, multilingual tasks, and image processing. "
            "Excellent for engaging responses and detailed content creation. Fast response times despite sophisticated processing. "
            "Weaknesses: May provide more detailed responses than needed for simple queries unless prompted for conciseness."
        ),
        max_tokens=8192,
        benchmarks={
            "MMLU": 0.800,
            "GPQA": 0.660,
            "HumanEval": 0.950,
            "MATH": 0.850
        },
        tokenCost=15.0,  # Output cost per million tokens ($15.00), input is $3.00 per MTok
        latency=0.80  # Estimated based on "Fast" classification
    ),
    "claude-3-7-sonnet-extended-thinking": ModelInfo(
        name="claude-3-7-sonnet-20250219",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Claude 3.7 Sonnet with extended thinking enabled for in-depth reasoning. "
            "Strengths: Self-reflective reasoning before answering, superior performance on math, physics, "
            "instruction-following, and coding tasks. Can have thinking budget controlled up to 128K tokens, allowing "
            "trade-offs between speed, cost, and quality. Supports up to 128K output tokens (15x more than other Claude models). "
            "Weaknesses: Longer response times due to additional reasoning process. Higher token usage and cost compared to "
            "standard mode. Not compatible with temperature and top_p modifications."
        ),
        max_tokens=64000,
        benchmarks={
            "MMLU": 0.840,
            "GPQA": 0.770,
            "HumanEval": 0.980,
            "MATH": 0.950
        },
        tokenCost=15.0,  # All extended thinking tokens billed as output tokens ($15.00 per MTok)
        latency=1.60,  # Higher latency due to extended reasoning process
        extended_thinking=True,  # Flag to enable extended thinking mode
        thinking_threshold=0.5,  # Default thinking threshold value
        thinking_budget=20000   # Default thinking budget (in tokens)
    ),
    "claude-3-5-haiku": ModelInfo(
        name="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Claude 3.5 Haiku is Anthropic's fastest model, optimized for speed and efficiency. "
            "Strengths: Intelligence at blazing speeds, quick and accurate targeted performance, ideal for applications "
            "requiring rapid responses. Excellent multilingual support and vision capabilities. "
            "Weaknesses: Less complex reasoning compared to Sonnet and Opus models, shorter maximum output."
        ),
        max_tokens=8192,
        benchmarks={
            "MMLU": 0.821,
            "GPQA": 0.490,
            "HumanEval": 0.792,
            "MATH": 0.650,
            "BFCL": 0.780,
            "MGSM": 0.830
        },
        tokenCost=4.00,  # Output cost per million tokens ($4.00), input is $0.80 per MTok
        latency=0.40  # Lowest latency among Claude models ("Fastest" classification)
    ),
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Anthropic's top performer for complex Q&A and multilingual tasks. "
            "Strengths: Research-grade answers, non-English queries, and precise tool usage. Claude 3.5 Sonnet stands out with its ability to write, edit, and execute code effectively, making it a valuable tool for fixing bugs, migrating codebases, and handling complex coding problems. Compared to previous Claude models, Sonnet operates significantly faster, which is beneficial for tasks requiring quick turnaround times like customer support or dynamic content generation. "
            "Weaknesses: While proficient in many tasks, Claude 3.5 Sonnet may not be as adept at deep analytical tasks requiring a high level of nuance and complexity compared to older models like Claude 3 Opus. Compared to other AI models, Claude might lack advanced features in areas like image recognition or scientific research depending on the specific application."
        ),
        max_tokens=8192,
        benchmarks={
            "MMLU": 0.887,
            "GPQA": 0.594,
            "HumanEval": 0.920,
            "MATH": 0.711,
            "BFCL": 0.902,
            "MGSM": 0.916
        },
        tokenCost=15.0,  
        latency=1.22  
    ),
    "deepseek-v3": ModelInfo(
        name="deepseek-ai/DeepSeek-V3",
        provider=ModelProvider.TOGETHER,
        description=(
            "DeepSeek Chat is an open‑source conversational model known for cost‑effective, high‐quality dialogue. "
            "Strengths: Excellent general conversation and Q&A. Likely better at general coding assistance (e.g., full-stack development, modern frameworks like React or TensorFlow) and explaining concepts in simpler terms. "
            "Weaknesses: May lag behind in complex coding or reasoning tasks. May sacrifice some niche expertise for versatility."
        ),
        max_tokens=64000,
        benchmarks={
            "MMLU": 0.880,
            "GPQA": 0.590,
            "HumanEval": 0.830,
            "MATH": 0.900,
            "MGSM": 0.850
        },
        tokenCost=1.1,  
        latency=7.31 
    ),
    "deepseek-r1": ModelInfo(
        name="deepseek-ai/DeepSeek-R1",
        provider=ModelProvider.TOGETHER,
        description=(
            "DeepSeek Reasoner (R1) is specialized for in-depth logical reasoning and multi-step problem solving. "
            "Strengths: Exceptional reasoning and math problem-solving; competitive with top-tier models on complex tasks. Might handle niche technical tasks (e.g., debugging legacy code, solving complex math problems) with higher precision if trained on domain-specific data. "
            "Weaknesses: May be slower and less polished in casual conversation. Struggles with broader context (e.g., explaining solutions to non-experts) or integrating real-time knowledge (e.g., latest APIs/frameworks)."
        ),
        max_tokens=64000,
        benchmarks={
            "MMLU": 0.908,
            "GPQA": 0.715,
            "HumanEval": 0.830,
            "MATH": 0.840,
            "MGSM": 0.900
        },
        tokenCost=2.19,  # Output Token per 1M tokens
        latency=60.68  # Estimated based on additional reasoning overhead (seconds)
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
            "GPQA": 0.515,
            "HumanEval": 0.900,
            "MATH": 0.868
        },
        tokenCost=0.30,  
        latency=0.21 
    ),
    "gemini-2.0-pro": ModelInfo(
        name="gemini-2.0-pro-exp-02-05",
        provider=ModelProvider.GEMINI,
        description=(
            "Google's most advanced model for technical and scientific tasks. It has the strongest coding performance and ability to handle complex prompts, with better understanding and reasoning of world knowledge, than any model Google has released so far. It comes with Google's largest context window at 2 million tokens, which enables it to comprehensively analyze and understand vast amounts of information, as well as the ability to call tools like Google Search and code execution. "
            "Strengths: STEM subjects, code generation, and multimodal reasoning. "
            "Weaknesses: Less conversational."
        ),
        max_tokens=32768,
        benchmarks={
            "MMLU": 0.800,
            "GPQA": 0.647,
            "HumanEval": 0.929,
            "MATH": 0.918
        },
        tokenCost=0.40,  # Output Token Cost (per 1M tokens)
        latency=0.56  # TTFT
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
        tokenCost=0.00,  # Using Gemini 1.5 Pro pricing as reference
        latency=0.66  # Estimated based on large context and multimodal processing
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelInfo(
        name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        description=(
            "Meta Llama 3.1 8B Instruct Turbo from Together AI, The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.730,
            "GPQA": 0.318,
            "HumanEval": 0.726,
            "MATH": 0.519,
            "BFCL": 0.654,
            "MGSM": 0.689
        },
        tokenCost=0.61,
        latency=2.32
    ),
    "Qwen/Qwen2-VL-72B-Instruct": ModelInfo(
        name="Qwen/Qwen2-VL-72B-Instruct",
        provider=ModelProvider.TOGETHER,
        description=(
            "Qwen2-VL (72B) Instruct from Together AI. "
            "Strengths: Exhibits excellent ability to interpret and understand complex visual scenes, including object recognition, scene classification, and visual question answering. Its architecture effectively integrates visual and textual information for enhanced multimodal processing. "
            "Weaknesses: May struggle with intricate, multi-step instructions. "
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.842,
            "GPQA": 0.379,
            "HumanEval": 0.646,
            "MATH": 0.511,
            "BFCL": 0.931,
            "MGSM": 0.760
        },
        tokenCost=0.40,
        latency=1.04
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": ModelInfo(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        provider=ModelProvider.TOGETHER,
        description=(
            "Mistral (7B) Instruct v0.2 from Together AI. "
            "Strengths: The primary strength of Mistral-7B v0.2 is unmatched performance at the 7B scale. It leverages clever training and data quality to punch far above its weight. This means users get fairly high accuracy and fluency from a model that can run on a single GPU or even on CPU in some cases. It's fast and memory-light, ideal for applications where a 70B model is impractical. Mistral7B is especially strong in general reasoning and knowledge. It's also surprisingly good at coding (nearly matching older 13B code-focused models) and at math word problems (thanks to training on chain-of-thought solutions). "
            "Weaknesses: As a 7B model, Mistral inevitably has limitations in complexity and knowledge depth. It may falter on extremely specialized or nuanced expert questions (GPQA is an example where 7B likely scores far lower than larger models). Likewise, its coding ability, though good, is not at the level of code-specialized models like CodeLlama-34B or larger GPTs – for very complex coding tasks, it will make mistakes or require multiple attempts. Another weakness is consistency in long outputs."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.689,
            "GPQA": 0.300,
            "HumanEval": 0.375,
            "MATH": 0.340
        },
        tokenCost=0.25,
        latency=0.32
    ),
    "microsoft/WizardLM-2-8x22B": ModelInfo(
        name="microsoft/WizardLM-2-8x22B",
        provider=ModelProvider.TOGETHER,
        description=(
            "WizardLM-2 (8x22B) from Together AI. "
            "Strengths: WizardLM-2 8x22B strengths are its excellent reasoning and conversational abilities. It was specifically tuned to handle complex instructions and dialogues (Wizard refers to mastering tricky queries), and it shows - the model can follow convoluted user requests, ask clarifying questions, and produce detailed, correct answers for multi-step problems. Another strength is coding: WizardLM-2 inherited Mistral coding skill and enhanced it via finetuning - it often produces correct, well-commented code for difficult challenges, making it useful as a coding assistant. "
            "Weaknesses: One notable weakness of WizardLM-2 8x22B is safety/alignment issues that arose. This suggests that while it is very capable, its filtering of harmful or disallowed content might not have been thoroughly refined - it may produce problematic content if prompted incautiously. Thus, in user-facing deployments, it would need an additional safety layer or further fine-tuning to mitigate this. In terms of abilities, the evaluation oddity on GPQA (only 17.6% in a 0-shot test) indicates a weakness in answering highly specialized science questions without sufficient prompting or few-shot examples. This could mean WizardLM-2 sometimes does not apply chain-of-thought unless coaxed, or that its knowledge, while broad, might miss some niche academic details that models like DeepSeek targeted."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.820,
            "GPQA": 0.176,
            "HumanEval": 0.817,
            "MATH": 0.223
        },
        tokenCost=0.50,
        latency=0.120 
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
        )
    ),
    "dall-e-2": ModelInfo(
        name="dall-e-2",
        provider=ModelProvider.OPENAI,
        description=(
            "OpenAI's efficient image generation model. "
            "Strengths: Fast generation and good quality for common use cases. "
            "Weaknesses: Less detailed and accurate compared to DALL-E 3."
        )
    )
}

# Combined models dictionary
MODELS = {**CHAT_MODELS, **IMAGE_MODELS} 