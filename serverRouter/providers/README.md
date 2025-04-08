# OmniRouter Providers

This directory contains implementations for various AI model providers integrated into the OmniRouter system. Each provider implements the standardized interfaces defined in `serverRouter/core/interfaces.py`.

## Available Providers and Models

### Text Models

| Model ID | Provider | Type | Pricing (per 1M tokens) | Latency (s) | Description |
|----------|----------|------|------------------------|------------|-------------|
| gpt-4 | OpenAI | Text | $60.00 | 2.17 | OpenAI's flagship model excelling in complex reasoning and coding. Strong in technical explanations, API integrations, and multi-step problem solving. |
| gpt-3.5-turbo | OpenAI | Text | $0.60 | 0.08 | Versatile conversational model for general Q&A, basic reasoning, text completion, and casual dialogue. Fast and cost-efficient for everyday tasks. |
| gpt-4o-mini | OpenAI | Text | $0.60 | 0.43 | Lightweight variant of GPT-4o with low cost and latency. Offers sophisticated language understanding while being 60% cheaper than GPT-3.5 Turbo. |
| gpt-4o | OpenAI | Text | $10.00 | 0.41 | Balance between quality and efficiency with improved ability to understand complex language nuances and context. Performs well in multiple languages. |
| claude-3-opus | Anthropic | Text | $75.00 | 1.29 | Designed for extremely complex tasks with excellent multilingual performance, advanced reasoning, and ability to process large amounts of information. |
| claude-3-7-sonnet | Anthropic | Text | $15.00 | 0.80 | Anthropic's most intelligent model with exceptional reasoning for complex tasks, top-tier performance on benchmarks, and fast response times. |
| claude-3-7-sonnet-extended-thinking | Anthropic | Text | $15.00 | 1.60 | Enabled with self-reflective reasoning for superior performance on math, physics, and coding tasks. Supports up to 128K output tokens. |
| claude-3-5-haiku | Anthropic | Text | $4.00 | 0.40 | Anthropic's fastest model optimized for speed and efficiency. Ideal for applications requiring rapid responses with excellent multilingual support. |
| claude-3-5-sonnet | Anthropic | Text | $15.00 | 1.22 | Top performer for complex Q&A and multilingual tasks. Excels at research-grade answers, non-English queries, and precise tool usage. |
| deepseek-v3 | Together | Text | $1.10 | 7.31 | Open-source conversational model for cost-effective, high-quality dialogue. Strong on general conversation, Q&A, and coding assistance. |
| deepseek-r1 | Together | Text | $2.19 | 60.68 | Specialized for in-depth logical reasoning and multi-step problem solving with exceptional math problem-solving capabilities. |
| gemini-2.0-flash-lite | Gemini | Text | $0.30 | 0.21 | Lightweight, high-speed variant optimized for low latency with fast response times for general queries. |
| gemini-2.0-pro | Gemini | Text | $0.40 | 0.56 | Google's advanced model for technical and scientific tasks with strong coding performance and 2M token context window. Excels in STEM subjects. |
| learnlm-1.5-pro-experimental | Gemini | Multimodal | $0.00 | 0.66 | Versatile multimodal model supporting audio, image, video processing and natural language understanding. |
| meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo | Together | Text | $0.61 | 2.32 | Optimized for multilingual dialogue with strong performance on industry benchmarks. |
| Qwen/Qwen2-VL-72B-Instruct | Together | Multimodal | $0.40 | 1.04 | Excellent at interpreting complex visual scenes, including object recognition and visual question answering. |
| mistralai/Mistral-7B-Instruct-v0.2 | Together | Text | $0.25 | 0.32 | High performance at 7B scale with good general reasoning, knowledge, coding, and math capabilities. Fast and memory-efficient. |
| microsoft/WizardLM-2-8x22B | Together | Text | $0.50 | 0.12 | Strong reasoning and conversational abilities, specifically tuned for complex instructions and dialogues. Good for coding tasks. |
| grok-2-1212 | xAI | Text | $10.00 | 0.35 | xAI's flagship text model with excellent performance in reasoning, coding, and mathematical understanding. Comparable to top-tier models. |
| grok-2-vision-1212 | xAI | Multimodal | $10.00 | 0.40 | Processes visual information including documents, diagrams, and charts with exceptional performance in vision-based tasks. |

### Image Models

| Model ID | Provider | Type | Pricing | Latency (s) | Description |
|----------|----------|------|---------|------------|-------------|
| dall-e-3 | OpenAI | Image | Varies | Varies | OpenAI's most advanced image generation model with highly detailed and photorealistic images and excellent prompt following. |
| dall-e-2 | OpenAI | Image | Varies | Varies | OpenAI's efficient image generation model offering fast generation and good quality for common use cases. |
| grok-2-image-1212 | xAI | Image | $0.07 per image | 1.50 | xAI's image generation model supporting multiple images per request (up to 10). |
| deepai-standard | DeepAI | Image | $0.05 per image | 1.30 | Standard text-to-image model with flexible image creation for general use cases. |
| deepai-hd | DeepAI | Image | $0.05 per image | 1.60 | High-definition text-to-image model creating more detailed images than the standard version. |
| deepai-genius | DeepAI | Image | $0.05 per image | 1.90 | Advanced text-to-image model optimized for cinematic style aesthetic. |
| deepai-genius-anime | DeepAI | Image | $0.05 per image | 1.90 | Specialized for creating high-quality anime-style illustrations from text prompts. |
| deepai-genius-photography | DeepAI | Image | $0.05 per image | 1.90 | Optimized for photographic realism with quality photographic lighting and details. |
| deepai-genius-graphic | DeepAI | Image | $0.05 per image | 1.90 | Designed for graphic art and design aesthetics, suitable for logos and digital art. |

## Provider Details

### OpenAI
- **API Documentation**: [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
- **Pricing**: [OpenAI Pricing](https://openai.com/pricing)
- **Supports**: Text completion, chat, image generation
- **Environment Variable**: `OPENAI_API_KEY`

### Anthropic
- **API Documentation**: [Anthropic API Docs](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- **Pricing**: [Anthropic Pricing](https://www.anthropic.com/pricing)
- **Supports**: Text completion, chat
- **Environment Variable**: `ANTHROPIC_API_KEY`

### Gemini (Google)
- **API Documentation**: [Gemini API Docs](https://ai.google.dev/docs)
- **Pricing**: [Gemini Pricing](https://ai.google.dev/pricing)
- **Supports**: Text completion, chat, multimodal
- **Environment Variable**: `GEMINI_API_KEY`

### Together AI
- **API Documentation**: [Together AI Docs](https://docs.together.ai/docs/inference-overview)
- **Pricing**: [Together AI Pricing](https://www.together.ai/pricing)
- **Supports**: Text completion, chat, various open source models
- **Environment Variable**: `TOGETHER_API_KEY`

### Stable Diffusion
- **API Documentation**: [Stability AI Docs](https://platform.stability.ai/docs/api-reference)
- **Pricing**: [Stability AI Pricing](https://stability.ai/pricing)
- **Supports**: Image generation
- **Environment Variable**: `STABILITY_API_KEY`

### xAI
- **API Documentation**: [xAI (Grok) Docs](https://platform.xai.org/docs)
- **Pricing**: $10 per million output tokens, $2 per million input tokens
- **Supports**: Text completion, chat, image generation
- **Environment Variable**: `XAI_API_KEY`

### DeepAI
- **API Documentation**: [DeepAI Docs](https://deepai.org/apis)
- **Pricing**: $5 per 100 API calls, or $5 per 500 for DeepAI Pro subscribers
- **Supports**: Image generation
- **Environment Variable**: `DEEP_AI_API_KEY`

## Implementing a New Provider

To add a new provider to OmniRouter:

1. Create a new folder in `serverRouter/providers/` for your provider
2. Implement the provider class by extending the appropriate interfaces (`ChatProvider`, `ImageProvider`)
3. Add the provider to the `ModelProvider` enum in `serverRouter/core/datamodels.py`
4. Register your provider in `serverRouter/router.py`
5. Add model information to `serverRouter/core/models.py`
6. Update environment variable configuration in `.env` 