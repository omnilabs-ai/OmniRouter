## Models

The router selects from the following models:

- **Flux**: Cheapest option, best for cost-sensitive applications
- **Turbo**: Mid-range option, good balance of quality and cost
- **DALL-E 3**: High-quality creative images, more expensive
- **Imagen**: Specialized in photorealistic images

## Parameters

- `prompt` (str): The image generation prompt
- `rel_cost` (float): Relative importance of cost (0.0 to 1.0)
  - 0.0 = quality is everything, cost doesn't matter
  - 1.0 = cost is everything, use cheapest model

## How the algorithm works

The model selection algorithm follows these decision rules:

1. **Cost-driven selection**: If cost efficiency is the highest priority (`rel_cost >= 0.9`), the router selects Flux, the most economical option.

2. **Content-based selection**:
   - **Realistic imagery**: If the prompt contains terms like "realistic," "photorealistic," "photograph," etc., the router selects Imagen, which excels at photorealistic results.
   
   - **Creative content**: If the prompt contains words like "creative," "artistic," "imaginative," or "fantasy":
     - When cost is less important (`rel_cost < 0.5`): The router selects DALL-E 3 for premium creative quality
     - When cost is somewhat important (`rel_cost >= 0.5`): The router selects Turbo for a good balance of creativity and cost

3. **Default selection**: If no specific keywords are detected and cost is not a priority, the router defaults to Imagen as a well-rounded choice.

The router analyzes both prompt semantics and cost preferences to make intelligent routing decisions that optimize for both quality and budget considerations.

