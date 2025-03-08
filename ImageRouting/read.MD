
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