from typing import Dict, Union, Tuple
from dataclasses import dataclass

@dataclass
class ImageRouterDecision:
    """Class to store and format the router's decision-making process."""
    prompt_analysis: Dict[str, bool]
    model_costs: Dict[str, float]
    chosen_model: str
    reason: str

    def format_verbose_output(self) -> str:
        """Format the decision process into a detailed string."""
        output = []
        
        # Prompt analysis
        output.append("=== Prompt Analysis ===")
        for key, value in self.prompt_analysis.items():
            output.append(f"{key}: {value}")
            
        # Model costs
        output.append("\n=== Model Costs ===")
        for model, cost in self.model_costs.items():
            output.append(f"{model}: ${cost:.6f}")
            
        # Final decision
        output.append(f"\nChosen Model: {self.chosen_model}")
        output.append(f"Reason: {self.reason}")
        
        return "\n".join(output)

class ImageRouter:
    def __init__(self) -> None:
        """Initialize the ImageRouter with model costs."""
        self.verbose = True
        from serverRouter.core.models import IMAGE_MODELS
        self.model_costs = {}
        model_mapping = {
            "flux": "black-forest-labs/FLUX.1-schnell-Free",
            "dalle3": "dall-e-3",
            "turbo": "stable-diffusion-3.5-turbo",
            "imagen": "imagen-3.0-generate-002"
        }
        
        # Populate costs from the registry
        for internal_name, registry_name in model_mapping.items():
            if registry_name in IMAGE_MODELS:
                self.model_costs[internal_name] = IMAGE_MODELS[registry_name].tokenCost
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, bool]:
        """Analyze the prompt for specific keywords."""
        prompt_lower = prompt.lower()
        
        contains_creative = any(word in prompt_lower for word in ["creative", "artistic", "imaginative", "fantasy"])
        contains_realistic = any(word in prompt_lower for word in ["realistic", "photorealistic", "real", "photograph", "realistic"])
        
        return {
            "contains_creative": contains_creative,
            "contains_realistic": contains_realistic
        }
    
    def _select_model(self, prompt_analysis: Dict[str, bool], rel_cost: float) -> Tuple[str, str]:
        """Select the appropriate model based on prompt analysis and cost preference."""
        # If cost is the only consideration (rel_cost = 1.0), use flux
        if rel_cost >= 0.9:
            return "flux", "Cost efficiency is the highest priority (rel_cost >= 0.9), selecting the cheapest model."
        
        # If prompt contains "realistic", use imagen
        if prompt_analysis["contains_realistic"]:
            return "imagen", "Prompt contains words related to realism, selecting Imagen for photorealistic results."
        
        # If prompt contains "creative", use either dalle3 or turbo based on rel_cost
        if prompt_analysis["contains_creative"]:
            if rel_cost < 0.5:
                return "dalle3", "Prompt contains creative terms and cost is less important (rel_cost < 0.5), selecting DALL-E 3."
            else:
                return "turbo", "Prompt contains creative terms but cost is somewhat important (rel_cost >= 0.5), selecting Turbo."

        else:
            return "imagen", "Imagen is the default model we use when no specific keywords are detected/ cost is not a priority"
    
    def get_best_image_model(self, prompt: str, rel_cost: float = 0.5) -> Union[str, Dict]:
        """Get the best image model for the given prompt and cost preference.
        
        Args:
            prompt (str): The image generation prompt
            rel_cost (float): Relative importance of cost (0.0 to 1.0)
                              0.0 = quality is everything, cost doesn't matter
                              1.0 = cost is everything, use cheapest model
        
        Returns:
            If verbose=False: str - The name of the chosen model
            If verbose=True: dict - A dictionary containing:
                - 'model': The name of the chosen model
                - 'explanation': A detailed explanation of the decision process
        """
        # Analyze the prompt
        prompt_analysis = self._analyze_prompt(prompt)
        
        # Select the model
        chosen_model, reason = self._select_model(prompt_analysis, rel_cost)
        
        # Create decision object for verbose output
        decision = ImageRouterDecision(
            prompt_analysis=prompt_analysis,
            model_costs=self.model_costs,
            chosen_model=chosen_model,
            reason=reason
        )
        
        return {
            'model': chosen_model,
            'explanation': decision.format_verbose_output()
        } 