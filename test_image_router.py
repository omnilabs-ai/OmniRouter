from ImageRouting.ImageRouter import ImageRouter

def test_image_router():
    router = ImageRouter()
    
    # Test cases
    test_prompts = [
        # should return imagen as it is default model
        ("A beautiful sunset over the mountains", 0.5), 

        # should return dalle3 as it is creative and cost is not a priority
        ("A creative and artistic interpretation of a dream about soccer", 0.3),

        # should return imagen as it is realistic
        ("A realistic portrait of a woman with blue eyes", 0.7),

        # should return flux as it is cost efficient
        ("A futuristic city with flying cars", 0.95),

        # should return turbo as it is creative and cost is not a priority
        ("An imaginative fantasy world with dragons", 0.6),

        # should return flux as cost takes precidence even though user wants creative
        ("An imaginative fantasy world with dragons", 0.95)
    ]
    
    for prompt, rel_cost in test_prompts:
        result = router.get_best_image_model(prompt, rel_cost)
        print(f"\nPrompt: {prompt}")
        print(f"Rel Cost: {rel_cost}")
        print(f"Selected Model: {result['model']}")
        print("\nDetailed Explanation:")
        print(result['explanation'])
        print("-" * 80)

if __name__ == "__main__":
    test_image_router() 