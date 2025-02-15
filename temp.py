from serverRouter.smartRouter.SmartRouter import SmartRouter

router = SmartRouter(verbose=False)
query = "Build a python function that takes a list of numbers and returns the sum of the numbers"

result = router.get_top_user_models(
    query="Solve this complex math problem...",
    rel_cost=0.3,
    rel_accuracy=0.7
)

print(result)

print(f"Selected model: {result['model']}")
print("\nDecision process:")
print(result['explanation'])