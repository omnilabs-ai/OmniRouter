# ***OmniRouter***

# For Developers

## Contributing

Do not make any direct changes to the **dev** or **main** branchs

Create a branch off of **dev** ```git checkout -b username-feature```

Push your new branch to the github: ```git push -u origin username-feature```

After finishing changes place a pull request into dev with a description of the task

## Building

Download all dependancies into your virtual environment by ```pip install -r requirements.txt```

If you install new packages make sure to update the package manager with ```pip freeze > requirements.txt```

## Testing

**Run Server**: ```python -m testLib.server```

**Run Chat/Image Client**: 
- ```python -m testLib.chat_client```
- ```python -m testLib.image_client```




# For Users

**Slogan:** One Key, One API, Hundreds of Models

**Description:**
A unified API interface for all modern LLMs, enabling seamless model switching and performance optimization.

**Benefits:**

- Single payment for multiple models
- Simplified model switching
- Optimized performance and cost-efficiency

**Core Features (Implementation Order):**

1. **Unified API Interface:**
    - Standardized API for all models
2. **Basic Documentation:**
    - Thorough, user-friendly guides and references
3. **Dynamic Routing:**
    - Route queries to the best or most cost-efficient model
    - Low-latency optimization
4. **Customizable Routing Rules:**
    - User-defined criteria for model selection (e.g., speed vs. accuracy)
5. **Chat Interface:**
    - GPT-like interface for model selection during queries


