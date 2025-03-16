# OmniRouter

## Overview
OmniRouter is a project that aims to provide a unified API interface for all modern LLMs (Large Language Models). It enables seamless model switching and performance optimization, offering a convenient solution for developers and users alike.

## Contributing
When contributing to the project, please follow these guidelines:
- Do not make direct changes to the **dev** or **main** branches. Create a new branch for your feature development.
- Push your branch to the GitHub repository and submit a **pull request** to merge your changes into the **dev** branch.

## Building
To build the project, follow these steps:
1. Create a virtual environment using `python -m venv venv`.
2. Activate the virtual environment with `venv\Scripts\activate`.
3. Install all dependencies by running `pip install -r requirements.txt`.
4. If you add new packages, update the package manager with `pip freeze > requirements.txt`.

## Testing
To run the server and client components for testing:
- Run the server with `python -m testLib.server`.
- Run the chat client with `python -m testLib.chat_client`.
- Run the image client with `python -m testLib.image_client`.

## Codebase Structure
The codebase is organized into the following main sections:
- `clientLib`: Contains client-side libraries and utilities.
- `testLib`: Contains the testing framework to run when validating new builds
- `serverRouter`: Includes the core functionality of the OmniRouter server.
  - `core`: Defines data models and core functionalities.
    - `datamodels`: Contains the response and request objects to be sent from the client
    - `interfaces`: Contains a chat and image interface for providers to extend and implement.
    - `models`: Contains the list of image and chat models along with their information.
    - `exceptions`: Contains exceptions for the API
  - `providers`: Contains provider implementations for different LLM models.
  - `router.py`: Main FastAPI router for handling API requests. Entry point to the application

## Core Features
The project offers the following core features in its implementation order:
1. Unified API Interface: Standardized API for all models.
2. Basic Documentation: User-friendly guides and references.
3. Dynamic Routing: Route queries to the best or most cost-efficient model.
4. Customizable Routing Rules: User-defined criteria for model selection.
5. Chat Interface: GPT-like interface for model selection during queries.

## For Users
- **Slogan:** One Key, One API, Hundreds of Models
- **Description:** A unified API interface for all modern LLMs, enabling seamless model switching and performance optimization.
- **Benefits:**
  - Single payment for multiple models.
  - Simplified model switching.
  - Optimized performance and cost-efficiency.



## Others

- Create a clean pip requirements: `pip list --format=freeze > requirements.txt`