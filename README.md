# Overview

This repository contains the code, datasets, and analysis for our study on the accessibility and performance of open, open-source, and closed large language models (LLMs) in the context of Automated Essay Scoring (AES). Our work systematically compares the performance, fairness, and cost-effectiveness of various LLMs for both essay assessment and generation.

# Repository Structure

```
📂 Data/             # Contains human-written and LLM-generated essays
📂 Scripts/          # Python scripts for essay generation, scoring, and analysis
📜 .gitattributes    # Git configuration file
📜 README.md         # This file
📜 olmo.script       # Script for running OLMo locally
📜 poetry.lock       # Dependency lock file for reproducibility
📜 pyproject.toml    # Configuration for managing dependencies with Poetry
📜 requirements.txt  # Contains a list of dependencies required to run the LLM-based scripts
```

# Data
We use two benchmark datasets:

## Automated Student Assessment Prize (ASAP)

- 12,979 essays across 8 prompts
- Scoring ranges from 1-6, 0-4, 0-30, 0-60  
Includes argumentative, response, and narrative essays

## Cambridge Learner Corpus - First Certificate in English (FCE)

- 2,466 essays spanning 5 genres
- Includes letter, commentary, suggestion essays  
Both datasets contain prompts, rubrics, and ground-truth human scores, used for LLM assessment.

# Model Selection

Closed-source Models  
- GPT-4
- GPT-4 Omni
- GPT-3.5

Open Models (Weights Available)  
- Llama 2 (70B)  
- Llama 3 (70B)  
- Llama 3.1 (405B)  
- DeepSeek-R1 (671B)  
- Qwen2.5 (72B)
  
Open-source Models (Weights + Training Data Available) 
- OLMo 2 (13B)  

# How Essays Were Generated

We prompted each LLM with unique essay prompts covering six writing types from ASAP and FCE.

The generation was done under Zero-shot setting to remain consistent with the human generation: LLMs received only the prompt (no examples).  
Each model generated ~1,537 essays (GPT-4 = 1,486, GPT-4 Omni = 1,527).  

APIs Used:

GPT models: OpenAI API  
Llama models & DeepSeek-r1: Replicate API & Llama API  
Qwen2.5: DeepInfra API  
OLMo: Run locally using the following command:
```python
poetry run python Scripts/Olmo-2-13B-Scoring.py
```
## Note
**OLMo uses vLLM, which runs generation & scoring locally.**  
Be sure to use a machine with sufficient resources (GPU, memory, etc.).

## Setting Up API Key Environment Variables

To configure API keys, run the following commands in your terminal:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export DEEPINFRA_API_KEY="your_deepinfra_api_key_here"
export REPLICATE_API_TOKEN="your_replicate_api_token_here"
```
⚠️ Security Notice: Avoid hardcoding API keys in scripts. Consider using os.getenv() for better security.

# How Essays Were Scored  

We used the same LLMs for automated scoring of human and LLM-generated essays.

Zero-shot assessment: LLMs received the rubric and prompt but no examples.  
Few-shot setting: LLMs received the prompt + three human-scored essays as references.  

## Evaluation Metrics  

We measured LLM vs. Human score agreement using:

Error Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE)  
Agreement Metrics: Quadratic Weighted Kappa (QWK), Pearson Correlation Coefficient (PCC), Spearman’s Rank Correlation (SRC)  

## Fairness Analysis  

We used ANOVA models to analyze biases in age (young/old) and race (Asian/non-Asian) from the FCE dataset.  

## Cost Analysis  

Token costs for inference were calculated using API pricing.  
Open LLMs were up to 37x cheaper than GPT-4.  

# Setup & Installation

This project uses **Poetry** for dependency management, in addition to the dependencies listed in `requirements.txt`.

1. Download pipx: https://pipx.pypa.io/stable/installation/
2. Install poetry: https://python-poetry.org/docs/#installing-with-pipx
3. To generate the cost figures:

```{python}
poetry run python Scripts/CostComparison.py
```

4. To generate the score and delta score figures

```{python}
poetry run python Scripts/Interactionplots.py
```
5. To run essay generation

```{python}
poetry run python Scripts/UnifiedGeneration.py
```
6. To run essay scoring

```{python}
poetry run python Scripts/UnifiedScoring.py
```

# Citing This Work

```
TO BE ADDED!

```
# Contributors

Kezia Oketch  
John Lalor    
Yi Yang    
Ahmed Abbasi    

# Acknowledgments

This research is supported by the University of Notre Dame's Human-centered Analytics Lab (HAL).
