## Overview
This repository contains the code, datasets, and analysis for our study on the accessibility and performance of open, open-source, and closed large language models (LLMs) in the context of Automated Essay Scoring (AES). Our work systematically compares the performance, fairness, and cost-effectiveness of various LLMs for both essay assessment and generation.

## Repository Structure
📂 Data/              # Contains human-written and LLM-generated essays
📂 Scripts/           # Python scripts for essay generation, scoring, and analysis
📜 .DS_Store         # macOS system file (can be ignored)
📜 .gitattributes    # Git configuration file
📜 README.md         # This file
📜 olmo.script       # Script for running OLMo locally
📜 poetry.lock       # Dependency lock file for reproducibility
📜 pyproject.toml    # Configuration for managing dependencies with Poetry

## Data
We use two human-generated essay datasets:

#### ASAP (Automated Student Assessment Prize)

12,979 essays across 8 prompts
Scoring ranges from 1-6, 0-4, 0-30, 0-60
Includes argumentative, response, and narrative essays
FCE (Cambridge Learner Corpus - First Certificate in English)

2,466 essays spanning 5 genres
Scoring based on English proficiency
Includes letter, commentary, suggestion essays
Both datasets contain prompts, rubrics, and ground-truth human scores, used for LLM assessment.

Model Selection
Closed-source Models
GPT-4
GPT-4 Omni
GPT-3.5
Open Models (Weights Available)
LLaMa 2 (70B)
LLaMa 3 (70B)
LLaMa 3.1 (405B)
DeepSeek-R1 (671B)
Qwen2.5 (72B)
Open-source Models (Weights + Training Data Available)
OLMo 2 (13B)

How Essays Were Generated
We prompted each LLM with 150 unique essay prompts covering six writing types from ASAP and FCE.

Zero-shot setting: LLMs received only the prompt (no examples).
Few-shot setting: LLMs received the prompt + three human-scored essays as references.
Each model generated ~1,537 essays (GPT-4 = 1,486, GPT-4 Omni = 1,527).
APIs Used:

GPT models: OpenAI API
LLaMa models: Replicate API & LLaMa API
Qwen2.5 & DeepSeek: DeepInfra API
OLMo: Run locally using olmo.script

How Essays Were Scored
We used the same LLMs for automated scoring of human and LLM-generated essays.

Zero-shot assessment: LLMs received the rubric and prompt but no examples.
Few-shot assessment: LLMs received three human-scored essays as references.
Evaluation Metrics
We measured LLM vs. Human score agreement using:

Error Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE)
Agreement Metrics: Quadratic Weighted Kappa (QWK), Pearson Correlation Coefficient (PCC), Spearman’s Rank Correlation (SRC)
Fairness Analysis
We used ANOVA models to analyze biases in age (young/old) and race (Asian/non-Asian) from the FCE dataset.
Cost Analysis
Token costs for inference were calculated using API pricing.
Open LLMs were up to 37x cheaper than GPT-4.

Setup & Installation
This project uses Poetry for dependency management.

1. Download pipx: https://pipx.pypa.io/stable/installation/
2. Install poetry: https://python-poetry.org/docs/#installing-with-pipx
3. To generate the cost figures:

```{python}
poetry run python Scripts/CostComparison.py
```

4. To generate the score and delta score figures

```{python}
poetry run python Interactionplots.py
```
5. To run essay generation
```{python}
poetry run python Scripts/Qwen2.5Generation.py. #replace with any of the generation files
```
6. To run essay scoring
```{python}
poetry run python Scripts/Llama3.1Scoring.py. #replace with any of the scoring files
```
Citing This Work
@article{your_reference,
  author = {Anonymous},
  title = {Bridging the LLM Accessibility Divide: How Open-source Models Compare in Terms of Performance, Fairness, and Cost},
  year = {2025},
  journal = {arXiv}
}

Contributors
Kezia Oketch
John Lalor
Yi Yang
Ahmed Abbasi

Acknowledgments
This research is supported by the University of Notre Dame's Human-centered Analytics Lab (HAL).
