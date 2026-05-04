# Sense and Sensitivity: `Reasoning' Models are More Robust, but can Diverge from Human Consensus in a Legal Interpretation Task

Note: The human data, model results, and analysis are available in a separate location here: https://georgetown.box.com/v/sense-and-sensitivity-analysis

The repository contains data for prompt construction, local model inference (with vllm), and openai platform for the paper.
The code is adapted from https://github.com/bwaldon/llms-legal-interp for the paper https://arxiv.org/abs/2510.25356.

- `data` contains prompt source and generation data
- `prompts.py` contains code for generating the prompts from the vague contracts items.
- `gpt_*.py` scripts are a sequence of steps for running the experiments using OpenAI batches.
-  `job.sh` and `main.py` are the entrypoints for running the local model inferences using [vllm](https://docs.vllm.ai/en/stable/)


For an queries about the code reach out to the authors.
