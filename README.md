# Sense and Sensitivity: `Reasoning' Models are More Robust, but can Diverge from Human Consensus in a Legal Interpretation Task

Note: The human data, model results, and analysis are available in a separate location here: https://georgetown.box.com/v/sense-and-sensitivity-analysis

The repository contains data for prompt construction, local model inference (with vllm), and openai platform for the paper.
The code is adapted from https://github.com/bwaldon/llms-legal-interp.

Python package requirements are maintained separately for the local model inference and openai platform. See `requirements.txt` and `requirements-openai.txt`. 

- `data` contains prompt source and generation data
- `prompts.py` contains code for generating the prompts from the vague contracts items.
- `gpt_*.py` scripts are a sequence of steps for running the experiments using OpenAI batches.
  - Note that the current implmentation does not explicitly match the id across the batch and output.
  - For our runs used `check_custom_id_mismatch.py` to check for any mismatches in the id across the batch and output.
-  `job.sh` and `main.py` are the entrypoints for running the local model inferences using [vllm](https://docs.vllm.ai/en/stable/)


Note: We haven't been able to fully test this stripped down version of the code. 

For any queries about the code reach out to the authors.
