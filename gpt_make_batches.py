"""
Ref: https://developers.openai.com/api/docs/guides/batch/
Ref: gpt.py
"""

import logging
import os

from functools import reduce
from collections import OrderedDict

import argparse
from pathlib import Path
import json
import numpy as np

import pandas as pd
from datasets import Dataset
from openai import OpenAI

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Martinez and other models
"""Martinez had 'Models included 7 flagship models from OpenAI (four large language models, and three “advanced reasoning models”), 
including GPT-3.5,
GPT-4, - Seems to be priced relatively very high (will check after token budgeting) 
GPT-4o - 
GPT-4.1, o3, o1-mini, and o1'
Based the docs: 
To keep parity between pre GPT-5.1 and post GPT-5.1
ReasoningEffort medium
And a larger max_output_tokens to allow for 'reasoning' budget
Since max_new_tokens is the controlling parameter - 
we can set it to some round tokens, leaving budget for answer tokens. 
"""
# FIXME deduplicate all this with gpt.py
model_list = [
]

# 4.1 mini is a test model
model_list = [
    "gpt-5-mini-2025-08-07",
    "o3-mini-2025-01-31",
    "gpt-4.1-mini-2025-04-14",
]

def get_config(model):
    config = {
        "max_output_tokens": 16,
        "top_logprobs": 20,
        "temperature": None,
        "reasoning": {
            "effort": None
        }
    }
    if "gpt-4.1" in model:
        config["temperature"] = 0

    else:
        config["reasoning"]["effort"] = "medium"
        config["max_output_tokens"] = 128 + 2048 # For o3 "medium" 1024 was not enough
        config["top_logprobs"] = None

    return config


def prompt_request(prompt, index, model):
    config = get_config(model)
    return {
        "custom_id": f"{model}-{index}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": prompt,
            "max_output_tokens": config["max_output_tokens"],
            "top_logprobs": config["top_logprobs"],
            "temperature": config["temperature"],
            "reasoning": config["reasoning"],
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test", default=False, action="store_true")


    # Use static prompts dataset from the newer version
    prompts_dataset = Dataset.from_csv("data/prompts/coverage_contracts2.csv")
    args = parser.parse_args()


    if args.test:
        logger.info("Running in test mode, using only a subset of the prompts dataset")
        # prompts that have some 0 values
        # samples = [36, 117, 753, 969, 971, 972, 974, 975, 977, 978]
        prompts_dataset = prompts_dataset[:10]

    prompts_list = prompts_dataset['prompt']
    logger.info(f"Creating batch requests for {len(prompts_list)} prompts and {len(model_list)} models")
    for model in model_list:
        config = get_config(model)
        # We want to make a batch request for all the prompts for the model
        logger.info(f"Creating {len(prompts_list)} batch requests for {model} model")
        model_batch = [prompt_request(prompt, index, model) for index, prompt in enumerate(prompts_list)]
        output_path = args.out / f"{model}-batch.jsonl"
        logger.info(f"Saving request batch for {model} to {output_path}")
        # Save batch as jsonl
        with open(output_path, "w") as f:
            for request in model_batch:
                json.dump(request, f)
                f.write("\n")