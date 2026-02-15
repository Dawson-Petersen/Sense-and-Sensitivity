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

batch_configs = [
    {
        "model_name": "gpt-5-mini-2025-08-07",
        "reasoning_effort": "minimal",
    },
    {
        "model_name": "gpt-5-mini-2025-08-07",
        "reasoning_effort" : "low",
    },
    {
        "model_name": "gpt-5-mini-2025-08-07",
        "reasoning_effort" : "medium",
    },
    {
        "model_name" : "gpt-5.2-2025-12-11",
        "reasoning_effort" : "none",
    },
    {
        "model_name" : "gpt-5.2-2025-12-11",
        "reasoning_effort" : "minimal",
    },
    {
        "model_name" : "gpt-5.2-2025-12-11",
        "reasoning_effort" : "low",
    },
    {
        "model_name" : "gpt-5.2-2025-12-11",
        "reasoning_effort" : "medium",
    },
]

def get_config(model_name, reasoning_effort):
    config = {
        "max_output_tokens": 16,
        "top_logprobs": 20,
        "temperature": None,
        "reasoning": {
            "effort": None
        }
    }
    if "gpt-4.1" in model_name:
        config["temperature"] = 0

    else:
        config["reasoning"]["effort"] = reasoning_effort
        config["max_output_tokens"] = 128 + 2048 # For o3 "medium" 1024 was not enough
        config["top_logprobs"] = None

    return config


def prompt_request(prompt, model, reasoning_effort, index):
    config = get_config(model, reasoning_effort)
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
    parser.add_argument("out", type=Path)
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
    logger.info(f"Creating batch requests for {len(prompts_list)} prompts and {len(batch_configs)} model configurations")
    batches = []
    for batch_config in batch_configs:
        model = batch_config["model_name"]
        reasoning_effort = batch_config["reasoning_effort"]
        # We want to make a batch request for all the prompts for the model
        logger.info(f"Creating {len(prompts_list)} batch requests for {model} model")
        logger.info(f"The model will use the config: {get_config(model, reasoning_effort)}")
        model_batch = [prompt_request(prompt, model, reasoning_effort, index) for index, prompt in enumerate(prompts_list)]
        output_path = args.out / f"{model}-{reasoning_effort}-batch.jsonl"
        logger.info(f"Saving request batch for {model} to {output_path}")
        # Save batch as jsonl
        with open(output_path, "w") as f:
            for request in model_batch:
                json.dump(request, f)
                f.write("\n")
        batches.append(
            {
                **batch_config,
                "batch_file": output_path,
            }
        )
    # Save batch configs to a csv file
    batch_configs_path = args.out / "batches.csv"
    logger.info(f"Saving batch configs to {batch_configs_path}")
    pd.DataFrame(batches).to_csv(batch_configs_path, index=False)