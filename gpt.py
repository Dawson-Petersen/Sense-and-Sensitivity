"""

OpenAI supports both single and batch requests.
Ref: https://github.com/Aatlantise/advarsarial-nli-amr/blob/main/open_ai_api.py
Ref: https://platform.openai.com/docs/guides/batch
Ref: https://platform.openai.com/docs/api-reference/completions

"""

import logging
import os


from functools import reduce
from collections import OrderedDict

import argparse
from pathlib import Path
import json
import numpy as np
from datasets import Dataset
from openai import OpenAI

from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# The dataset takes < 80K tokens, 5 sents
# I would want to scale it up to

# Martinez and other models
"""Martinez had 'Models included 7 flagship models from OpenAI (four large language models, and three “advanced reasoning models”), 
including GPT-3.5,
GPT-4, - Seems to be priced relatively very high (will check after token budgeting) 
GPT-4o - 
GPT-4.1,
and o1
o3,
o1-mini, 

Based the docs: 
To keep parity between pre GPT-5.1 and post GPT-5.1
ReasoningEffort medium
And a larger max_output_tokens to allow for 'reasoning' budget
Since max_new_tokens is the controlling parameter - 
we can set it to some round tokens, leaving budget for answer tokens. 
"""
model_list = [
]

# 4.1 mini is a test model
model_list = [
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-mini-2025-04-14",
    "o3-mini-2025-01-31",
    # "gpt-4o-mini-2024-07-18",
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
        config["max_output_tokens"] = 16 + 2048 # For o3 "medium" 1024 was not enough
        config["top_logprobs"] = None

    return config

def infer_and_extract_output(client, model, config, prompt_list):
    # model config
    # set_temperature : Boolean | None
    # max_output_tokens : int [16 for non reasoning, higher for reasoning]
    # reasoning_effort : "medium" | None
    # top_logprobs: int [20]
    logger.info(f"Using {config} for inference with {model}")
    max_output_tokens = config["max_output_tokens"]
    top_logprobs = config["top_logprobs"]
    temperature = config["temperature"]
    reasoning = config["reasoning"]

    def response(prompt):
        return client.responses.create(
            model=model,
            #messages=[{"role": "user", "content": prompt}],
            input=prompt,
            # No use of prompt template right now
            include=["message.output_text.logprobs"] if top_logprobs is not None else None,
            # This should include a reasoning budget for only the "reasoning" models
            max_output_tokens=max_output_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            reasoning=reasoning,
        )

    responses = [response(prompt) for prompt in tqdm(prompt_list, desc="Inferring prompts")]
    # print(responses)
    def output_from_response(response):
        # FIX:  content = response.output[0].content[0], # TypeError: 'NoneType' object is not subscriptable
        logging.info(f"Response for prompt: {response}")
        # For reasoning models, the output is in the second content
        for output in response.output:
            if output.content is not None:
                content = output.content[0]
                break
        # print(type(content[0]))
        # print(content[0])
        # TODO add the first token
        return {
            "text": content.text,
        }
    outputs = [output_from_response(response) for response in tqdm(responses, desc="Extracting output")]
    return outputs

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


    logger.info(f"Using {len(prompts_list)} prompts for inference")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    for model in model_list:
        config = get_config(model)
        logger.info(f"Inferring {model} for {len(prompts_list)} prompts")
        outputs = infer_and_extract_output(client, model, config, prompts_list)
        if len(outputs) != len(prompts_list):
            logger.errpr(f"!!!! Length mismatch: {len(outputs)} outputs vs {len(prompts_dataset)} dataset")


        def collate_outputs(dataset, outputs):
            # TODO abhip: unify candidate variables with model.py
            # Candidates copied from model.py
            candidates = candidates = ['AGREE', 'Agree', 'agree', 'DISAGREE', 'Disagree', 'disagree']
            MIN_FLOAT = np.finfo(float).min


            results_dict = {
                "title": dataset["Title"],
                "frame": dataset["frame"],
                "prompt": dataset["prompt"],
                "version": dataset["version"],
                # "output": [list(output["logprobs"])[0] for output in outputs],
                "output_text": [output["text"] for output in outputs],
                # "cum_logprob": [output.cumulative_logprob for output in outputs],
            }

            # candidate_logprobs = OrderedDict()
            #
            # for candidate in candidates:
            #     candidate_logprobs[candidate] = list()
            #
            # for output in outputs:
            #     for candidate in candidates:
            #         candidate_logprobs[candidate].append(
            #             output["logprobs"].get(candidate, MIN_FLOAT)
            #         )
            #
            # for candidate in candidate_logprobs:
            #     results_dict[candidate + "_probs"] = candidate_logprobs[candidate]

            return results_dict


        results_dict = collate_outputs(prompts_dataset, outputs)
        results = Dataset.from_dict(results_dict)
        # Save to JSON
        results.to_json(args.out.__str__() + f"-{model}.json", index=False)
        # Save to CSV
        results.to_csv(args.out.__str__() + f"-{model}.csv", index=False)
        logger.info(f"Saved results to {args.out.with_suffix('.json')} and {args.out.with_suffix('.csv')}")
