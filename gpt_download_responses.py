import os
import json
import argparse
import logging
from datetime import datetime

from pathlib import Path

import pandas as pd

import openai

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("batch_submits", type=Path)
    arg_parser.add_argument("batch_downloads", type=Path)
    arg_parser.add_argument("outputs_dir", type=Path)
    args = arg_parser.parse_args()

    outputs_dir = args.outputs_dir
    if not outputs_dir.is_dir():
        raise ValueError(f"Results directory does not exist: {outputs_dir}")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)

    batches = pd.read_csv(args.batch_submits)
    batches = batches.to_dict(orient="records")
    for batch_submission in batches:
        batch_id = batch_submission["batch_id"]
        batch_file = batch_submission["batch_file"]
        batch_response = client.batches.retrieve(batch_id)
        logger.info(f"Batch {batch_id} for file {batch_file} is {batch_response.status}")
        logger.debug(f"Batch response for batch {batch_id} for file {batch_file} is : {batch_response}")
        if batch_response.output_file_id is None:
            logger.error(f"Batch {batch_id} for file {batch_file} is not complete yet, no output file available.")

        logger.info(f"Collecting the responses for batch {batch_id} for file {batch_file} from {batch_response.output_file_id}")
        outputs_response = client.files.content(batch_response.output_file_id)
        outputs = outputs_response.text

        # Saving the outputs to a file named after the batch_file in the results directory, they give jsonl
        model_name = batch_submission["model_name"]
        reasoning_effort = batch_submission["reasoning_effort"]
        output_file_path = outputs_dir / (f"{model_name}-{reasoning_effort}" + "_outputs.jsonl")
        logger.info(f"Saving the outputs for batch {batch_id} for file {batch_file} to {output_file_path}")
        batch_submission["output_file_path"] = output_file_path
        with open(output_file_path, "w") as f:
            f.write(outputs)

    logger.info(f"Finished collecting outputs for all batches, saving metadata to {args.batch_downloads}")
    pd.DataFrame(batches).to_csv(args.batch_downloads, index=False)