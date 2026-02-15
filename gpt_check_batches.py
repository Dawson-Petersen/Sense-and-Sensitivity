import os
import json
import argparse
import logging
from datetime import datetime

from pathlib import Path

import pandas as pd

import openai

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("batch_submits", type=Path)
    args = arg_parser.parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)

    batches = pd.read_csv(args.batch_submits)
    logger.info(f"Read {len(batches)} batches from {args.batch_submits}")
    statuses = []
    for _, batch in batches.iterrows():
        batch_id = batch["batch_id"]
        batch_file = batch["batch_file"]
        logger.info(f"Checking status {batch_id} corresponds to file {batch_file}")
        response = client.batches.retrieve(batch_id)
        logger.info(f"{batch_file}, {batch_id} is {response.status}")
        statuses.append(
            {
                "model_name": batch["model_name"],
                "reasoning_effort": batch["reasoning_effort"],
                "run_status": response.status,
            }
        )
    for status in statuses:
        logger.info(f"{status}")