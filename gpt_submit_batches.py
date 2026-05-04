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
    arg_parser.add_argument("uploads_file", type=Path)
    arg_parser.add_argument("submits_file", type=Path)
    args = arg_parser.parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)

    uploads = pd.read_csv(args.uploads_file)
    # Submit ('create') batches for each uploaded batch file
    logger.info(f"Uploads : {uploads.describe()}")
    submits = []
    for batch_upload in uploads.to_dict(orient="records"):
        batch_file, file_id = batch_upload["uploaded_filename"], batch_upload["file_id"]
        logger.info(f"Submitting batch for {batch_file} with file_id {file_id}")
        batch_submit_response = client.batches.create(
            input_file_id=file_id,
            endpoint = "/v1/responses",
            completion_window= "24h",
            metadata = {
                "description": f"Batch submit for {batch_file} at {datetime.now().isoformat()}",
            }
        )

        logger.info(f"Got {batch_submit_response} for batch submit of {batch_file}")
        submits.append({
            **batch_upload,
            "batch_file": batch_file,
            "file_id": batch_submit_response.input_file_id,
            "batch_id": batch_submit_response.id,
            "completion_window": batch_submit_response.completion_window,
            "created_at": batch_submit_response.created_at,
            "metadata": batch_submit_response.metadata,
        })


    submit_file =  args.submits_file
    logger.info(f"Submitted {len(submits)} batches, saving submit metadata to {submit_file}")
    pd.DataFrame(submits).to_csv(submit_file, index=False)

