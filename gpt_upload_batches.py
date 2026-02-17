import os
import json
import argparse
import logging
from pathlib import Path


import pandas as pd

import openai

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":


    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("batches_metadata", type=Path)
    arg_parser.add_argument("out_file", type=Path)
    args = arg_parser.parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)

    batches_metadata = args.batches_metadata
    batches = pd.read_csv(batches_metadata)
    batches = batches.to_dict(orient="records")

    logger.info(f"Uploading {len(batches)} batches from {batches_metadata}")
    uploads = []
    for batch in batches:
        batch_file = batch["batch_file"]
        logger.info(f"Uploading {batch_file} for batch {batch}")
        with open(batch_file, "rb") as f:
            batch_file_response = client.files.create(file=f, purpose="batch")

        logger.info(f"Got {batch_file_response} for {batch_file}")
        uploads.append({
            **batch,
            "file_id": batch_file_response.id,
            "object": batch_file_response.object,
            "bytes": batch_file_response.bytes,
            "created_at": batch_file_response.created_at,
            "purpose": batch_file_response.purpose,
            "uploaded_filename": batch_file_response.filename,
        })

    uploads_file = args.out_file
    logger.info(f"Uploaded {len(uploads)} batches, saving upload metadata to {uploads_file}")
    pd.DataFrame(uploads).to_csv(uploads_file, index=False)