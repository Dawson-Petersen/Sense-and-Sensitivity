import argparse
import logging
import json
from collections import Counter
from pathlib import Path


import pandas as pd

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("prompts_file", type=str)
    arg_parser.add_argument("outputs_dir", type=str)
    args = arg_parser.parse_args()

    outputs_dir = args.outputs_dir
    logger.info(f"Processing the outputs in {outputs_dir}")
    dataset = pd.read_csv(args.prompts_file, sep="\t")
    for output_file in Path(outputs_dir).glob("*.jsonl"):
        logger.info(f"Processing {output_file}")
        responses = []
        with open(output_file, "r") as f:
            for line in f:
                responses.append(json.loads(line))

        def have_text(responses):
            return [
                "content" in response["response"]["body"]["output"][0] or
                (len(response["response"]["body"]["output"]) > 1 and "content" in response["response"]["body"]["output"][1])
                for response in responses
            ]

        def get_text_from_response(response):
                if "content" in response["response"]["body"]["output"][0]:
                    return response["response"]["body"]["output"][0]["content"][0]["text"]

                elif len( response["response"]["body"]["output"]) > 1 and "content" in response["response"]["body"]["output"][1]:
                    return response["response"]["body"]["output"][1]["content"][0]["text"]

                else:
                    logger.error(f"No content found for response {response}")
                    return None

        have_texts = have_text(responses)
        logger.info(f"Texts status {Counter(have_texts)} for {len(responses)} responses")
        if not all(have_texts):
            logger.error(f"Not all responses have text content, skipping {output_file}")

        outputs = [
            {
                "text" : get_text_from_response(response),
                "model" : response["response"]["body"]["model"],
            } for response in responses
        ]
        # Assuming the parity between prompts -> batches -> outputs, we can add the outputs to the dataset by matching the output file name to the batch file name in the dataset
        dataset["model"] = [output["model"] for output in outputs]
        dataset["output_text"] = [output["text"] for output in outputs]

        assert len(set(dataset["model"])) == 1, "Expected all outputs in the same file to be from the same model"
        model_name = dataset["model"][0]
        # Saving the dataset with outputs to a new file
        output_file = Path(outputs_dir) / f"{model_name}-results.csv"
        logger.info(f"Saving the dataset with outputs to {output_file}")
        dataset.to_csv(output_file, index=False)