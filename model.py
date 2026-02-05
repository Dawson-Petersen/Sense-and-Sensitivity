from typing import List, Tuple
from collections import OrderedDict
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, CompletionOutput
import torch


class MetaLinguisticJudgement:
    def __init__(self, model_name, seed, max_model_len=216):
        self.model_name = model_name
        self.infer_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=64,
            seed=seed)
        self.logprob_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=1,
            seed=seed,
            logprobs=1,
            prompt_logprobs=True
        )

        if 'gpt' in model_name:
            max_model_len = 256
        elif 'bloom' in model_name:
            max_model_len = 224
        elif 'ministral' in model_name:
            max_model_len = 240
        if 'bloom' in model_name:
            gpu_memory_utilization = 0.3
        else:
            gpu_memory_utilization = 0.92
        self.max_model_len = max_model_len

        if "70B" in self.model_name:
            tensor_parallel_size = 4
        else:
            tensor_parallel_size = 1

        num_gpus = torch.cuda.device_count()
        if num_gpus < tensor_parallel_size:
            raise RuntimeError(
                    f"Requested tensor_parallel_size={tp_size}, "
                            f"but only {num_gpus} GPU(s) are available."
                                )
        self.llm = LLM(
            model_name,
            max_model_len=max_model_len,
            seed=seed,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            disable_custom_all_reduce=True
        )

    def infer(self, prompts: List[str]) -> List[CompletionOutput]:
        outputs = self.llm.generate(prompts, self.infer_params)
        return [output.outputs[0] for output in outputs]

    def probs(self, prompts: List[str]) -> dict[str, List[float]]:
        # BLOOM/vLLM engine settings
        self.llm.llm_engine.max_num_seqs = 1

        SPACE = " "
        candidates = ['AGREE', 'Agree', 'agree', 'DISAGREE', 'Disagree', 'disagree']
        tokenizer = self.llm.get_tokenizer()

        candidate_token_data = OrderedDict()
        for candidate in candidates:
            token_ids = tokenizer.encode(SPACE + candidate, add_special_tokens=False)
            candidate_token_data[candidate] = token_ids

        prompts_with_token = [p + SPACE + answer for p in prompts for answer in candidates]

        outputs = self.llm.generate(prompts_with_token, self.logprob_params)

        candidate_logprobs = OrderedDict({cand: [] for cand in candidates})


        for batch_idx, output in enumerate(outputs):
            candidate_name = candidates[batch_idx % len(candidates)]
            target_token_ids = candidate_token_data[candidate_name]
            num_tokens = len(target_token_ids)

            if output.prompt_logprobs is not None:
                relevant_logprobs = output.prompt_logprobs[-num_tokens:]

                sequence_logprob = 0.0
                for i, token_id in enumerate(target_token_ids):
                    logprob_obj = relevant_logprobs[i].get(token_id)
                    if logprob_obj is not None:
                        sequence_logprob += logprob_obj.logprob
                    else:
                        sequence_logprob += -100.0

                candidate_logprobs[candidate_name].append(sequence_logprob)
            else:
                candidate_logprobs[candidate_name].append(float('-inf'))

        return candidate_logprobs

