import copy
import math

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from .definitions import HUGGIGNFACE_MODEL_PATHS, MAX_MODEL_LEN, TOKENIZER_NAME, OVERRIDE_CHAT_TEMPLATES


def parse_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def get_model(model_name, cache_dir, dtype, device, vllm=True):
    device_map = "auto" if device.type == "cpu" else (device.index or 0)

    huggingface_model_name = HUGGIGNFACE_MODEL_PATHS[model_name][dtype]
    tokenizer_name = TOKENIZER_NAME[model_name]

    if vllm:
        if dtype in ["float32", "float16"]:
            kwargs = {"dtype": dtype}
        elif dtype == "int8":
            kwargs = {"quantization": "GPTQ", "dtype": torch.float16}
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        model = LLM(
            huggingface_model_name,
            download_dir=cache_dir,
            max_model_len=MAX_MODEL_LEN[model_name],
            max_num_seqs=8,
            trust_remote_code=True,
            **kwargs,
        )
    else:
        has_flash_attn = False
        try:
            import flash_attn

            has_flash_attn = True
        except ImportError:
            print("Flash attention not found.")
            pass

        if dtype in ["float32", "float16"]:
            kwargs = {
                "torch_dtype": parse_dtype(dtype),
            }
        elif dtype == "int8":
            kwargs = {"load_in_8bit": True}
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            huggingface_model_name,
            low_cpu_mem_usage=True,
            device_map=device_map,
            attn_implementation="flash_attention_2" if has_flash_attn else None,
            cache_dir=cache_dir,
            **kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if model_name in OVERRIDE_CHAT_TEMPLATES:
        tokenizer.chat_template = OVERRIDE_CHAT_TEMPLATES[model_name]

    return model, tokenizer

@torch.no_grad()
def get_answer_probabilities(
    judge_model, judge_tokenizer, dataset, choices, base_answer
):
    ## This only works for single token answers...
    choices_ids = [
        judge_tokenizer.encode(choices[i], add_special_tokens=False)[-1]
        for i in range(len(choices))
    ]
    
    if type(judge_model) == LLM:
        prompts = [
            judge_tokenizer.apply_chat_template(
                dataset[i]["conversation_history"] + [{"role": "assistant", "content": base_answer}], tokenize=False,
            )
            for i in range(len(dataset))
        ]
        inputs = [
            judge_tokenizer(prompt, truncation=False, add_special_tokens=False)[
                "input_ids"
            ]
            for prompt in prompts
        ]
        
        # Remove from the end of the message ids until we find one of the choice_ids
        new_inputs = []
        for i, input_ids in enumerate(inputs):
            for j in range(len(input_ids)-1, -1, -1):
                if input_ids[j] in choices_ids:
                    new_inputs.append(input_ids[:j])
                    break

        outputs = judge_model.generate(
            prompt_token_ids=new_inputs,
            sampling_params=SamplingParams(
                **{"max_tokens": 1, "n": 1, "logprobs": len(judge_tokenizer)}
            ),
        )

        probabilities = []
        for out in outputs:
            try:
                probs = [math.exp(out.outputs[0].logprobs[0][x]) for x in choices_ids]
            except:
                probs = [np.nan for _ in choices_ids]
            probabilities.append(probs)
    else:
        raise NotImplementedError("deprecated")
        probabilities = []
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            prompt = judge_tokenizer.apply_chat_template(
                dataset[i]["conversation_history"]
            )
            inputs = judge_tokenizer(
                prompt, return_tensors="pt", truncation=False, add_special_tokens=False
            )["input_ids"]

            outputs = judge_model(inputs.to(judge_model.device))

            probs = torch.nn.functional.softmax(outputs.logits[0, -1], dim=-1)

            probs = probs[choices_ids].detach().cpu().numpy()

            probabilities.append(probs)

    return np.array(probabilities)


@torch.no_grad()
def get_model_generations(
    advocate_model,
    advocate_tokenizer,
    dataset,
    max_new_tokens=256,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    temperature=1.0,
    desc="Generating",
):
    if type(advocate_model) == LLM:
        prompts = [
            advocate_tokenizer.apply_chat_template(
                dataset[i]["conversation_history"], tokenize=False
            )
            for i in range(len(dataset))
        ]
        inputs = [
            advocate_tokenizer(prompt, truncation=False, add_special_tokens=False)[
                "input_ids"
            ]
            for prompt in prompts
        ]

        outputs = advocate_model.generate(
            prompt_token_ids=inputs,
            sampling_params=SamplingParams(
                **{
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                    "n": 1,
                }
            ),
        )

        generations = []
        for i in range(len(dataset)):
            generation = outputs[i].outputs[0].text

            generations.append(
                copy.deepcopy(dataset[i]) | {"generation": generation.strip()}
            )
    else:
        generations = []
        for i in tqdm(range(len(dataset)), desc=desc):
            prompt = advocate_tokenizer.apply_chat_template(
                dataset[i]["conversation_history"], tokenize=False
            )

            inputs = advocate_tokenizer(
                prompt, return_tensors="pt", truncation=False, add_special_tokens=False
            )["input_ids"]

            outputs = advocate_model.generate(
                inputs.to(advocate_model.device),
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1,
            )

            generated_tokens = outputs[0][inputs.shape[-1] :]
            generation = advocate_tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            generations.append(
                copy.deepcopy(dataset[i]) | {"generation": generation.strip()}
            )

    return generations
