import copy
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from .definitions import HUGGIGNFACE_MODEL_PATHS


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

    has_flash_attn = False
    try:
        import flash_attn

        has_flash_attn = True
    except ImportError:
        print("Flash attention not found.")
        pass

    huggingface_model_name = HUGGIGNFACE_MODEL_PATHS[model_name][dtype]

    if vllm:
        if dtype in ["float32", "float16"]:
            kwargs = {"dtype": dtype}
        elif dtype == "int8":
            kwargs = {"quantization": "GPTQ"}
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        model = LLM(
            huggingface_model_name,
            download_dir=cache_dir,
            max_model_len=2048,
            max_num_seqs=8,
            **kwargs,
        )
    else:
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

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

    return model, tokenizer


def get_model_probabilities(
    judge_model, judge_tokenizer, conversation_history, append_tokens, choices_ids
):
    prompt = judge_tokenizer.apply_chat_template(conversation_history, tokenize=False)
    inputs = judge_tokenizer(
        prompt, return_tensors="pt", truncation=False, add_special_tokens=False
    )["input_ids"]
    if len(append_tokens) > 0:
        inputs = torch.cat(
            [inputs, torch.tensor(append_tokens)[None].to(inputs.device)], dim=-1
        )

    outputs = judge_model(inputs.to(judge_model.device))

    probs = torch.nn.functional.softmax(outputs.logits[0, -1], dim=-1)

    return probs[choices_ids].detach().cpu().numpy()


@torch.no_grad()
def get_answer_probabilities(
    judge_model, judge_tokenizer, dataset, choices, append_tokens
):
    choices_ids = [
        judge_tokenizer.encode(choices[i], add_special_tokens=False)
        for i in range(len(choices))
    ]
    assert all([len(choices_ids[i]) == 1 for i in range(1, len(choices_ids))])
    choices_ids = [x[0] for x in choices_ids]

    probabilities = []
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        probs = get_model_probabilities(
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            conversation_history=dataset[i]["conversation_history"],
            append_tokens=append_tokens,
            choices_ids=choices_ids,
        )

        probabilities.append(probs)

    return probabilities


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

        outputs = advocate_model.generate(
            prompts,
            SamplingParams(
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
