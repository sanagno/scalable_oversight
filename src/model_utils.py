import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def get_model(model_name, cache_dir, dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = "auto" if device.type == "cpu" else (device.index or 0)

    has_flash_attn = False
    try:
        import flash_attn

        has_flash_attn = True
    except ImportError:
        print("Flash attention not found.")
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=parse_dtype(dtype),
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="flash_attention_2" if has_flash_attn else None,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    judge_model, judge_tokenizer, dataset, choices, append_tokens, method="max"
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
