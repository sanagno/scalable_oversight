import time
import os
import copy
import numpy as np
from .definitions import ADDITIONAL_SYSTEM_PROMPTS, LEVELS


LEVEL_COLORS = {
    "level_0": "#df2d1d",
    "level_1": "#18255f",
    "level_2": "#465ecf",
    "level_3": "#6c8ff1",
    "level_4": "#ea7a5f",
    "level_5": "#d9421c",
}


DATASET_NAMES = {
    "gpqa": "GPQA",
    "quality": "QuALITY",
    "openbookqa": "OpenBookQA",
    "boolq": "BoolQ",
    "commonsense_qa": "CommonsenseQA",
    "piqa": "PIQA",
    "siqa": "SocialIQA",
    "wiki_qa": "WikiQA",
}

MODEL_NAMES = {
    "Llama-2-70b-chat": "$\it{Llama}$",
    "Mixtral": "$\it{Mixtral}$",
    "falcon-40b": "$\it{Falcon}$",
}

RANDOM_PERFORMANCE = {
    "gpqa": 0.25,
    "quality": 0.25,
    "openbookqa": 0.25,
    "boolq": 0.5,
    "commonsense_qa": 0.2,
    "piqa": 0.5,
    "siqa": 0.3333,
    "wiki_qa": 0.25697722567287784,
}


def convert_to_timestamp(time_str):
    return time.mktime(time.strptime(time_str, "%Y-%m-%d-%H-%M-%S"))


def get_last_exp_by_time(data_path, model, dataset_name, additional_syste_prompt):
    base_dir = os.path.join(data_path, model, dataset_name)
    if additional_syste_prompt != "None":
        base_dir = os.path.join(base_dir, additional_syste_prompt)

    exps = os.listdir(base_dir)
    exps = [exp for exp in exps if exp not in list(ADDITIONAL_SYSTEM_PROMPTS.keys())]

    # Sort orders by time
    sorted_orders = sorted(exps, key=lambda x: convert_to_timestamp(x), reverse=True)

    return os.path.join(base_dir, sorted_orders[0])


def add_question_idx_in_dataset(dataset):
    if "question_idx" not in dataset[0]:
        new_dataset = copy.deepcopy(dataset)

        i = 0
        cnt = 0
        while i < len(dataset):
            if new_dataset[i]["explanation_level"] not in LEVELS:
                # only one sample
                new_dataset[i]["question_idx"] = i
                i += 1
                cnt += 1
            else:
                for _ in range(len(new_dataset[i]["random_order"])):
                    new_dataset[i]["question_idx"] = cnt
                    i += 1
                cnt += 1

        assert i == len(dataset)

        return new_dataset
    else:
        return dataset


def probability_stats(
    dataset, probabilities, dataset_bias, probabilities_bias, renomrmalize=True
):
    results = []

    for i in range(len(dataset_bias)):
        question_idx = dataset_bias[i]["question_idx"]

        assert dataset[question_idx]["question_idx"] == question_idx

        prob_original_correct = probabilities[question_idx][
            dataset[question_idx]["correct_answers_idx"][0]
        ] / np.sum(probabilities[question_idx])
        is_original_correct = (
            np.argmax(probabilities[question_idx])
            in dataset[question_idx]["correct_answers_idx"]
        )

        prob_bias_correct = probabilities_bias[i][
            dataset_bias[i]["correct_answers_idx"][0]
        ] / np.sum(probabilities_bias[i])

        is_bias_correct = (
            np.argmax(probabilities_bias[i]) in dataset_bias[i]["correct_answers_idx"]
        )

        is_explanation_correct = dataset_bias[i]["explanation_is_correct"]

        results.append(
            {
                "prob_original_correct": prob_original_correct,
                "is_original_correct": is_original_correct,
                "prob_bias_correct": prob_bias_correct,
                "is_bias_correct": is_bias_correct,
                "is_explanation_correct": is_explanation_correct,
            }
        )

    return {k: np.array([r[k] for r in results]) for k in results[0]}
