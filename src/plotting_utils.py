import time
import numpy as np


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


def get_last_exp_by_time(exps):
    # Sort orders by time
    sorted_orders = sorted(exps, key=lambda x: convert_to_timestamp(x), reverse=True)

    return sorted_orders[0]


def get_instuction_following_percentage(dataset, probabilities):
    assert len(dataset) == len(probabilities)

    instruction_following = []
    for i in range(len(dataset)):
        instruction_following.append(
            dataset[i]["explanation_advocate_idx"] == np.argmax(probabilities[i])
        )

    return np.array(instruction_following)
