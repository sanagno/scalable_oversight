import numpy as np

from .definitions import LEVELS


def get_metrics(
    advocate_level,
    probabilities,
    dataset,
    evaluation_method="argmax",
    indices=None,
):
    assert len(probabilities) == len(dataset)
    metrics = {
        "no_explanation": [],
        "dataset_explanation": [],
        "correct_advocate_explanation": [],
        "incorrect_advocate_explanation": [],
    }

    for i in range(len(dataset)):
        if indices is not None and i not in indices:
            continue

        # check if any probablity is nan
        if np.isnan(probabilities[i]).any():
            # something went wrong, e.g. too many tokens in input
            continue

        if evaluation_method == "argmax":
            value = np.argmax(probabilities[i]) in dataset[i]["correct_answers_idx"]
        else:
            raise ValueError(f"Unknown method: {evaluation_method}")

        if advocate_level == "None":
            metrics["no_explanation"].append(value)
        elif advocate_level == "dataset":
            metrics["dataset_explanation"].append(value)
        elif advocate_level in LEVELS:
            assert dataset[i]["explanation_is_correct"] is not None

            if dataset[i]["explanation_is_correct"]:
                metrics["correct_advocate_explanation"].append(value)
            else:
                metrics["incorrect_advocate_explanation"].append(value)
        else:
            raise ValueError(f"Unknown advocate_level: {advocate_level}")

    return {k: np.mean(v) for k, v in metrics.items()}


def get_instuction_following_percentage(dataset, probabilities, indices=None):
    assert len(dataset) == len(probabilities)

    instruction_following = []
    for i in range(len(dataset)):
        if indices is not None and i not in indices:
            continue

        instruction_following.append(
            dataset[i]["explanation_advocate_idx"] == np.argmax(probabilities[i])
        )

    return np.array(instruction_following)


def get_sample_metrics(
    probabilities,
    dataset,
    evaluation_method="argmax",
):
    assert len(probabilities) == len(dataset)

    values = []

    for i in range(len(dataset)):
        # check if any probablity is nan
        if np.isnan(probabilities[i]).any():
            # something went wrong, e.g. too many tokens in input
            continue

        if evaluation_method == "argmax":
            value = np.argmax(probabilities[i]) in dataset[i]["correct_answers_idx"]
        else:
            raise ValueError(f"Unknown method: {evaluation_method}")

        values.append(value)

    return np.array(values)
