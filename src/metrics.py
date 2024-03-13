import numpy as np
from .definitions import LEVELS


def get_metrics(
    dataset_name, advocate_level, probabiltiies, dataset, evaluation_method="argmax"
):
    assert len(probabiltiies) == len(dataset)
    metrics = {
        "no_explanation": [],
        "dataset_explanation": [],
        "correct_advocate_explanation": [],
        "incorrect_advocate_explanation": [],
    }

    for i in range(len(dataset)):
        if evaluation_method == "argmax":
            value = np.argmax(probabiltiies[i]) == dataset[i]["correct_answer_idx"]
        else:
            raise ValueError(f"Unknown method: {evaluation_method}")

        if advocate_level == "None":
            metrics["no_explanation"].append(value)
        elif advocate_level == "dataset":
            metrics["dataset_explanation"].append(value)
        elif advocate_level in LEVELS:
            if dataset_name == "gpqa":
                if i % 4 == 0:
                    metrics["correct_advocate_explanation"].append(value)
                else:
                    metrics["incorrect_advocate_explanation"].append(value)
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")
        else:
            raise ValueError(f"Unknown advocate_level: {advocate_level}")

    return {k: np.mean(v) for k, v in metrics.items()}
