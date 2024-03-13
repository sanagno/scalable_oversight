import numpy as np


def get_metric(probabiltiies, dataset, method="max"):
    assert len(probabiltiies) == len(dataset)
    metric = []

    for i in range(len(dataset)):
        if method == "max":
            metric.append(
                np.argmax(probabiltiies[i]) == dataset[i]["correct_answer_idx"]
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    return np.mean(metric)
