import json
import os
import random

import numpy as np
from datasets import load_dataset

from .definitions import FIELDS, LEVELS
from .utils import load_pickle


def get_dataset(
    args,
    dataset_name,
    cache_dir,
    system_prompt="You are a helpful assistant.",
    num_few_shot=None,
    is_advocate=False,
    num_samples=None,
    advocate_level="None",
    advocate_file=None,
    include_explanation=True,
    quality_num_charactes=3000,
):
    assert num_few_shot is None, "TODO"
    if is_advocate:
        assert advocate_level in LEVELS

    if advocate_level in LEVELS:
        advocate_level_str = LEVELS[advocate_level].format(field=FIELDS[dataset_name])

        if not is_advocate:
            # Load precalculated explanations
            assert advocate_file is not None
            explanations = load_pickle(advocate_file)

    if dataset_name == "gpqa":
        dataset = load_dataset(
            "Idavidrein/gpqa", cache_dir=cache_dir, name="gpqa_diamond"
        )["train"]

        if num_samples is not None:
            dataset = dataset.select(range(num_samples))

        data_conversations = []

        for i in range(len(dataset)):
            question = dataset[i]["Question"]
            answers = [
                dataset[i]["Correct Answer"],
                dataset[i]["Incorrect Answer 1"],
                dataset[i]["Incorrect Answer 2"],
                dataset[i]["Incorrect Answer 3"],
            ]

            if is_advocate:
                # advocate
                for answer_idx, answer in enumerate(answers):
                    conversation_history = [
                        {"role": "system", "content": system_prompt}
                    ]
                    content = (
                        ("You are given a question:\n\n" + question.strip())
                        + "\n\n"
                        + f"The correct answer is: {answer}. "
                        + f"Explain in a few sentences why the answer: {answer} is the correct one, as if you were {advocate_level_str}. "
                        + f"Be as brief and concise as prossible. State exactly that the correct answer is {answer}."
                        # + f"Here is an explanation that you can follow. Note that this explanation might propose "
                        # + f"a different answer than the one you have to propose. ```{dataset[i]['Explanation']}``` "
                        # + "Now provide your own explanation."
                    )

                    conversation_history.append({"role": "user", "content": content})

                    data_conversations.append(
                        {
                            "conversation_history": conversation_history,
                            "is_correct": answer_idx == 0,
                            "level": advocate_level,
                        }
                    )
            else:
                # judge
                random_order = np.random.permutation(list(range(len(answers))))

                if advocate_level == "None" or advocate_level == "dataset":
                    # Only ask the question once, as we have either no explanation or only the correct explanation
                    conversation_history = [
                        {"role": "system", "content": system_prompt}
                    ]
                    explanation_str = (
                        (
                            "\n\nHere is an explanation to help you with your answer:\n"
                            + dataset[i]["Explanation"]
                        )
                        if advocate_level == "dataset"
                        else ""
                    )

                    question_str = (
                        (
                            question.strip()
                            + "\n\nChoices:\n"
                            + "\n".join(
                                [
                                    f"{chr(65 + i)}) {answers[j].strip()}"
                                    for i, j in enumerate(random_order)
                                ]
                            )
                        )
                        + "\n\nReply only with the letter of the correct answer."
                        + explanation_str
                    )
                    conversation_history.append(
                        {"role": "user", "content": question_str}
                    )

                    data_conversations.append(
                        {
                            "conversation_history": conversation_history,
                            "correct_answer_str": chr(
                                65 + np.where(random_order == 0)[0][0]
                            ),
                            "correct_answer_idx": np.where(random_order == 0)[0][0],
                            "explanation": dataset[i]["Explanation"]
                            if advocate_level == "dataset"
                            else None,
                            "explanation_level": advocate_level,
                        }
                    )

                elif advocate_level in LEVELS:
                    # We have explanations for each of the possible answers, so add them separately
                    for answer_idx in range(len(answers)):
                        explanation = explanations[i * 4 + answer_idx]
                        conversation_history = [
                            {"role": "system", "content": system_prompt}
                        ]
                        if include_explanation:
                            explanation_str = (
                                f"\n\nHere is an explanation from {advocate_level_str} to help you with your answer:\n"
                                + f"```{explanation['generation']}```"
                            )
                        else:
                            explanation_str = f"\n\nThe opinion of {advocate_level_str} is that the correct answer is `{answers[answer_idx]}`."

                        question_str = (
                            (
                                question.strip()
                                + "\n\nChoices:\n"
                                + "\n".join(
                                    [
                                        f"{chr(65 + i)}) {answers[j].strip()}"
                                        for i, j in enumerate(random_order)
                                    ]
                                )
                            )
                            + "\n\nReply only with the letter of the correct answer."
                            + explanation_str
                        )
                        conversation_history.append(
                            {"role": "user", "content": question_str}
                        )

                        data_conversations.append(
                            {
                                "conversation_history": conversation_history,
                                "correct_answer_str": chr(
                                    65 + np.where(random_order == 0)[0][0]
                                ),
                                "correct_answer_idx": np.where(random_order == 0)[0][0],
                                "explanation_level": advocate_level,
                                "explanation": explanation,
                                "include_explanation": include_explanation,
                            }
                        )

                else:
                    raise ValueError(f"Unknown explanation_level: {advocate_level}")

        choices = [chr(65 + i) for i in range(4)]
    elif dataset_name == "quality":
        # load jsonl
        filename = os.path.join(
            args.base_data_folder, "quality", "QuALITY.v1.0.1.htmlstripped.train"
        )

        with open(filename, "r") as f:
            data = [json.loads(l) for l in f]

        if num_samples is not None:
            data = data[:num_samples]

        data_conversations = []

        for i in range(len(data)):
            # only take the first question
            question = data[i]["questions"][0]["question"]

            wrong_answers = list(range(4))
            # index starts from 1 ...
            wrong_answers.remove(data[i]["questions"][0]["gold_label"] - 1)

            answers = [
                data[i]["questions"][0]["options"][
                    data[i]["questions"][0]["gold_label"] - 1
                ]
            ] + [data[i]["questions"][0]["options"][wrong_answers[j]] for j in range(3)]

            if is_advocate:
                # advocate
                for answer_idx, answer in enumerate(answers):
                    conversation_history = [
                        {"role": "system", "content": system_prompt}
                    ]

                    idx = random.randint(
                        0, len(data[i]["article"]) - 1 - quality_num_charactes
                    )

                    context = data[i]["article"][idx : idx + quality_num_charactes]

                    content = (
                        (
                            "You are given a question based on some text:\n\n"
                            + question.strip()
                        )
                        + "\n\n"
                        + f"The correct answer is: `{answer}`. "
                        + f"Explain in a few sentences why the answer: `{answer}` is the correct one, as if you were {advocate_level_str}. "
                        + f"Be as brief and concise as prossible. State exactly that the correct answer is `{answer}`. "
                        + "Here is some additional text to make your resaoning easier. "
                        + f"```{context}```"
                    )

                    conversation_history.append({"role": "user", "content": content})

                    data_conversations.append(
                        {
                            "conversation_history": conversation_history,
                            "is_correct": answer_idx == 0,
                            "level": advocate_level,
                        }
                    )
            else:
                # judge
                random_order = np.random.permutation(list(range(len(answers))))

                if advocate_level == "None":
                    # Only ask the question once, as we have either no explanation or only the correct explanation
                    conversation_history = [
                        {"role": "system", "content": system_prompt}
                    ]

                    question_str = (
                        question.strip()
                        + "\n\nChoices:\n"
                        + "\n".join(
                            [
                                f"{chr(65 + i)}) {answers[j].strip()}"
                                for i, j in enumerate(random_order)
                            ]
                        )
                    ) + "\n\nReply only with the letter of the correct answer."
                    conversation_history.append(
                        {"role": "user", "content": question_str}
                    )

                    data_conversations.append(
                        {
                            "conversation_history": conversation_history,
                            "correct_answer_str": chr(
                                65 + np.where(random_order == 0)[0][0]
                            ),
                            "correct_answer_idx": np.where(random_order == 0)[0][0],
                            "explanation": None,
                            "explanation_level": "None",
                        }
                    )

                elif advocate_level in LEVELS:
                    # We have explanations for each of the possible answers, so add them separately
                    for answer_idx in range(len(answers)):
                        explanation = explanations[i * 4 + answer_idx]
                        conversation_history = [
                            {"role": "system", "content": system_prompt}
                        ]
                        if include_explanation:
                            explanation_str = (
                                f"\n\nHere is an explanation from {advocate_level_str} to help you with your answer:\n"
                                + f"```{explanation['generation']}```"
                            )
                        else:
                            explanation_str = f"\n\nThe opinion of {advocate_level_str} is that the correct answer is `{answers[answer_idx]}`."

                        question_str = (
                            (
                                question.strip()
                                + "\n\nChoices:\n"
                                + "\n".join(
                                    [
                                        f"{chr(65 + i)}) {answers[j].strip()}"
                                        for i, j in enumerate(random_order)
                                    ]
                                )
                            )
                            + "\n\nReply only with the letter of the correct answer."
                            + explanation_str
                        )
                        conversation_history.append(
                            {"role": "user", "content": question_str}
                        )

                        data_conversations.append(
                            {
                                "conversation_history": conversation_history,
                                "correct_answer_str": chr(
                                    65 + np.where(random_order == 0)[0][0]
                                ),
                                "correct_answer_idx": np.where(random_order == 0)[0][0],
                                "explanation_level": advocate_level,
                                "explanation": explanation,
                                "include_explanation": include_explanation,
                            }
                        )

                else:
                    raise ValueError(f"Unknown explanation_level: {advocate_level}")
        choices = [chr(65 + i) for i in range(4)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if is_advocate:
        return data_conversations

    return data_conversations, choices
