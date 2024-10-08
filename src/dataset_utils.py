import copy
import json
import os
import random

import numpy as np
from datasets import load_dataset

from .definitions import FIELDS, LEVELS
from .utils import load_pickle

MAX_SAMPLES = 200


def get_dataset_to_format(
    dataset_name,
    cache_dir,
    base_data_folder,
    quality_num_charactes=2000,
    wiki_qa_max_num_answers=8,  # only keep questions with up to this many answers ...
):
    if dataset_name == "gpqa":
        dataset = load_dataset(
            "Idavidrein/gpqa", cache_dir=cache_dir, name="gpqa_diamond"
        )["train"]

        data = []
        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break
            question = (
                "You are given a question. Question: " + dataset[i]["Question"]
            )
            correct_answers = [dataset[i]["Correct Answer"].strip()]
            incorrect_answers = [
                dataset[i]["Incorrect Answer 1"].strip(),
                dataset[i]["Incorrect Answer 2"].strip(),
                dataset[i]["Incorrect Answer 3"].strip(),
            ]
            explanation = dataset[i]["Explanation"]

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": explanation,
                }
            )

        choices = [" " + chr(65 + i) for i in range(4)]
    elif dataset_name == "quality":
        filename = os.path.join(
            base_data_folder, "quality", "QuALITY.v1.0.1.htmlstripped.train"
        )

        with open(filename, "r") as f:
            dataset = [json.loads(l) for l in f]

        data = []

        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break
            # only take the first question
            question = (
                "You are given a question. Question: "
                + dataset[i]["questions"][0]["question"]
            )

            wrong_answers = list(range(4))
            # index starts from 1 ...
            wrong_answers.remove(dataset[i]["questions"][0]["gold_label"] - 1)

            idx = random.randint(
                0, len(dataset[i]["article"]) - 1 - quality_num_charactes
            )

            context = dataset[i]["article"][idx : idx + quality_num_charactes]

            data.append(
                {
                    "question": question,
                    "correct_answers": [
                        dataset[i]["questions"][0]["options"][
                            dataset[i]["questions"][0]["gold_label"] - 1
                        ].strip()
                    ],
                    "incorrect_answers": [
                        dataset[i]["questions"][0]["options"][
                            wrong_answers[j]
                        ].strip()
                        for j in range(3)
                    ],
                    "explanation": context,
                }
            )

        choices = [" " + chr(65 + i) for i in range(4)]
    elif dataset_name == "wiki_qa":
        dataset = load_dataset("wiki_qa", cache_dir=cache_dir)["test"]

        data = []

        i = 0
        while i < len(dataset):
            if len(data) == MAX_SAMPLES:
                break
            question_id = dataset[i]["question_id"]

            indices = [i]
            i += 1

            while i < len(dataset):
                if question_id == dataset[i]["question_id"]:
                    indices.append(i)
                    i += 1
                else:
                    break

            if (
                len(indices) == 0
                or np.where([dataset[x]["label"] == 1 for x in indices])[
                    0
                ].shape[0]
                != 1
                or len(indices) > wiki_qa_max_num_answers
            ):
                continue

            data.append(
                {
                    "question": "You are given a question. Question: "
                    + dataset[indices[0]]["question"],
                    "correct_answers": [
                        dataset[i]["answer"].strip()
                        for i in indices
                        if dataset[i]["label"] == 1
                    ],
                    "incorrect_answers": [
                        dataset[i]["answer"].strip()
                        for i in indices
                        if dataset[i]["label"] == 0
                    ],
                    "explanation": "",
                }
            )

        choices = [" " + chr(65 + i) for i in range(wiki_qa_max_num_answers)]
    elif dataset_name == "boolq":
        dataset = load_dataset("boolq", cache_dir=cache_dir)["validation"]

        data = []

        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break
            question = (
                "You are given a question. Question: " + dataset[i]["question"]
            )
            assert dataset[i]["answer"] in [True, False]
            correct_answers = ["True"] if dataset[i]["answer"] else ["False"]
            incorrect_answers = ["False"] if dataset[i]["answer"] else ["True"]
            explanation = dataset[i]["passage"]

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": explanation,
                }
            )
        choices = [" " + chr(65 + i) for i in range(2)]
    elif dataset_name == "openbookqa":
        dataset = load_dataset("allenai/openbookqa", cache_dir=cache_dir)[
            "validation"
        ]

        data = []

        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break

            question = (
                "Continue the following sentence: "
                + dataset[i]["question_stem"]
            )
            correct_answers = [
                dataset[i]["choices"]["text"][idx]
                for idx in range(4)
                if dataset[i]["answerKey"]
                == dataset[i]["choices"]["label"][idx]
            ]
            incorrect_answers = [
                dataset[i]["choices"]["text"][idx]
                for idx in range(4)
                if dataset[i]["answerKey"]
                != dataset[i]["choices"]["label"][idx]
            ]

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": "",
                }
            )
        choices = [" " + chr(65 + i) for i in range(4)]
    elif dataset_name == "commonsense_qa":
        dataset = load_dataset("tau/commonsense_qa", cache_dir=cache_dir)[
            "validation"
        ]

        data = []

        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break

            question = (
                "You are given a question. Question: " + dataset[i]["question"]
            )
            correct_answers = [
                dataset[i]["choices"]["text"][idx]
                for idx in range(5)
                if dataset[i]["answerKey"]
                == dataset[i]["choices"]["label"][idx]
            ]
            incorrect_answers = [
                dataset[i]["choices"]["text"][idx]
                for idx in range(5)
                if dataset[i]["answerKey"]
                != dataset[i]["choices"]["label"][idx]
            ]

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": "",
                }
            )
        choices = [" " + chr(65 + i) for i in range(5)]
    elif dataset_name == "commonsense_qa-train":
        dataset = load_dataset("tau/commonsense_qa", cache_dir=cache_dir)[
            "train"
        ]

        data = []

        for i in range(len(dataset)):
            question = (
                "You are given a question. Question: " + dataset[i]["question"]
            )
            correct_answers = [
                dataset[i]["choices"]["text"][idx]
                for idx in range(5)
                if dataset[i]["answerKey"]
                == dataset[i]["choices"]["label"][idx]
            ]
            incorrect_answers = [
                dataset[i]["choices"]["text"][idx]
                for idx in range(5)
                if dataset[i]["answerKey"]
                != dataset[i]["choices"]["label"][idx]
            ]

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": "",
                }
            )
        choices = [" " + chr(65 + i) for i in range(5)]
    elif dataset_name == "piqa":
        dataset = load_dataset("piqa", cache_dir=cache_dir)["validation"]

        data = []

        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break

            question = (
                "You are given a goal. You have to choose the best solution"
                " based on commonsense reasoning. Goal: "
                + dataset[i]["goal"]
            )
            correct_answers = [
                dataset[i]["sol1"]
                if dataset[i]["label"] == 0
                else dataset[i]["sol2"]
            ]
            incorrect_answers = [
                dataset[i]["sol2"]
                if dataset[i]["label"] == 0
                else dataset[i]["sol1"]
            ]

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": "",
                }
            )

        choices = [" " + chr(65 + i) for i in range(2)]
    elif dataset_name == "siqa":
        dataset = load_dataset("social_i_qa", cache_dir=cache_dir)[
            "validation"
        ]

        data = []

        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break

            question = (
                "Answer the following question based on the provided context."
                " Question: "
                + dataset[i]["question"]
                + " Context: "
                + dataset[i]["context"]
            )

            answers = [
                dataset[i]["answerA"],
                dataset[i]["answerB"],
                dataset[i]["answerC"],
            ]

            correct_answer_idx = int(dataset[i]["label"]) - 1

            correct_answers = [answers[correct_answer_idx]]
            answers.pop(correct_answer_idx)
            incorrect_answers = answers

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": "",
                }
            )
        choices = [" " + chr(65 + i) for i in range(3)]
    elif dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", name="main", cache_dir=cache_dir)[
            "test"
        ]

        data = []

        for i in range(len(dataset)):
            if i == MAX_SAMPLES:
                break

            question = (
                "You are given a question. Question: " + dataset[i]["question"]
            )
            explanation, correct_answer = dataset[i]["answer"].split("\n####")
            correct_answer = correct_answer.strip().replace(",", "")

            correct_answers = [correct_answer]
            current_answers = [int(correct_answer)]
            incorrect_answers = []

            for _ in range(3):
                random_answer = random.randint(10, 1000)
                if random_answer in current_answers:
                    continue
                incorrect_answers.append(str(random_answer))

            data.append(
                {
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "explanation": explanation,
                }
            )

        choices = [" " + chr(65 + i) for i in range(4)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return data, choices


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
    return_raw=False,
    response_type="None",
    base_answer_str="The right answer is the letter A",
    additional_advocate_prompt="",
    num_explanations=None,
    extra_advocate_str="",
    only_correct_explanations=False,
):
    if is_advocate:
        assert advocate_level in LEVELS
        assert num_few_shot is None

    if advocate_level in LEVELS:
        advocate_level_str = LEVELS[advocate_level].format(
            field=FIELDS[dataset_name]
        )

        if not is_advocate:
            # Load precalculated explanations
            assert advocate_file is not None
            advocate_explanations = load_pickle(advocate_file)
            advocate_explanations_idx = -1

    dataset, choices = get_dataset_to_format(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        base_data_folder=args.base_data_folder,
    )

    if return_raw:
        return dataset

    if num_samples is not None:
        dataset = dataset[:num_samples]

    data_conversations = []

    for i in range(len(dataset)):
        question = dataset[i]["question"]
        correct_answers = dataset[i]["correct_answers"]
        incorrect_answers = dataset[i]["incorrect_answers"]
        dataset_explanation = dataset[i]["explanation"]

        answers = correct_answers + incorrect_answers

        if is_advocate:
            # advocate
            for answer_idx, answer in enumerate(
                correct_answers + incorrect_answers
            ):
                conversation_history = [
                    {"role": "system", "content": system_prompt}
                ]
                content = (
                    question.strip()
                    + "\n\n"
                    + f"The correct answer is: `{answer}`. "
                    + f"Explain in a few sentences why the answer: `{answer}`"
                    " is the correct one, remember that you are"
                    f" {advocate_level_str}. "
                    + "Be very brief and concise. State exactly that the"
                    f" correct answer is `{answer}`."
                )

                if len(dataset_explanation) > 0:
                    content += (
                        f"\nHere is some additional information to make your"
                        f" resaoning easier.\n"
                        + f"```{dataset_explanation}```"
                        + additional_advocate_prompt
                    )

                conversation_history.append(
                    {"role": "user", "content": content}
                )

                data_conversations.append(
                    {
                        "conversation_history": conversation_history,
                        "is_correct": answer_idx < len(correct_answers),
                        "level": advocate_level,
                        "answer_idx": answer_idx,
                        "question_idx": i,
                    }
                )
        else:
            # judge
            random_order = np.random.permutation(list(range(len(answers))))
            # random_order = np.arange(len(answers))

            common = {
                "wrong_answers_idx": [
                    np.where(random_order == i)[0][0]
                    for i in range(len(correct_answers), len(answers))
                ],
                "correct_answers_idx": [
                    np.where(random_order == i)[0][0]
                    for i in range(len(correct_answers))
                ],
                "random_order": random_order,
                "question_idx": i,
                "answer_str": base_answer_str[:-2]
                + choices[np.where(random_order == 0)[0][0]],
            }

            if advocate_level == "None" or advocate_level == "dataset":
                # Only ask the question once, as we have either no explanation or only the correct explanation

                explanation_str = (
                    (
                        "\n\nHere is an explanation to help you with your"
                        " answer:\n"
                        + dataset_explanation
                    )
                    if advocate_level == "dataset"
                    else ""
                )
                if advocate_level == "None":
                    explanation_is_correct = None
                elif advocate_level == "dataset":
                    explanation_is_correct = True

                question_str = (
                    (
                        question.strip()
                        + "\n\nChoose one from the choices:\n"
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
                conversation_history = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_str},
                ]

                data_conversations.append(
                    {
                        "conversation_history": conversation_history,
                        "explanation": dataset_explanation
                        if advocate_level == "dataset"
                        else None,
                        "explanation_level": advocate_level,
                        "explanation_is_correct": explanation_is_correct,
                    }
                    | copy.deepcopy(common)
                )

            elif advocate_level in LEVELS:
                for answer_idx, answer in enumerate(
                    correct_answers + incorrect_answers
                ):
                    if num_explanations is None:
                        advocate_explanations_idx += 1
                        advocate_explanation = advocate_explanations[
                            advocate_explanations_idx
                        ]

                        if include_explanation:
                            explanation_str = (
                                "\n\nHere is an explanation from"
                                f" {advocate_level_str} to help you with your"
                                " answer:\n"
                                + extra_advocate_str
                                + f"```{advocate_explanation['generation']}```"
                            )
                        else:
                            explanation_str = (
                                f"\n\nThe opinion of {advocate_level_str} is"
                                " that the correct answer is"
                                f" `{answers[answer_idx]}`."
                                + extra_advocate_str
                            )

                        explanation_is_correct = advocate_explanation[
                            "is_correct"
                        ]
                    else:
                        num_explanations_to_provide = min(
                            num_explanations, len(answers)
                        )
                        # Take num_explanations_to_provide random answers
                        # and provide explanations for them
                        explanation_str = ""

                        random_sample_explanations = np.random.permutation(
                            len(answers)
                        )[:num_explanations_to_provide]

                        for j in random_sample_explanations:
                            advocate_explanation = advocate_explanations[
                                advocate_explanations_idx + j + 1
                            ]
                            if include_explanation:
                                explanation_str += (
                                    "\n\nHere is an explanation from"
                                    f" {advocate_level_str} to help you with"
                                    " your answer:\n"
                                    + extra_advocate_str
                                    + f"```{advocate_explanation['generation']}```"
                                )
                            else:
                                explanation_str += (
                                    "\n\nThe opinion of"
                                    f" {advocate_level_str} is that the"
                                    f" correct answer is `{answers[j]}`."
                                    + extra_advocate_str
                                )

                        advocate_explanations_idx += len(answers)
                        explanation_is_correct = "Invalid"

                        common[
                            "random_sample_explanations"
                        ] = random_sample_explanations

                    question_str = (
                        (
                            question.strip()
                            + "\n\nChoose one from the choices:\n"
                            + "\n".join(
                                [
                                    f"{chr(65 + i)}) {answers[j].strip()}"
                                    for i, j in enumerate(random_order)
                                ]
                            )
                        )
                        + "\n\nReply only with the letter of the correct"
                        " answer."
                        + explanation_str
                    )

                    conversation_history = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question_str},
                    ]

                    if (
                        only_correct_explanations
                        and not explanation_is_correct
                    ):
                        # messy does not always work...
                        pass
                    else:
                        data_conversations.append(
                            {
                                "conversation_history": conversation_history,
                                "explanation": advocate_explanation[
                                    "generation"
                                ]
                                if include_explanation
                                else None,
                                "include_explanation": include_explanation,
                                "explanation_level": advocate_level,
                                "explanation_is_correct": explanation_is_correct,
                                "explanation_advocate_idx": np.where(
                                    answer_idx == random_order
                                )[0][0],
                            }
                            | copy.deepcopy(common)
                        )

                    if num_explanations is not None:
                        # we have already updated advocate_explanations_idx
                        break

                    if num_few_shot is not None and i < num_few_shot:
                        # skip the remaining answers
                        advocate_explanations_idx += len(random_order) - 1
                        break
            else:
                raise ValueError(
                    f"Unknown explanation_level: {advocate_level}"
                )

    if is_advocate:
        return data_conversations

    if num_few_shot is not None:
        common_conversation_history = [
            data_conversations[0]["conversation_history"][0]
        ] + [
            x
            for i in range(num_few_shot)
            for x in [data_conversations[i]["conversation_history"][1]]
            + [
                {
                    "role": "assistant",
                    "content": data_conversations[i]["answer_str"],
                }
            ]
        ]
        for i in range(num_few_shot, len(data_conversations)):
            data_conversations[i]["conversation_history"] = copy.deepcopy(
                common_conversation_history
            ) + [
                {
                    "role": "user",
                    "content": data_conversations[i]["conversation_history"][
                        -1
                    ]["content"],
                }
            ]
        data_conversations = data_conversations[num_few_shot:]
        # random_order = np.random.permutation(list(range(len(data_conversations)))
        # )[:num_few_shot]
        # data_conversations = [data_conversations[i] for i in random_order]

    return data_conversations, choices
