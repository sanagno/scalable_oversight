import numpy as np
from datasets import load_dataset


def get_dataset(
    dataset,
    cache_dir,
    system_prompt="You are a helpful assistant.",
    few_shot=False,
    explantion_level=None,
):
    assert not few_shot
    add_explanation = explantion_level is not None

    if dataset == "gpqa":
        dataset = load_dataset(
            "Idavidrein/gpqa", cache_dir=cache_dir, name="gpqa_diamond"
        )["train"]

        data_conversations = []
        # Create question and get correct answer
        for i in range(len(dataset)):
            conversation_history = [{"role": "system", "content": system_prompt}]

            question = dataset[i]["Question"]
            answers = [
                dataset[i]["Correct Answer"],
                dataset[i]["Incorrect Answer 1"],
                dataset[i]["Incorrect Answer 2"],
                dataset[i]["Incorrect Answer 3"],
            ]

            random_order = np.random.permutation(list(range(len(answers))))

            question_with_answers = (
                question.strip()
                + "\n\nChoices:\n"
                + "\n".join(
                    [
                        f"{chr(65 + i)}) {answers[j].strip()}"
                        for i, j in enumerate(random_order)
                    ]
                )
            )

            question_str = (
                question_with_answers
                + "\n\nReply only with the letter of the correct answer."
                + (
                    (explantion_level + dataset[i]["Explanation"])
                    if add_explanation
                    else ""
                )
            )

            conversation_history.append({"role": "user", "content": question_str})

            data_conversations.append(
                {
                    "conversation_history": conversation_history,
                    "correct_answer_str": chr(65 + np.where(random_order == 0)[0][0]),
                    "correct_answer_idx": np.where(random_order == 0)[0][0],
                    "explanation": dataset[i]["Explanation"],
                }
            )

            choices = [chr(65 + i) for i in range(4)]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return data_conversations, choices
