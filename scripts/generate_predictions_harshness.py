import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging

from src.arguments import get_judge_args
from src.dataset_utils import get_dataset
from src.definitions import (ADDITIONAL_SYSTEM_PROMPTS, FIELDS,
                             HARSHNESSES_LIST, LEVELS, RESPONSE_TYPES,
                             SYSTEM_PROMPTS)
from src.metrics import get_metrics
from src.model_utils import get_answer_probabilities, get_model
from src.utils import save_pickle, set_seed

if __name__ == "__main__":
    args = get_judge_args()
    set_seed(args.seed)

    logger = logging.getLogger("base")
    judge_model, judge_tokenizer = get_model(
        args.model_judge, args.cache_dir, args.dtype, args.device
    )

    system_prompts = {
        k: v.format(field=FIELDS[args.dataset]) for k, v in SYSTEM_PROMPTS.items()
    }
    system_prompt_name = "assistant"
    advocate_level = "assistant"
    system_prompt = system_prompts[system_prompt_name]
    extra_system_prompt = ADDITIONAL_SYSTEM_PROMPTS[args.additional_system_prompt]

    base_answer_str = RESPONSE_TYPES[args.response_type]

    for harshness_idx, harshness in enumerate(HARSHNESSES_LIST):
        advocate_level_str = advocate_level_str = LEVELS[advocate_level].format(
            field=FIELDS[args.dataset]
        )
        extra_advocate_str = harshness.format(advocate_level_str=advocate_level_str)
        # f"Note that the {advocate_level_str} is {confidence} confident about the explanation.\n"

        for include_explanation in [False, True]:
            dataset, choices = get_dataset(
                args,
                args.dataset,
                args.cache_dir,
                system_prompt=system_prompt + extra_system_prompt,
                is_advocate=False,
                num_samples=args.num_samples,
                advocate_level=advocate_level,
                num_few_shot=args.num_fewshot_samples,
                advocate_file=None
                if advocate_level not in LEVELS
                else os.path.join(
                    args.advocate_data_folder, f"generations_{advocate_level}.pkl"
                ),
                include_explanation=include_explanation,
                base_answer_str=base_answer_str,
                num_explanations=args.num_explanations,
                extra_advocate_str=extra_advocate_str,
                only_correct_explanations=True,
            )

            probabilities = get_answer_probabilities(
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                dataset=dataset,
                choices=choices,
                base_answer=base_answer_str,
            )

            save_pickle(
                probabilities,
                os.path.join(
                    args.log_folder,
                    f"probabilities_{system_prompt_name}_{advocate_level}_{include_explanation}_{harshness_idx}.pkl",
                ),
            )

            save_pickle(
                dataset,
                os.path.join(
                    args.log_folder,
                    f"dataset_{system_prompt_name}_{advocate_level}_{include_explanation}_{harshness_idx}.pkl",
                ),
            )

            metrics = get_metrics(
                advocate_level=advocate_level,
                probabilities=probabilities,
                dataset=dataset,
                evaluation_method=args.evaluation_method,
            )

            logger.info(
                f"Harshness {harshness_idx} include_explanation {include_explanation} metric is: {metrics}"
            )
