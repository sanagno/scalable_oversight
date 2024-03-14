import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging

from src.arguments import get_judge_args
from src.dataset_utils import get_dataset
from src.model_utils import get_model, get_answer_probabilities
from src.utils import save_pickle, set_seed
from src.metrics import get_metrics
from src.definitions import (
    FIELDS,
    SYSTEM_PROMPTS,
    POSSIBLE_ADVOCATES,
    LEVELS,
    HAS_DATASET_EXPLANATIONS,
)
import logging
from tqdm import tqdm


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

    for system_prompt_name, system_prompt in tqdm(
        system_prompts.items(), desc="System prompts"
    ):
        for advocate_level, include_explanation in POSSIBLE_ADVOCATES:
            if (
                advocate_level == "dataset"
                and not HAS_DATASET_EXPLANATIONS[args.dataset]
            ):
                continue

            dataset, choices = get_dataset(
                args,
                args.dataset,
                args.cache_dir,
                system_prompt=system_prompt,
                is_advocate=False,
                num_samples=args.num_samples,
                advocate_level=advocate_level,
                advocate_file=None
                if advocate_level not in LEVELS
                else os.path.join(
                    args.advocate_data_folder, f"generations_{advocate_level}.pkl"
                ),
                include_explanation=include_explanation,
            )

            probabilities = get_answer_probabilities(
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                dataset=dataset,
                choices=choices,
                append_tokens=args.append_tokens,
            )

            save_pickle(
                probabilities,
                os.path.join(
                    args.log_folder,
                    f"probabilities_{system_prompt_name}_{advocate_level}_{include_explanation}.pkl",
                ),
            )

            metrics = get_metrics(
                dataset_name=args.dataset,
                advocate_level=advocate_level,
                probabiltiies=probabilities,
                dataset=dataset,
                evaluation_method=args.evaluation_method,
            )

            logger.info(
                f"System prompt {system_prompt_name} advocate level {advocate_level} include_explanation {include_explanation} metric is: {metrics}"
            )
