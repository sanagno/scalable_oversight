import logging
import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.arguments import get_advocate_args
from src.dataset_utils import get_dataset
from src.definitions import FIELDS, LEVELS, SYSTEM_PROMPTS
from src.model_utils import get_model, get_model_generations
from src.utils import save_pickle, set_seed

if __name__ == "__main__":
    args = get_advocate_args()
    set_seed(args.seed)

    advocate_model, advocate_tokenizer = get_model(
        args.model_advocate, args.cache_dir, args.dtype, args.device
    )

    logger = logging.getLogger("base")

    for level in LEVELS:
        logger.info(f"Starting for level: {level}")

        system_prompt = SYSTEM_PROMPTS[level].format(field=FIELDS[args.dataset])

        dataset = get_dataset(
            args.dataset,
            args.cache_dir,
            system_prompt=system_prompt,
            advocate_level=level,
            is_advocate=True,
            num_samples=args.num_samples,
        )

        generations = get_model_generations(
            advocate_model=advocate_model,
            advocate_tokenizer=advocate_tokenizer,
            dataset=dataset,
            desc=f"Advocate Generations for {args.dataset} {level}",
        )

        save_pickle(
            generations,
            os.path.join(args.advocate_data_folder, f"generations_{level}.pkl"),
        )
