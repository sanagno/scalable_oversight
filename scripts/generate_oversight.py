import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.arguments import get_oversight_args
from src.dataset_utils import get_dataset
from src.definitions import CONFIDENCES, FIELDS, SYSTEM_PROMPTS
from src.model_utils import get_model, get_model_generations
from src.utils import save_pickle, set_seed

if __name__ == "__main__":
    args = get_oversight_args()
    set_seed(args.seed)

    oversight_model, oversight_tokenizer = get_model(
        args.model_oversight, args.cache_dir, args.dtype, args.device
    )

    logger = logging.getLogger("base")

    level = "assistant"

    system_prompt = SYSTEM_PROMPTS[level].format(field=FIELDS[args.dataset])

    for confidence in CONFIDENCES:
        logger.info(f"Starting for confidence: {confidence}")

        additional_advocate_prompt = CONFIDENCES[confidence]

        dataset = get_dataset(
            args,
            args.dataset,
            args.cache_dir,
            system_prompt=system_prompt,
            advocate_level=level,
            is_advocate=True,
            num_samples=args.num_samples,
            additional_advocate_prompt=additional_advocate_prompt,
        )

        generations = get_model_generations(
            advocate_model=oversight_model,
            advocate_tokenizer=oversight_tokenizer,
            dataset=dataset,
            desc=f"Advocate Generations for {confidence}",
        )

        save_pickle(
            generations,
            os.path.join(args.oversight_data_folder, f"generations_{confidence}.pkl"),
        )
