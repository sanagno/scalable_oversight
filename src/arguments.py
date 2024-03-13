import os
import torch
import argparse
import time
import logging
from .definitions import MODELS, APPEND_TOKENS


def get_advocate_data_folder(base_data_folder, dataset, model_advocate, num_samples):
    return os.path.join(
        base_data_folder,
        dataset + (("_" + str(num_samples)) if num_samples is not None else ""),
        model_advocate,
    )


def get_judge_args(notebook=False, notebook_args=[]):
    parser = argparse.ArgumentParser(description="LLM scalable oversight")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--model_judge",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        choices=MODELS,
    )
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--dataset", type=str, default="gpqa")
    parser.add_argument(
        "--few_shot", type=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_fewshot_samples", type=int, default=10)
    parser.add_argument("--evaluation_method", type=str, default="argmax")
    parser.add_argument("--base_logdir", type=str, default="logs")
    parser.add_argument("--base_data_folder", type=str, default="advocate_data")
    parser.add_argument(
        "--model_advocate", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)

    if notebook:
        args = parser.parse_known_args(notebook_args)[0]
    else:
        args = parser.parse_args()

    # tokens appended before checking for specific choices
    args.append_tokens = APPEND_TOKENS[args.model_judge]

    args.log_folder = (
        f"{args.base_logdir}/{args.model_judge}/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    os.makedirs(args.log_folder, exist_ok=True)
    args.logfile = f"{args.log_folder}/log.txt"

    logger = logging.getLogger("base")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(args.logfile, mode="w")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info(f"Starting run for model: {args.model_judge}")

    args.advocate_data_folder = get_advocate_data_folder(
        args.base_data_folder, args.dataset, args.model_advocate, args.num_samples
    )
    assert os.path.exists(args.advocate_data_folder)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def get_advocate_args(notebook=False, notebook_args=[]):
    parser = argparse.ArgumentParser(description="LLM scalable oversight")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--model_advocate",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        choices=MODELS,
    )
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--dataset", type=str, default="gpqa")
    parser.add_argument("--base_data_folder", type=str, default="advocate_data")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    if notebook:
        args = parser.parse_known_args(notebook_args)[0]
    else:
        args = parser.parse_args()

    args.advocate_data_folder = get_advocate_data_folder(
        args.base_data_folder, args.dataset, args.model_advocate, args.num_samples
    )
    os.makedirs(args.advocate_data_folder, exist_ok=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
