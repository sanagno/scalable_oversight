import argparse
import logging
import sys
import os
import time
import json

import torch

from .definitions import MODELS, ADDITIONAL_SYSTEM_PROMPTS


def get_advocate_data_folder(base_data_folder, dataset, model_advocate, num_samples):
    return os.path.join(
        base_data_folder,
        "advocate_data",
        dataset + (("_" + str(num_samples)) if num_samples is not None else ""),
        model_advocate,
    )


def get_judge_args(notebook=False, notebook_args=[]):
    parser = argparse.ArgumentParser(description="LLM scalable oversight")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--model_judge",
        type=str,
        default="Llama-2-13b-chat",
        choices=MODELS,
    )
    parser.add_argument("--dtype", type=str, default="int8")
    parser.add_argument("--dataset", type=str, default="gpqa")
    parser.add_argument("--num_fewshot_samples", type=int, default=None)
    parser.add_argument("--evaluation_method", type=str, default="argmax")
    parser.add_argument("--base_logdir", type=str, default="logs")
    parser.add_argument("--base_data_folder", type=str, default="data")
    parser.add_argument("--model_advocate", type=str, default="Llama-2-13b-chat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--additional_system_prompt",
        type=str,
        default="None",
        choices=list(ADDITIONAL_SYSTEM_PROMPTS.keys()),
    )
    parser.add_argument(
        "--response_type",
        type=str,
        default="None",
        choices=list(RESPONSE_TYPES.keys()),
    )

    if notebook:
        args = parser.parse_known_args(notebook_args)[0]
    else:
        args = parser.parse_args()

    # for legacy reasons
    if args.additional_system_prompt == "None" and args.response_type == "None":
        args.log_folder = f"{args.base_logdir}/{args.model_judge}/{args.dataset}/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    elif args.additional_system_prompt != "None" and args.response_type == "None":
        args.log_folder = f"{args.base_logdir}/{args.model_judge}/{args.dataset}/{args.additional_system_prompt}/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    elif args.additional_system_prompt == "None" and args.response_type != "None":
        args.log_folder = f"{args.base_logdir}/{args.model_judge}/{args.dataset}/res{args.response_type}/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    else:
        args.log_folder = f"{args.base_logdir}/{args.model_judge}/{args.dataset}/{args.additional_system_prompt}/res{args.response_type}/{time.strftime('%Y-%m-%d-%H-%M-%S')}"

    os.makedirs(args.log_folder, exist_ok=True)
    args.logfile = f"{args.log_folder}/log.txt"

    # log args
    with open(f"{args.log_folder}/args.txt", "w") as f:
        json.dump(vars(args), f, indent=4)

    logger = logging.getLogger("base")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(args.logfile, mode="w")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    # add stdout handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

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
        default="Llama-2-13b-chat",
        choices=MODELS,
    )
    parser.add_argument("--dtype", type=str, default="int8")
    parser.add_argument("--dataset", type=str, default="gpqa")
    parser.add_argument("--base_data_folder", type=str, default="data")
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
