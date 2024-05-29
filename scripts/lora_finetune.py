import argparse
import torch
import os
import pickle
import sys
import pandas as pd
import datasets

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.arguments import get_advocate_data_folder
from src.definitions import RESPONSE_TYPES
from src.dataset_utils import get_dataset
from src.model_utils import get_model

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    PreTrainedTokenizer,
)
from transformers.data.data_collator import DataCollatorMixin
from trl import SFTTrainer
from dataclasses import dataclass


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        examples = []

        for idx in range(len(dataset)):
            conversation = dataset[idx]["conversation_history"]
            conversation.append(
                {"role": "assistant", "content": dataset[idx]["answer_str"]}
            )
            examples.append(tokenizer.apply_chat_template(conversation))

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class DataCollatorForLastTokenPrediction(DataCollatorMixin):
    tokenizer: PreTrainedTokenizer
    return_tensors = "pt"

    def torch_call(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        # Tensorize if necessary.

        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Creating the full tensor and filling it with our data.
        lengths = [e.size(0) for e in examples]
        max_length = max(x.size(0) for x in examples)

        examples = [
            torch.nn.functional.pad(
                x,
                (0, max_length - x.size(0)),
                value=self.tokenizer.pad_token_id,
            )
            for x in examples
        ]

        input_ids = torch.stack(examples, dim=0)
        labels = input_ids.clone()
        labels = torch.roll(labels, -1, -1)

        for i, l in enumerate(lengths):
            labels[i, : l - 4] = -100
            labels[i, l - 3 :] = -100

        return {"input_ids": input_ids, "labels": labels}


def main(args):
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,  # TODO was True
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading model")
    judge_model, judge_tokenizer = get_model(
        "Llama-2-7b-chat",
        None,
        "float16",
        torch.device("cuda:0"),
        vllm=False,
        **{
            "quantization_config": bnb_config,
            "torch_dtype": "auto",
        },
    )
    print(judge_model)
    judge_model.config.pretraining_tp = 1

    judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_tokenizer.padding_side = "right"

    print("Loading Dataset")
    base_answer_str = RESPONSE_TYPES[args.response_type]
    raw_dataset, _ = get_dataset(
        args,
        args.dataset,
        None,
        system_prompt="You are a helpful assistant.",
        is_advocate=False,
        num_samples=None,
        advocate_level=args.advocate_level,
        num_few_shot=None,
        advocate_file=os.path.join(
            get_advocate_data_folder(
                args.base_data_folder,
                args.dataset,
                args.model_advocate,
                args.num_samples,
            ),
            f"generations_{args.advocate_level}.pkl",
        ),
        include_explanation=True,
        base_answer_str=base_answer_str,
        num_explanations=None,
    )

    print("Preprocessing Dataset")
    dataset_examples = CustomDataset(raw_dataset, judge_tokenizer)

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "lm_head",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # prepare model for training
    judge_model = prepare_model_for_kbit_training(judge_model)
    judge_model = get_peft_model(judge_model, peft_config)

    results_dir = f"{args.save_dir}/summarization_epochs-{args.epochs}_rank-{args.lora_r}"

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=50,
        learning_rate=2e-5,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        save_steps=50,
    )

    max_seq_length = (
        4096  # max sequence length for model and packing of the dataset
    )

    trainer = SFTTrainer(
        model=judge_model,
        train_dataset=dataset_examples,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=judge_tokenizer,
        packing=False,
        args=training_args,
        data_collator=DataCollatorForLastTokenPrediction(
            tokenizer=judge_tokenizer
        ),
        num_of_sequences=1,  # do not pack sequences ...
        formatting_func=lambda x: x,  # do nothing. require by the trainer ...
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    judge_tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--save_dir", default="lora_finetuned_ckpt", type=str)
    args = parser.parse_args()

    args.dropout = 0.0
    args.base_data_folder = "data"
    args.dataset = "commonsense_qa-train"
    args.model_advocate = "Llama-2-7b-chat"
    args.num_samples = None
    args.response_type = "None"
    args.advocate_level = "assistant"

    main(args)
