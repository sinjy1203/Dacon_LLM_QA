import os
import yaml
import argparse
import torch

from torch.utils.data import DataLoader
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import DataCollatorWithPadding
from transformers import Trainer

from predict import predict
from utils import (
    load_train_data,
    output_parsing,
    get_latest_checkpoint,
    calculate_f1_score,
)
from constants import TRAIN_CONFIG_PATH, OUTPUT_DIR, TRAIN_DATA_PATH
from dataset import QADataset, QADataCollator
from constants import PROMPT_TEMPLATE, TARGET_TEMPLATE

torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM")
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    train_df = load_train_data(TRAIN_DATA_PATH)

    train_dataset = QADataset(train_df, tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    lora_config = LoraConfig(
        r=config["lora_r"],
        target_modules=config["lora_target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        run_name=config["run_name"],
        output_dir=OUTPUT_DIR + config["run_name"],
        save_strategy="epoch",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        optim="paged_adamw_8bit",
        learning_rate=config["lr"],
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        # group_by_length=True,
        lr_scheduler_type="linear",
        do_eval=False,
        evaluation_strategy="no",
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        device_map="auto",
        quantization_config=nf4_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = get_peft_model(model, lora_config)

    # model = AutoModelForCausalLM.from_pretrained(
    #     config["model_id"],
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    #     # attn_implementation="flash_attention_2",
    #     # attn_implementation='eager'
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train(resume_from_checkpoint=config["resume"])


if __name__ == "__main__":
    args = get_args()
    if args.sweep:
        pass
    else:
        with open(TRAIN_CONFIG_PATH) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        main(config)
