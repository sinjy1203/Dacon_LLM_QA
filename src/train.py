import os
import yaml
import argparse
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import DataCollatorWithPadding
from transformers import Trainer
import wandb

from utils import (
    load_train_val_data,
)
from constants import TRAIN_CONFIG_PATH, OUTPUT_DIR, TRAIN_DATA_PATH
from dataset import QADataset

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM")
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def main(config):
    wandb.login(key="d967952ddab2b90fb63bbd9e3bd309513052beec")
    wandb.init(project="QA_finetuning", entity="sinjy1203", name=config["run_name"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    train_df = load_train_val_data(TRAIN_DATA_PATH)

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
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
        report_to="wandb",
        do_eval=False,
        evaluation_strategy="no",
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        device_map="auto",
        quantization_config=nf4_config,
        attn_implementation="flash_attention_2",
    )
    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train(resume_from_checkpoint=config["resume"])

    wandb.finish()


if __name__ == "__main__":
    args = get_args()
    if args.sweep:
        pass
    else:
        with open(TRAIN_CONFIG_PATH) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        main(config)
