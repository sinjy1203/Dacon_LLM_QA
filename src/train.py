import os
import yaml
import argparse
import torch

# from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# from trl import SFTTrainer
from transformers import Trainer
import wandb

from predict import predict
from utils import (
    load_train_val_data,
    output_parsing,
    get_latest_checkpoint,
    calculate_f1_score,
)
from constants import TRAIN_CONFIG_PATH, OUTPUT_DIR, TRAIN_DATA_PATH
from dataset import QADataset, QADataCollator

os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM")
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def main(config):
    wandb.login(key="d967952ddab2b90fb63bbd9e3bd309513052beec")
    wandb.init(project="QA_finetuning", entity="sinjy1203", name=config["run_name"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    train_df, val_df = load_train_val_data(TRAIN_DATA_PATH)
    # train_dataset, val_dataset = Dataset.from_pandas(train_df), Dataset.from_pandas(
    #     val_df
    # )
    train_dataset = QADataset(train_df, tokenizer=tokenizer)
    val_dataset = QADataset(val_df, tokenizer=tokenizer)
    data_collator = QADataCollator(tokenizer=tokenizer)

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
        evaluation_strategy="epoch",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        optim="adamw_hf",
        learning_rate=config["lr"],
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
        report_to="wandb",
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

    # trainer = SFTTrainer(
    #     model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     dataset_text_field="prompt",
    #     max_seq_length=1024,
    #     args=training_args,
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train(resume_from_checkpoint=config["resume"])

    val_df["response"] = predict(
        val_df,
        config["model_id"],
        get_latest_checkpoint(OUTPUT_DIR + config["run_name"]),
    )
    val_df.to_csv(OUTPUT_DIR + config["run_name"] + "/val_df.csv", index=False)
    val_df["response"] = val_df["response"].apply(output_parsing)
    val_df["f1"] = val_df.apply(
        lambda row: calculate_f1_score(row.response, row.answer), axis=1
    )

    wandb.log({"val_f1": val_df["f1"].mean(), "val_dxf": val_df} | config)
    wandb.finish()


if __name__ == "__main__":
    args = get_args()
    if args.sweep:
        pass
    else:
        with open(TRAIN_CONFIG_PATH) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        main(config)
