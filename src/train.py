import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer
import wandb


model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives answers to the user's questions based only on the following context. pleas answer concisely without additional explanation.
### Context:
{context}
### Human:
{question}
### Assistant:
"""


np.random.seed(1203)

df = pd.read_csv("../data/train.csv")

df["prompt"] = df.apply(
    lambda row: prompt_template.format(
        context=row.context.replace("\n\n", " ").replace("\n", " "),
        question=row.question,
    )
    + row.answer
    + tokenizer.eos_token,
    axis=1,
)

idxs = np.arange(len(df))
np.random.shuffle(idxs)

train_df, eval_df = df.loc[idxs[:-1000]], df.loc[idxs[-1000:]]


wandb.login(key="d967952ddab2b90fb63bbd9e3bd309513052beec")
wandb.init(project="QA_finetuning", entity="sinjy1203")

target_modules = ["q_proj", "v_proj"]
r = 16

epochs = 5
batch_size = 2
gradient_accumulation_steps = 16
lr = 1e-5

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

lora_config = LoraConfig(
    r=r,
    target_modules=target_modules,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
training_args = TrainingArguments(
    run_name=model_id.split("/")[-1],
    output_dir="../results/EEVE-v2",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim="adamw_hf",
    learning_rate=lr,
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
    model_id,
    device_map="auto",
    quantization_config=nf4_config,
    attn_implementation="flash_attention_2",
)
model = get_peft_model(model, lora_config)
trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="prompt",
    max_seq_length=1024,
    args=training_args,
)

trainer.train(resume_from_checkpoint=True)
