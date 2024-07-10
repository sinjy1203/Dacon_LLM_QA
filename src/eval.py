import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(
    base_model, "../results/EEVE-Korean-Instruct-10.8B-v1.0/checkpoint-2044"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives answers to the user's questions based only on the following context. pleas answer concisely without additional explanation.
### Context:
{context}
### Human:
{question}
### Assistant:
"""

import re


def output_parsing(text):
    pattern = r"### Assistant:\n(.*?)<"

    # 패턴 찾기
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return "빔"


def eval(df):
    output_texts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row.to_dict()
        input_ids = tokenizer(
            prompt_template.format(**query), return_tensors="pt"
        ).input_ids

        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            eos_token_id=[
                tokenizer.convert_tokens_to_ids("#"),
                tokenizer.convert_tokens_to_ids("##"),
                tokenizer.convert_tokens_to_ids("###"),
                tokenizer.eos_token_id,
            ],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output_texts += [output_text]
    return output_texts


# submission = pd.read_csv("./data/sample_submission.csv")
# test_df = pd.read_csv("./data/test.csv")

# test_output_texts = eval(test_df)

# submission["answer"] = test_output_texts
# submission.to_csv("./data/submission.csv", index=False)


np.random.seed(1203)

df = pd.read_csv("../data/train.csv")

df["prompt"] = df.apply(
    lambda row: prompt_template.format(
        context=row.context.replace("\n\n", " ").replace("\n", " "),
        question=row.question,
    )
    + row.answer
    + "<|end_of_text|>",
    axis=1,
)

idxs = np.arange(len(df))
np.random.shuffle(idxs)

train_df, val_df = df.loc[idxs[:-1000]], df.loc[idxs[-1000:]]


val_output_texts = eval(val_df)

val_df["answer"] = val_output_texts
val_df.to_csv("../data/val_df.csv", index=False)

import wandb

wandb.login(key="d967952ddab2b90fb63bbd9e3bd309513052beec")
run = wandb.init(project="QA_finetuning", entity="sinjy1203")

artifact = wandb.Artifact(
    "evaluation",
    type="result",
    description="fine-tuned model evaluation",
    metadata={
        "model_id": "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
        "llm_type": "bfloat16",
        "prompt_template": prompt_template,
    },
)
artifact.add(wandb.Table(data=val_df), "val_df")
# artifact.add(wandb.Table(data=submission), "submission")
run.log_artifact(artifact)
run.finish()
