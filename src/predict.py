from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

from constants import PROMPT_TEMPLATE, RESPONSE_TEMPLATE


def predict(df, model_id, model_path):
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    output_texts = []
    template = PROMPT_TEMPLATE + RESPONSE_TEMPLATE
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        query = row.to_dict()
        input_ids = tokenizer(template.format(**query), return_tensors="pt").input_ids

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
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        output_texts += [output_text]
    return output_texts
