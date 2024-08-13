from tqdm import tqdm
import yaml
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

from constants import PROMPT_TEMPLATE, PREDICT_CONFIG_PATH, TEST_DATA_PATH, OUTPUT_DIR
from utils import output_parsing, get_latest_checkpoint


def predict(df, model_id, model_path):
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    output_texts = []
    template = PROMPT_TEMPLATE
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        query = row.to_dict()
        input_ids = tokenizer(template.format(**query), return_tensors="pt").input_ids
        input_ids = input_ids.to(torch.device("cuda:0"))

        outputs = model.generate(
            input_ids,
            max_length=4000,
            eos_token_id=[
                tokenizer.eos_token_id,
            ],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

        # outputs = model.generate(
        #     input_ids,
        #     max_length=4000,
        #     eos_token_id=[
        #         tokenizer.eos_token_id,
        #     ],
        #     pad_token_id=tokenizer.pad_token_id,
        #     num_beams=3,
        #     num_return_sequences=1,
        #     remove_invalid_values=True,
        # )

        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        output_texts += [output_text]
    return output_texts


def main(config):
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["context"] = test_df["context"].apply(
        lambda x: x.replace("\n\n", " ").replace("\n", " ").replace("  ", " ")
    )
    preds = predict(
        test_df,
        config["model_id"],
        get_latest_checkpoint(OUTPUT_DIR + config["run_name"]),
    )
    test_df["answer"] = preds
    test_df.to_csv(OUTPUT_DIR + config["run_name"] + "/test_df.csv", index=False)
    test_df["answer"] = test_df["answer"].apply(output_parsing)
    test_df = test_df.drop(["context", "question"], axis=1)
    test_df.to_csv(OUTPUT_DIR + config["run_name"] + "/submission.csv", index=False)


if __name__ == "__main__":
    with open(PREDICT_CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
