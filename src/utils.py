import os
import re
from collections import Counter
import numpy as np
import pandas as pd

from constants import PROMPT_TEMPLATE, RESPONSE_TEMPLATE


def load_train_val_data(path, tokenizer):
    np.random.seed(1203)

    df = pd.read_csv(path)
    df["prompt"] = df.apply(
        lambda row: PROMPT_TEMPLATE.format(
            context=row.context.replace("\n\n", " ").replace("\n", " "),
            question=row.question,
        )
        + RESPONSE_TEMPLATE
        + row.answer
        + tokenizer.eos_token,
        axis=1,
    )

    idxs = np.arange(len(df))
    np.random.shuffle(idxs)

    train_df, val_df = df.loc[idxs[:-1000]], df.loc[idxs[-1000:]]
    return train_df, val_df


def output_parsing(text):
    pattern = r"### Assistant:\n(.*?)<\|im_end\|>"

    # 패턴 찾기
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return "ㅂ"


def get_latest_checkpoint(checkpoints_dir):
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")

    checkpoint_dirs = [
        os.path.join(checkpoints_dir, d)
        for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d))
        and checkpoint_pattern.match(d)
    ]

    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda x: int(checkpoint_pattern.search(os.path.basename(x)).group(1)),
        reverse=True,
    )

    return checkpoint_dirs[0] if checkpoint_dirs else None


def calculate_f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


if __name__ == "__main__":
    print(get_latest_checkpoint("../results/eeve-v2"))
