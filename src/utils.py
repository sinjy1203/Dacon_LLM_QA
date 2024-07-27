import os
import re
from collections import Counter
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from uuid import UUID
from tqdm.auto import tqdm
from langchain.schema.output import LLMResult
from langchain_core.callbacks import BaseCallbackHandler


class BatchCallback(BaseCallbackHandler):
    def __init__(self, total: int, desc):
        super().__init__()
        self.count = 0
        self.progress_bar = tqdm(total=total, desc=desc)  # define a progress bar

    # Override on_llm_end method. This is called after every response from LLM
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self.count += 1
        self.progress_bar.update(1)

    def __enter__(self):
        self.progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.progress_bar.__exit__(exc_type, exc_value, exc_traceback)

    def __del__(self):
        self.progress_bar.__del__()


def load_train_val_data(path, validate):
    np.random.seed(1203)

    df = pd.read_csv(path)

    idxs = np.arange(len(df))
    np.random.shuffle(idxs)

    train_df = df.loc[idxs]
    return train_df


def output_parsing(text):
    pattern = r"### Answer:\n(.*?)<\|im_end\|>"

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
    checkpoint_path = checkpoint_dirs[0] if checkpoint_dirs else None
    print(f"latest checkpoint path: {checkpoint_path}")
    return checkpoint_path


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
