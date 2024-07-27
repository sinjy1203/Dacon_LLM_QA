import copy
from typing import Sequence, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from utils import load_train_val_data
from constants import PROMPT_TEMPLATE, IGNORE_INDEX, TARGET_TEMPLATE


class QADataset(Dataset):
    def __init__(self, df, tokenizer):
        super().__init__()

        sources = df.apply(
            lambda row: PROMPT_TEMPLATE.format(
                context=row.context.replace("\n\n", " ").replace("\n", " "),
                question=row.question,
            ),
            axis=1,
        ).to_list()
        targets = df.apply(
            lambda row: TARGET_TEMPLATE.format(reason=row.reason, answer=row.answer)
            + tokenizer.eos_token,
            axis=1,
        ).to_list()

        sr_tgs = [s + t for s, t in zip(sources, targets)]
        sources_tokenized = self._tokenize_fn(sources, tokenizer)
        sr_tgs_tokenized = self._tokenize_fn(sr_tgs, tokenizer)

        input_ids = sr_tgs_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        self.input_ids, self.labels = input_ids, labels

    def _tokenize_fn(self, texts, tokenizer):
        tokenized_texts = [
            tokenizer(
                text,
                padding="longest",
                return_tensors="pt",
            )
            for text in texts
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_texts]
        input_ids_lens = labels_lens = [len(input_id) for input_id in input_ids]

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class QADataCollator(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": instance["input_ids"]} for instance in instances]
        labels = [{"input_ids": instance["labels"]} for instance in instances]
        batch_data = self.tokenizer.pad(input_ids)
        batch_data["labels"] = self.tokenizer.pad(labels)["input_ids"]
        batch_data["labels"][
            batch_data["labels"] == self.tokenizer.pad_token_id
        ] = IGNORE_INDEX
        return batch_data


if __name__ == "__main__":
    train_df, val_df = load_train_val_data("../data/train.csv")
    tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
    train_dataset = QADataset(train_df, tokenizer=tokenizer)
    data_collator = QADataCollator(tokenizer=tokenizer)
    batch_data = data_collator(train_dataset[:10])
    print()
