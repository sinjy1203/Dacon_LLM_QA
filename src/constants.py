# PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives answers to the user's questions based only on the following context. please answer concisely without additional explanation.
# ### Context:
# {context}
# ### Human:
# {question}
# ### Assistant:
# """

PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives reason and answer to the questions based only on the following context. The answers must be the exact words or phrases from the following context.
### Context:
{context}
### Question:
{question}
### Reason:
"""

TARGET_TEMPLATE = """{reason}
### Answer:
{answer}
"""


TRAIN_CONFIG_PATH = "../configs/train.yaml"
PREDICT_CONFIG_PATH = "../configs/predict.yaml"

TRAIN_DATA_PATH = "../data/train_cot2.csv"
TEST_DATA_PATH = "../data/test.csv"
SUBMISSION_PATH = "../data/submission"

OUTPUT_DIR = "../results/"

IGNORE_INDEX = -100
