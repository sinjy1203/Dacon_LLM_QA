PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives answers to the user's questions based only on the following context. please answer concisely without additional explanation.
### Context:
{context}
### Human:
{question}
"""

RESPONSE_TEMPLATE = "### Assistant:\n"

TRAIN_CONFIG_PATH = "../configs/train.yaml"

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

OUTPUT_DIR = "../results/"
