# PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives answers to the user's questions based only on the following context. please answer concisely without additional explanation.
# ### Context:
# {context}
# ### Human:
# {question}
# ### Assistant:
# """

# PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives reason and answer to the questions based only on the following context. The answers must be the exact words or phrases from the following context.

# ### Context:
# {context}

# ### Question:
# {question}

# ### Reason:
# """

# TARGET_TEMPLATE = """{reason}

# ### Answer:
# {answer}"""

PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives answer to the questions based only on the following context. The answers must be the exact words or phrases from the following context.

### Context:
{context}

### Question:
{question}

### Answer:
"""

TARGET_TEMPLATE = """{answer}"""

# PROMPT_TEMPLATE = """<start_of_turn>user
# 다음 뉴스기사 내용을 바탕으로 주어진 질문에 대한 답과 근거를 알려줘. 반드시 뉴스기사 내용에 있는 단어나 구로 답변해야해.

# 뉴스기사 내용:
# {context}

# 질문:
# {question}<end_of_turn>
# <start_of_turn>model
# """

# TARGET_TEMPLATE = """근거:
# {reason}

# 답:
# {answer}<end_of_turn>"""


TRAIN_CONFIG_PATH = "../configs/train.yaml"
PREDICT_CONFIG_PATH = "../configs/predict.yaml"

# TRAIN_DATA_PATH = "../data/train_cot2.csv"
TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"
SUBMISSION_PATH = "../data/submission"

OUTPUT_DIR = "../results/"

IGNORE_INDEX = -100
