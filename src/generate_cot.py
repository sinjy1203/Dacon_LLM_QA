import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import BatchCallback

load_dotenv()

temperature = 0.1

llm = ChatOpenAI(
    base_url=os.getenv("VLLM_BASE_URL"),
    openai_api_key="dummy",
    model="Qwen/Qwen2-72B-Instruct",
    temperature=temperature,
    max_tokens=4096,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.",
        ),
        (
            "user",
            """### context ###
{context}

### question ###
{question}

### answer ###
{answer}

### instruction ###
위에 question에 대한 answer를 하기 위해 context에서 어떤 부분을 참고해야 하는지 근거를 간결하게 말해주세요.""",
        ),
    ]
)

chain = prompt | llm | StrOutputParser()


def main():
    train_df = pd.read_csv("../data/train.csv")
    train_df["context"] = train_df["context"].apply(
        lambda x: x.strip()
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace("     ", " ")
    )
    train_df = train_df.iloc[20000:]

    queries = train_df.to_dict(orient="records")

    with BatchCallback(len(queries), f"gen_reason") as cb:
        reasons = chain.batch(
            queries, config={"max_concurrency": 16, "callbacks": [cb]}
        )
    train_df["reason"] = reasons
    train_df.to_csv("../data/train_cot_2.csv", index=False)


if __name__ == "__main__":
    main()
