import pandas as pd
from datasets import Dataset, DatasetDict
df = pd.read_csv("gs://llmops_project_bucket/yoda_sentences.csv")

dataset = Dataset.from_pandas(df)

def convert_to_conversation(example):
    return {
        "messages": [
            {"role": "user", "content": example["sentence"]},
            {"role": "assistant", "content": example["translation"]},
            {"role": "assistant", "content": example["translation_extra"]},
        ]
    }

formatted_dataset = dataset.map(convert_to_conversation)

splits = formatted_dataset.train_test_split(test_size=0.1, seed=42)

train_df = splits["train"].to_pandas()
test_df = splits["test"].to_pandas()

train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)


