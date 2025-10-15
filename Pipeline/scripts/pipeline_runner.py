import pandas as pd
from datasets import Dataset
df = pd.read_csv("gs://llmops_project_bucket/yoda_sentences.csv")

print(df.head())

