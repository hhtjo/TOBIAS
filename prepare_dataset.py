from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import DBSCAN


df_train_raw = pd.read_csv("NEWTS_train_2400.csv")
df_test_raw = pd.read_csv("NEWTS_test_600.csv")


def rename_2_columns(df, col_names):
    for col_name in col_names:
        df.loc[
            (df["variable"] == col_name + "1") | (df["variable"] == col_name + "2"),
            "variable",
        ] = col_name
    return df


def fill_na_columns(df, col_names):
    for col_name in col_names:
        df[col_name] = df[col_name].fillna(df[col_name].dropna().reset_index(drop=True))
    return df


def unpivot_topics(df):
    df1 = pd.melt(df, id_vars=["AssignmentId", "docId", "article"])
    rename_2_columns(df1, ["tid", "words", "phrases", "sentences", "summary"])
    df1 = pd.concat(
        [df1, pd.pivot(df1, columns=["variable"], values=["value"])["value"]], axis=1
    )
    fill_na_columns(df1, ["phrases", "sentences", "summary", "words"])
    df1 = df1.dropna().drop(["variable", "value"], axis=1)
    return df1


df_train = unpivot_topics(df_train_raw)
df_test = unpivot_topics(df_test_raw)

train = Dataset.from_pandas(df_train)
test = Dataset.from_pandas(df_test)

dataset_dict = {"train": train, "test": test}

dataset_cleaned = DatasetDict(dataset_dict)
dataset_cleaned = dataset_cleaned.remove_columns(["__index_level_0__"])


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
bart = AutoModel.from_pretrained("facebook/bart-base")
embedder = bart._modules["encoder"]._modules["embed_tokens"]


def clean_prompt_outliers(example):
    prompt = " " + "".join(example["words"].split(",")).strip()
    tokenized_prompt_input_ids = torch.tensor(tokenizer(prompt)["input_ids"])
    prompt_embeds = embedder(tokenized_prompt_input_ids)

    db = DBSCAN(eps=0.75, min_samples=4, metric="cosine").fit(
        prompt_embeds.detach().numpy()
    )

    if np.count_nonzero(db.labels_ == 0) > 0:
        # If there are outliers, remove them
        new_prompt = tokenized_prompt_input_ids[db.labels_ == 0]
    else:
        new_prompt = tokenized_prompt_input_ids

    decoded_new_words = (
        tokenizer.decode(new_prompt, skip_special_tokens=True).strip().split(" ")
    )

    new_words = ", ".join(decoded_new_words)
    example["new_words"] = new_words
    example["word_diff"] = set(example["words"].split(", ")) - set(decoded_new_words)
    return example

dataset_cleaned = dataset_cleaned.map(clean_prompt_outliers)
dataset_cleaned.save_to_disk("./dataset/newts-cleaned")
