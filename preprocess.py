import pandas as pd
from datasets import load_from_disk, load_dataset
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import re
import spacy

spacy_model = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
dataset = load_dataset("roneneldan/TinyStories")


def process(subset):
    texts = subset["text"]
    tokens = []
    for doc in spacy_model.pipe(texts, batch_size=512 * 4, n_process=1):
        tokens.append([token.text.lower() for token in doc if token.text.isprintable() and not token.is_space])
    return {"tokens": tokens}


dataset["train"] = dataset["train"].map(
    process, batched=True, batch_size=512 * 4, num_proc=8
)
dataset["validation"] = dataset["validation"].map(
    process, batched=True, batch_size=512 * 4, num_proc=8
)

dataset.save_to_disk("TinyStories_processed")