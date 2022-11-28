import torch
import pandas as pd
import numpy as np
import pickle as pkl
from transformers import CTRLTokenizer, CTRLLMHeadModel


class DatasetClass(torch.utils.data.Dataset):
    def __init__(self, df_path):
        df = pd.read_csv(df_path)
        df = df.dropna()
        self.categories = df.category.to_list()
        self.searchterms = df.searchterm.to_list()

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, item):
        return "{} {}".format(self.categories[item], self.searchterms[item])

    def get_item_as_tuple(self, item):
        return self.categories[item], self.searchterms[item]


class CTRLTrainer:
    def __init__(self, model_name='ctrl'):
        self.tokenizer = CTRLTokenizer.from_pretrained(model_name)
        self.model = CTRLLMHeadModel.from_pretrained(model_name)

        # Add padding token which doesn't exist on CTRL
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Update the model with the new tokenizer length
        self.model.resize_token_embeddings(len(self.tokenizer))

    def tokenize_function(self, examples):
        return self.tokenizer(examples, padding="max_length", max_length=20, truncation=True, return_tensors='pt')
