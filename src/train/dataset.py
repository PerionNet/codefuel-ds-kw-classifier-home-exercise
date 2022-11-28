import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import os


class KWClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.data = data_dict


    def __getitem__(self, idx):
        input_ids = self.data['input_ids'][idx]
        token_type_ids = self.data['token_type_ids'][idx]
        attention_mask = self.data['attention_mask'][idx]
        y_h1 = self.data['h1_label'][idx]
        y_h2 = self.data['h2_label'][idx]
        y_h3 = self.data['h3_label'][idx]
        y_depth = self.data['depth_label'][idx]
        category = self.data['category_id'][idx]
        # Change to put in config

        return input_ids, token_type_ids, attention_mask, y_h1, y_h2, y_h3, y_depth, category

    def __len__(self):
        return len(self.data['input_ids'])


class KWClassifierDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds, batch_size=64, weight_arr=None):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        if weight_arr is not None:
            self.sampler = torch.utils.data.WeightedRandomSampler(weight_arr.type('torch.DoubleTensor'), len(weight_arr))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)#, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)#, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)#, num_workers=os.cpu_count())











