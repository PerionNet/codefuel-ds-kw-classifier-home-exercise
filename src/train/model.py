import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch import nn
import pandas as pd
import numpy as np
import boto3
import pickle
import logging
import os
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup, BertTokenizerFast
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE

from src.create_data.utils import parse_taxonomy_dict, get_mapping_indicator_matrix


class BaseModel(nn.Module):
    def __init__(self, n_h1=27, n_h2=218, n_h3=384, p=0.2):
        super(BaseModel, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.h1 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(p), nn.Linear(768, n_h1))
        self.h2 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(p), nn.Linear(768, n_h2))
        self.h3 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(p), nn.Linear(768, n_h3))


class KWClassifier(pl.LightningModule):
    def __init__(self, n_h1=27, n_h2=218, n_h3=384, h1_labels_mapping_path="", h2_labels_mapping_path="",
                 h3_labels_mapping_path="",
                 h1_to_h2_mapping_path="", h2_to_h3_mapping_path="", taxonomy_path="", h1_weights=None, p=0.2,
                 device_type='cuda', is_negative_class=False, temperature_h1_path="", temperature_h2_path="",
                 temperature_h3_path=""):
        super().__init__()

        with open(temperature_h1_path, 'rb') as f:
            self.temperature_h1 = pickle.load(f)
        with open(temperature_h2_path, 'rb') as f:
            self.temperature_h2 = pickle.load(f)
        with open(temperature_h3_path, 'rb') as f:
            self.temperature_h3 = pickle.load(f)

        if is_negative_class:
            self.model = BaseModel(n_h1, n_h2 + 1, n_h3 + 1, p=p)
        else:
            self.model = BaseModel(n_h1, n_h2, n_h3, p=p)

        self.h1_labels_mapping_dict = pd.read_csv(h1_labels_mapping_path).to_dict()
        self.h2_labels_mapping_dict = pd.read_csv(h2_labels_mapping_path).to_dict()
        self.h3_labels_mapping_dict = pd.read_csv(h3_labels_mapping_path).to_dict()
        self.h1_to_h2_mapping_dict = pd.read_csv(h1_to_h2_mapping_path).to_dict()
        self.h2_to_h3_mapping_dict = pd.read_csv(h2_to_h3_mapping_path).to_dict()
        self.h1_h2_indicator_matrix, self.h2_h3_indicator_matrix = get_mapping_indicator_matrix(
            self.h1_labels_mapping_dict,
            self.h2_labels_mapping_dict,
            self.h3_labels_mapping_dict,
            self.h1_to_h2_mapping_dict,
            self.h2_to_h3_mapping_dict,
            is_negative_class=is_negative_class)
        self.device_type = device_type
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.taxonomy_dict = parse_taxonomy_dict(taxonomy_path)
        self.inverse_taxonomy_dict = {v: k for k, v in self.taxonomy_dict.items()}
        self.loss_h1 = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.sm = nn.Softmax(dim=1)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=200000000,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, input_ids, token_type_ids, attention_mask):
        logits = self.model.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask).pooler_output
        h1_logits = self.model.h1(logits)
        h2_logits = self.model.h2(logits)
        h3_logits = self.model.h3(logits)
        # depth_classifier_logits = self.model.depth_classifier(logits)
        return h1_logits, h2_logits, h3_logits

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        input_ids, token_type_ids, attention_mask, y1, y2, y3, y_depth, _ = train_batch
        h1_logits, h2_logits, h3_logits = self.forward(input_ids, token_type_ids,
                                                       attention_mask)
        loss_h1 = self.loss_h1(h1_logits, y1)
        loss_h2 = self.loss(h2_logits, y2)
        loss_h3 = self.loss(h3_logits, y3)
        # loss_depth = self.loss(depth_classifier_logits, y_depth)
        total_loss = loss_h1 + loss_h2 + loss_h3
        self.log("loss_h1_train", loss_h1)
        self.log("loss_h2_train", loss_h2)
        self.log("loss_h3_train", loss_h3)
        # self.log("loss_depth_train", loss_depth)
        self.log("train_loss_train", total_loss)
        return {"loss": total_loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """
        input_ids, token_type_ids, attention_mask, y1, y2, y3, y_depth, category_id = val_batch
        h1_logits, h2_logits, h3_logits = self.forward(input_ids, token_type_ids,
                                                       attention_mask)

        loss_h1 = self.loss(h1_logits, y1)
        loss_h2 = self.loss(h2_logits, y2)
        loss_h3 = self.loss(h3_logits, y3)
        # loss_depth = self.loss(depth_classifier_logits, y_depth)
        total_loss = loss_h1 + loss_h2 + loss_h3
        self.log("loss_h1_val", loss_h1)
        self.log("loss_h2_val", loss_h2)
        self.log("loss_h3_val", loss_h3)
        # self.log("loss_depth_val", loss_depth)
        self.log("train_loss_val", total_loss)

        return {"val_step_loss": total_loss, 'h1_logits': h1_logits, 'h2_logits': h2_logits, 'h3_logits': h3_logits,
                'y1': y1, 'y2': y2, 'y3': y3, 'y_depth': y_depth,
                'category_id': category_id}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy

        :param outputs: outputs after every epoch end

        :return: output - average valid loss
        """
        # Stack batches
        h1_logits_raw = torch.cat([x["h1_logits"] for x in outputs])
        h1_logits = torch.argmax(h1_logits_raw, dim=1)
        h2_logits_raw = torch.cat([x["h2_logits"] for x in outputs])
        h2_logits = torch.argmax(h2_logits_raw, dim=1)
        h3_logits_raw = torch.cat([x["h3_logits"] for x in outputs])
        h3_logits = torch.argmax(h3_logits_raw, dim=1)
        # depth_classifier_logits = torch.cat([x["depth_classifier_logits"] for x in outputs])
        # depth_classifier_logits = torch.argmax(depth_classifier_logits, dim=1)

        y1 = torch.cat([x["y1"] for x in outputs])
        y2 = torch.cat([x["y2"] for x in outputs])
        y3 = torch.cat([x["y3"] for x in outputs])
        # y_depth = torch.cat([x["y_depth"] for x in outputs])

        h1_logits = h1_logits.cpu().detach().numpy()
        h2_logits = h2_logits.cpu().detach().numpy()
        h3_logits = h3_logits.cpu().detach().numpy()
        # depth_classifier_logits = depth_classifier_logits.cpu().detach().numpy()

        y1 = y1.cpu().detach().numpy()
        y2 = y2.cpu().detach().numpy()
        y3 = y3.cpu().detach().numpy()
        # y_depth = y_depth.cpu().detach().numpy()
        epoch_str = str(self.current_epoch)
        # Get the accuracy metric
        h1_preds, h2_preds, h3_preds = self.calculate_classifiers_preds(h1_logits_raw, h2_logits_raw, h3_logits_raw)

        # H1
        h1_acc = accuracy_score(y1, h1_preds)
        h1_recall = recall_score(y1, h1_preds, average='macro')
        h1_prec = precision_score(y1, h1_preds, average='macro')
        h1_f1 = f1_score(y1, h1_preds, average='macro')
        self.log("h1_accuracy_val", h1_acc)
        self.log("h1_recall_val", h1_recall)
        self.log("h1_precision_val", h1_prec)
        self.log("h1_f1_val", h1_f1)

        # H2
        h2_acc = accuracy_score(y2, h2_preds)
        h2_recall = recall_score(y2, h2_preds, average='macro')
        h2_prec = precision_score(y2, h2_preds, average='macro')
        h2_f1 = f1_score(y2, h2_preds, average='macro')
        self.log("h2_accuracy_val", h2_acc)
        self.log("h2_recall_val", h2_recall)
        self.log("h2_precision_val", h2_prec)
        self.log("h2_f1_val", h2_f1)

        # H3
        h3_acc = accuracy_score(y3, h3_preds)
        h3_recall = recall_score(y3, h3_preds, average='macro')
        h3_prec = precision_score(y3, h3_preds, average='macro')
        h3_f1 = f1_score(y3, h3_preds, average='macro')
        self.log("h3_accuracy_val", h3_acc)
        self.log("h3_recall_val", h3_recall)
        self.log("h3_precision_val", h3_prec)
        self.log("h3_f1_val", h3_f1)

        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)

    def calculate_classifiers_preds(self, h1_logits, h2_logits, h3_logits):
        h1_preds = h1_logits.argmax(1)
        h1_h2_batch_mapping = self.h1_h2_indicator_matrix[h1_preds]
        res = h2_logits.clone()
        res[h1_h2_batch_mapping != 1] = -torch.inf
        h2_preds = res.argmax(1)
        h2_h3_batch_mapping = self.h2_h3_indicator_matrix[h2_preds]
        res = h3_logits.clone()
        res[h2_h3_batch_mapping != 1] = -torch.inf
        h3_preds = res.argmax(1)
        return h1_preds, h2_preds, h3_preds

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        input_ids, token_type_ids, attention_mask, y1, y2, y3, y_depth, category_id = test_batch
        h1_logits, h2_logits, h3_logits = self.forward(input_ids, token_type_ids,
                                                       attention_mask)
        # Get argmax from depth

        # Start checking it per example, based on length of prediction

        # Use mappings from h1 -> h2 or h2 -> h3 to see if the mapping exists, or we need to have a stop.

        # Define accuracy metric for partial correctness aswell

        return {'h1_logits': h1_logits, 'h2_logits': h2_logits, 'h3_logits': h3_logits,
                'y1': y1, 'y2': y2, 'y3': y3, 'y_depth': y_depth,
                'category_id': category_id}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        :param outputs: outputs after every epoch end
        :return: output - average test loss
        """
        # Stack batches
        h1_logits_raw = torch.cat([x["h1_logits"] for x in outputs])
        h1_logits = torch.argmax(h1_logits_raw, dim=1)
        h2_logits_raw = torch.cat([x["h2_logits"] for x in outputs])
        h2_logits = torch.argmax(h2_logits_raw, dim=1)
        h3_logits_raw = torch.cat([x["h3_logits"] for x in outputs])
        h3_logits = torch.argmax(h3_logits_raw, dim=1)

        y1 = torch.cat([x["y1"] for x in outputs])
        y2 = torch.cat([x["y2"] for x in outputs])
        y3 = torch.cat([x["y3"] for x in outputs])
        self.calibrate_model(h1_logits_raw, h2_logits_raw, h3_logits_raw, y1, y2, y3)
        h1_logits = h1_logits.cpu().detach().numpy()
        h2_logits = h2_logits.cpu().detach().numpy()
        h3_logits = h3_logits.cpu().detach().numpy()

        y1 = y1.cpu().detach().numpy()
        y2 = y2.cpu().detach().numpy()
        y3 = y3.cpu().detach().numpy()
        # Get the accuracy metric
        h1_preds, h2_preds, h3_preds = self.calculate_classifiers_preds(h1_logits_raw, h2_logits_raw, h3_logits_raw)
        h1_preds = h1_preds.cpu().detach().numpy()
        h2_preds = h2_preds.cpu().detach().numpy()
        h3_preds = h3_preds.cpu().detach().numpy()
        # H1
        h1_acc = accuracy_score(y1, h1_preds)
        h1_recall = recall_score(y1, h1_preds, average='macro')
        h1_prec = precision_score(y1, h1_preds, average='macro')
        h1_f1 = f1_score(y1, h1_preds, average='macro')
        self.log("h1_accuracy_test", h1_acc)
        self.log("h1_recall_test", h1_recall)
        self.log("h1_precision_test", h1_prec)
        self.log("h1_f1_test", h1_f1)

        # H2
        h2_acc = accuracy_score(y2, h2_preds)
        h2_recall = recall_score(y2, h2_preds, average='macro')
        h2_prec = precision_score(y2, h2_preds, average='macro')
        h2_f1 = f1_score(y2, h2_preds, average='macro')
        self.log("h2_accuracy_test", h2_acc)
        self.log("h2_recall_test", h2_recall)
        self.log("h2_precision_test", h2_prec)
        self.log("h2_f1_test", h2_f1)

        # H3
        h3_acc = accuracy_score(y3, h3_preds)
        h3_recall = recall_score(y3, h3_preds, average='macro')
        h3_prec = precision_score(y3, h3_preds, average='macro')
        h3_f1 = f1_score(y3, h3_preds, average='macro')
        self.log("h3_accuracy_test", h3_acc)
        self.log("h3_recall_test", h3_recall)
        self.log("h3_precision_test", h3_prec)
        self.log("h3_f1_test", h3_f1)

    def predict(self, x):
        """
        param x: List of input strings to check categories [Str1, Str2, Str3 ... StrN]
        :return: List of outputs categories and categories ID and confidence [(Cat1, Id1, Conf1) ... (CatN, IdN, ConfN)]
        """
        tokenized_x = self.tokenizer(x, padding="max_length", max_length=20, truncation=True,
                                     return_tensors='pt')
        input_ids = tokenized_x['input_ids']
        token_type_ids = tokenized_x['token_type_ids']
        attention_mask = tokenized_x['attention_mask']
        if torch.cuda.is_available():
            input_ids.to('cuda')
            token_type_ids.to('cuda')
            attention_mask.to('cuda')
        h1_logits, h2_logits, h3_logits = self.forward(input_ids, token_type_ids,
                                                       attention_mask)
        h1_preds, h2_preds, h3_preds = self.calculate_classifiers_preds(h1_logits, h2_logits, h3_logits)
        h1_preds = h1_preds.cpu().detach().numpy()
        h2_preds = h2_preds.cpu().detach().numpy()
        h3_preds = h3_preds.cpu().detach().numpy()

        h1_logits_sm = self.sm(h1_logits)
        h2_logits_sm = self.sm(h2_logits)
        h3_logits_sm = self.sm(h3_logits)
        calibrated_h1 = self.temperature_h1.transform(h1_logits_sm.detach().cpu().numpy())
        calibrated_h2 = self.temperature_h2.transform(h2_logits_sm.detach().cpu().numpy())
        calibrated_h3 = self.temperature_h3.transform(h3_logits_sm.detach().cpu().numpy())
        calibrated_h1 = calibrated_h1.max(1)
        calibrated_h2 = calibrated_h2.max(1)
        calibrated_h3 = calibrated_h3.max(1)
        res = np.multiply(np.multiply(calibrated_h1, calibrated_h2), calibrated_h3)
        str_res = [str(x) for x in res]
        h1_category_dict = self.h1_labels_mapping_dict['category']
        h2_category_dict = self.h2_labels_mapping_dict['category']
        h3_category_dict = self.h3_labels_mapping_dict['category']

        # Handling empty classes
        h2_category_dict[218] = ""
        h3_category_dict[384] = ""
        str_h1 = [h1_category_dict[p] for p in h1_preds]
        str_h2 = [h2_category_dict[p] for p in h2_preds]
        str_h3 = [h3_category_dict[p] for p in h3_preds]
        final_list = [str_h1, str_h2, str_h3]
        final_list_T = np.array(final_list).T.tolist()
        final_preds = [max(j, key=len) for j in final_list_T]
        return list(zip(final_preds, str_res))

    def calibrate_model(self, h1_logits, h2_logits, h3_logits, y1, y2, y3, n_bins=10):
        """
        This method calibrates our classifiers using temperature in the accuracy / confidence plane, to perform unbiased confidence predictions.
        The method to measure the miscalibration called ECE (Expected calibration error)
        :param h1_logits: H1 Logits
        :param h2_logits: H2 Logits
        :param h3_logits: H3 Logits
        :param y1: List of sparse categorical labels of h1 (1xn)
        :param y2: List of sparse categorical labels of h2 (1xn)
        :param y3: List of sparse categorical labels of h3 (1xn)
        :param n_bins: Number of bins to split ECE into
        :return: calibrated score of the model on this validation data (X)
        """
        m = nn.Softmax(dim=1)
        h1_sm, h2_sm, h3_sm = m(h1_logits), m(h2_logits), m(h3_logits)
        # y1 = np.array(y1)
        y1 = y1.detach().cpu().numpy()
        y2 = y2.detach().cpu().numpy()
        y3 = y3.detach().cpu().numpy()
        h1_sm = h1_sm.detach().cpu().numpy()
        h2_sm = h2_sm.detach().cpu().numpy()
        h3_sm = h3_sm.detach().cpu().numpy()
        self.temperature_h1 = TemperatureScaling()
        self.temperature_h2 = TemperatureScaling()
        self.temperature_h3 = TemperatureScaling()
        self.temperature_h1.fit(h1_sm, y1)
        self.temperature_h2.fit(h2_sm, y2)
        self.temperature_h3.fit(h3_sm, y3)
        with open('temperature_h1.pkl', 'wb') as fid:
            pickle.dump(self.temperature_h1, fid)
        with open('temperature_h2.pkl', 'wb') as fid:
            pickle.dump(self.temperature_h2, fid)
        with open('temperature_h3.pkl', 'wb') as fid:
            pickle.dump(self.temperature_h3, fid)
