# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb

from torch import cuda
import logging

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)

class KGDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len, model="t5", is_eval=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.head_event
        self.ctext = self.data.tail_event
        self.model = model
        self.is_eval = is_eval

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        if self.model == "t5":
            source = self.tokenizer.batch_encode_plus([text], pad_to_max_length=True, max_length=self.source_len, return_tensors='pt', truncation=True)
            target = self.tokenizer.batch_encode_plus([ctext], pad_to_max_length=True, max_length=self.summ_len, return_tensors='pt', truncation=True)

            # source_for_len = self.tokenizer.batch_encode_plus([text], max_length=100, truncation=True)
            # target_for_len = self.tokenizer.batch_encode_plus([ctext], max_length=100, truncation=True)
        else:
            if self.is_eval:
                source = self.tokenizer.batch_encode_plus([text], pad_to_max_length=False, max_length=self.source_len, return_tensors='pt', truncation=True)
                target = self.tokenizer.batch_encode_plus([ctext], pad_to_max_length=False, max_length=self.summ_len, return_tensors='pt', truncation=True)
            else:
                source = self.tokenizer.batch_encode_plus([text + ' ' + ctext], pad_to_max_length=True, max_length=self.source_len + self.summ_len, return_tensors='pt', truncation=True)
                target = source
        if index < 5:
            logger.info("Source: {}".format(self.tokenizer.batch_decode(source['input_ids'])))
            logger.info("Target: {}".format(self.tokenizer.batch_decode(target['input_ids'])))

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_mask.to(dtype=torch.long)
        }