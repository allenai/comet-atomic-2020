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
logger = logging.getLogger(__name__)

def log_eval(epoch, tokenizer, model, device, loader, sample_limit=5000, model_class="t5"):
    model.eval()
    total_loss = 0
    loss_count = 0
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            if model_class == "t5":
                outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            else:
                outputs = model(input_ids = ids, attention_mask = mask, labels=ids)

            loss = outputs[0]
            total_loss += loss.item()
            loss_count += 1
    wandb.log({"Eval Loss": total_loss / loss_count})
    logger.info("Eval Loss: {}".format(total_loss / loss_count))