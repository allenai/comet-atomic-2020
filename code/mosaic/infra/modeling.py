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
import logging
from tqdm import tqdm

logger = logging.getLogger("modeling")
from mosaic.infra.logging import log_eval


def train(epoch, tokenizer, model, device, loader, optimizer, val_loader=None, model_class="t5",
          save_dir="/models"):
    model.train()
    batch_count = len(loader)
    for iteration, data in tqdm(enumerate(loader, 0)):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        if model_class == "t5":
            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids,
                            lm_labels=lm_labels)
        else:
            outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs[0]

        if iteration % 100 == 0:
            wandb.log({"Training Loss": loss.item(), "Epoch": epoch,
                       "Batches left": batch_count - iteration})
            batches_left = batch_count - iteration
            logger.info(
                f'\nEpoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}, Batches left: {batches_left}')

        if iteration % 500 == 0:
            logger.info(f'\nEpoch: {epoch}, Loss:  {loss.item()}, BatchesLeft: {batches_left}')

        if iteration % 5000 == 0:
            model.save_pretrained(save_dir + "/iter_{}_model".format(iteration))
            tokenizer.save_pretrained(save_dir + "/iter_{}_tokenizer".format(iteration))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0 and val_loader != None:
            log_eval(epoch, tokenizer, model, device, val_loader, model_class=model_class)
            model.train()


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=True,
                max_length=int(os.environ['OUT_LEN']),
                num_beams=5,
                top_k=50,
                top_p=0.95
            )

            preds = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                g in generated_ids]
            target = [
                tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                t in y]
            source = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                s in ids]

            if _ % 100 == 0:
                logger.info(f'Completed {_}')

            sources.extend(source)
            predictions.extend(preds)
            actuals.extend(target)
    return sources, predictions, actuals


def beam_generations(tokenizer, model, device, loader, top_k=40):
    # This method assumes batch size of 1
    model.eval()
    predictions = []
    actuals = []
    sources = []
    records = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                temperature=1.0,
                do_sample=False,
                max_length=int(os.environ['OUT_LEN']),
                top_p=0.9,
                top_k=top_k,
                repetition_penalty=1.0,
                num_return_sequences=10 if top_k > 1 else 1,
                num_beams=10
            )

            preds = [tokenizer.decode(g, clean_up_tokenization_spaces=True) for g in generated_ids]
            try:
                target = [tokenizer.decode(t, clean_up_tokenization_spaces=True) for t in y]
            except:
                target = ['']
            source = [tokenizer.decode(s, clean_up_tokenization_spaces=True) for s in ids]

            records.append({
                'source': source[0],
                'target': target[0],
                'generations': preds
            })

            if _ % 100 == 0:
                logger.info(f'Completed {_}')

    return records
#
# def batch_greedy_generate(tokenizer, model, dataloader, device, max_num_tokens_to_produce=20):
#
#     model.eval()
#     with torch.no_grad():
#         for _, data in enumerate(dataloader, 0):
#             input_ids = data['source_ids'].to(device, dtype = torch.long)
#             attn_mask = data['source_mask'].to(device, dtype = torch.long)
#
#             pad_token_id = tokenizer.pad_token_id
#             eos_token_id = tokenizer.eos_token_id
#             eos_not_in_sents = torch.ones(input_ids.shape[0]).long()
#
#             last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
#
#             start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size).unsqueeze(1)
#             past = None
#             seq_len = input_ids.size(1)
#             position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])])
#             for i, position_ids_slice in enumerate(position_ids):
#                 position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]
#
#             for step in range(max_num_tokens_to_produce):
#                 outputs = model(input_ids, attention_mask=attn_mask, position_ids=position_ids)
#
#                 if step == 0:
#                     next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
#                 else:
#                     next_token_logits = outputs[0][:, -1, :]
#
#                 next_tokens = torch.argmax(next_token_logits, dim=-1)
#
#                 # this updates which sentences have not seen an <EOS> token so far
#                 # if one <EOS> token was seen the sentence is finished
#                 eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())
#
#                 # either append a padding token here if <EOS> has been seen or append next token
#                 tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)
#
#                 # Update input_ids, attn_mask and position_ids
#                 input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
#                 attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long()], dim=1)
#                 position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
#
