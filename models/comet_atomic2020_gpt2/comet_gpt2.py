# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
from typing import List

# Importing the GPT2 modules from huggingface/transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb
import logging

from torch import cuda

from split.utils import write_items

from optparse import OptionParser

device = 'cuda' if cuda.is_available() else 'cpu'

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)

# logger.info for allenai beaker verification
logger.info(device)
logger.info(torch.cuda.device_count())

from mosaic.infra.modeling import train, validate, beam_generations
from mosaic.datasets.KGDataset import KGDataset

DEBUG = False
NUM_INST = 100


def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def main():
    wandb.init(project="gpt2_comet_atomic")

    config = wandb.config
    config.TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 2))
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 2))
    config.TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 3))
    config.VAL_EPOCHS = int(os.environ.get("VAL_EPOCHS", 1))
    config.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
    config.SEED = int(os.environ.get("SEED", 42))
    config.IN_LEN = int(os.environ.get("IN_LEN", 16))
    config.OUT_LEN = int(os.environ.get("OUT_LEN", 34))
    config.SUMMARY_LEN = 0 # Used for t5
    config.OUT_DIR = os.environ.get("OUT_DIR", "/models")
    config.DO_TRAIN = os.environ.get("DO_TRAIN", "False") == "True"
    config.DO_PRED = os.environ.get("DO_PRED", "True") == "True"
    config.PRED_FILE = str(os.environ.get("PRED_FILE", ""))
    config.TOP_K = int(os.environ.get("TOP_K", 40))
    config.PRED_BATCH = 64
    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl")

    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    model_name = "gpt2" if 'GPT2_MODEL' not in os.environ else os.environ['GPT2_MODEL']

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER)

    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'additional_special_tokens': [
            'LocationOfAction',
            'HinderedBy',
            'HasFirstSubevent',
            'NotHasProperty',
            'NotHasA',
            'HasA',
            'AtLocation',
            'NotCapableOf',
            'CausesDesire',
            'HasPainCharacter',
            'NotDesires',
            'MadeUpOf',
            'InstanceOf',
            'SymbolOf',
            'xReason',
            'isAfter',
            'HasPrerequisite',
            'UsedFor',
            'MadeOf',
            'MotivatedByGoal',
            'Causes',
            'oEffect',
            'CreatedBy',
            'ReceivesAction',
            'NotMadeOf',
            'xWant',
            'PartOf',
            'DesireOf',
            'HasPainIntensity',
            'xAttr',
            'DefinedAs',
            'oReact',
            'xIntent',
            'HasSubevent',
            'oWant',
            'HasProperty',
            'IsA',
            'HasSubEvent',
            'LocatedNear',
            'Desires',
            'isFilledBy',
            'isBefore',
            'InheritsFrom',
            'xNeed',
            'xEffect',
            'xReact',
            'HasLastSubevent',
            'RelatedTo',
            'CapableOf',
            'NotIsA',
            'ObjectUse',
            '[GEN]'
        ]
    })
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_dataset = pd.read_csv(
        os.environ.get('TRAIN_DATA_PATH', "/tmp/gpt2data/atomic_train.tsv"),
        encoding='latin-1', sep="\t")
    if DEBUG:
        train_dataset = train_dataset.head(NUM_INST)
    # train_dataset = train_dataset[['head_event', 'tail_event', 'relation']]
    train_dataset.head_event = train_dataset.head_event + ' ' + train_dataset.relation \
                               + " [GEN]"
    train_dataset.tail_event = train_dataset.tail_event + ' [EOS]'
    logger.info(train_dataset.head())
    logger.info(train_dataset.tail_event)

    val_dataset = pd.read_csv(os.environ.get('DEV_DATA_PATH', "/tmp/gpt2data/atomic_dev.tsv"), encoding='latin-1', sep="\t")
    if DEBUG:
        val_dataset = val_dataset.head(NUM_INST)
    val_dataset = val_dataset[['head_event', 'tail_event', 'relation']]
    val_dataset.head_event = val_dataset.head_event + ' ' + val_dataset.relation + " [GEN]"
    val_dataset.tail_event = val_dataset.tail_event + ' [EOS]'
    logger.info(val_dataset.tail_event)
    logger.info(val_dataset.head())

    test_dataset = pd.read_csv(os.environ.get('TEST_DATA_PATH', "/tmp/gpt2data/atomic_test.tsv"), encoding='latin-1', sep="\t")
    if DEBUG:
        test_dataset = test_dataset.head(NUM_INST)
    test_dataset = test_dataset[['head_event', 'tail_event', 'relation']]
    test_dataset.head_event = test_dataset.head_event + ' ' + test_dataset.relation \
                              + " [GEN]"
    test_dataset.tail_event = test_dataset.tail_event + ' [EOS]'
    logger.info(test_dataset.tail_event)
    logger.info(test_dataset.head())

    val_dataset_mini = pd.read_csv(os.environ.get('DEV_DATA_PATH', "/tmp/gpt2data/atomic_dev.tsv"), encoding='latin-1', sep="\t")
    if DEBUG:
        val_dataset_mini = val_dataset_mini.head(5)
    val_dataset_mini = val_dataset_mini.sample(n=min(int(val_dataset_mini.size / 3), 100),
                                               random_state=config.SEED)
    val_dataset_mini = val_dataset_mini[['head_event', 'tail_event', 'relation']]
    val_dataset_mini.head_event = val_dataset_mini.head_event + ' ' + val_dataset_mini.relation + " [GEN]"
    val_dataset_mini.tail_event = val_dataset_mini.tail_event + ' [EOS]'
    logger.info(val_dataset_mini.tail_event)
    logger.info(val_dataset_mini.head())

    logger.info("TRAIN Dataset tuple count: {}".format(train_dataset.shape))
    logger.info("DEV Dataset tuple_count: {}".format(val_dataset.shape))
    logger.info("DEV MINI Dataset tuple_count: {}".format(val_dataset_mini.shape))

    training_set = KGDataset(train_dataset, tokenizer, config.OUT_LEN, config.SUMMARY_LEN, model="gpt2")
    val_set = KGDataset(val_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
    val_set_mini = KGDataset(val_dataset.head(2000), tokenizer, config.IN_LEN,  config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
    test_set = KGDataset(test_dataset, tokenizer, config.IN_LEN,  config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params, drop_last=True)
    val_loader = DataLoader(val_set, **val_params, drop_last=True)
    test_loader = DataLoader(test_set, **val_params, drop_last=True)
    val_loader_mini = DataLoader(val_set_mini, **val_params, drop_last=True)
    
    logging.info("Loading model from {}".format(model_name))
    model = GPT2LMHeadModel.from_pretrained(model_name, use_cdn=False)
    logging.info("Move model to device {}".format(device))
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    wandb.watch(model, log="all")

    if config.DO_TRAIN:
        logger.info('Initiating Fine-Tuning for the model on our dataset')

        for epoch in range(config.TRAIN_EPOCHS):
            train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader_mini, model_class="gpt2")
            model.save_pretrained('{}/checkpoint_{}'.format(config.OUT_DIR, epoch))
            tokenizer.save_pretrained('{}/checkpoint_{}'.format(config.OUT_DIR, epoch))
        model.save_pretrained('/models')

    if config.DO_PRED:

        if config.PRED_FILE.endswith("jsonl"):
            records = read_jsonl_lines(config.PRED_FILE)
            pred_dataset = pd.DataFrame.from_records(records)
            pred_dataset = pred_dataset.rename(columns={"head": "head_event", "tails": "tail_event"})
            pred_dataset = pred_dataset.explode('tail_event')
        else:
            pred_dataset = pd.read_csv(config.PRED_FILE, encoding='latin-1', sep="\t")

        if DEBUG:
            pred_dataset = pred_dataset.head(NUM_INST)

        pred_dataset = pred_dataset.drop_duplicates(['head_event', 'relation'], ignore_index=True)

        pred_dataset.head_event = pred_dataset.head_event + ' ' + pred_dataset.relation + " [GEN]"
        pred_dataset.tail_event = pred_dataset.tail_event + ' [EOS]'
        logger.info(pred_dataset.tail_event)
        logger.info(pred_dataset.head())

        pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations(tokenizer, model, device, pred_loader, top_k=config.TOP_K)
        write_items(os.path.join(config.OUT_DIR, "pred_generations.jsonl"),
                    [json.dumps(r) for r in pred_generations])

        # Resave the model to keep generations and model associated
        model.save_pretrained('/models')
        tokenizer.save_pretrained('/models')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--test_install",
                      action="store_true", default=False,
                      help="Test install, without running any modeling code.")

    (options, args) = parser.parse_args()
    if not options.test_install:
        main()
