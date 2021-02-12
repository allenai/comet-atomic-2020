# Importing stock libraries
import json
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2Model, \
    GPT2LMHeadModel

# Import os for env varibles via Beaker
import os
import tqdm

# WandB – Import the wandb library
import wandb

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import inflect
inflection_engine = inflect.engine()

import spacy
nlp = spacy.load("en")

# Print for allenai beaker verification
print(device)
print(torch.cuda.device_count())

KGS_TO_EVAL = ["atomic", "atomic2020", "conceptnet", "transomcs"]


def write_items(items: List[str], output_file):
    with open(output_file, 'w') as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()


def article(word):
    return "an" if word[0] in ['a', 'e', 'i', 'o', 'u'] else "a"


def posessive(word):
    if inflection_engine.singular_noun(word) is False:
        return "have"
    else:
        return "has"


def vp_present_participle(phrase):
    doc = nlp(phrase)
    return ' '.join([
        inflection_engine.present_participle(token.text) if token.pos_ == "VERB" and token.tag_ != "VGG" else token.text
        for token in doc
    ])


def fact_to_prompt(kg, fact):
    head = fact['head']
    relation = fact['relation']
    tail = fact['tails'][0]

    if kg == "conceptnet" or kg == "transomcs":
        if relation == "AtLocation":
            prompt = "You are likely to find {} {} in {} ".format(
                article(head), head, article(tail)
            )
        elif relation == "CapableOf":
            prompt = "{} can ".format(head)
        elif relation == "CausesDesire":
            prompt = "{} would make you want to ".format(head)
        elif relation == "Causes":
            prompt = "Sometimes {} causes ".format(head)
        elif relation == "CreatedBy":
            prompt = "{} is created by".format(head)
        elif relation == "Desires":
            prompt = "{} {} desires".format(article(head), head)
        elif relation == "HasA":
            prompt = "{} {} ".format(head, posessive(head))
        elif relation == "HasPrerequisite":
            prompt = "{} requires ".format(vp_present_participle(head))
        elif relation == "HasProperty":
            prompt = "{} is ".format(head)
        elif relation == "MotivatedByGoal":
            prompt = "You would {} because you are ".format(head)
        elif relation == "ReceivesAction":
            prompt = "{} can be ".format(head)
        elif relation == "UsedFor":
            prompt = "{} {} is for ".format(article(head).upper(), head)
        elif relation == "HasFirstSubevent" or relation == "HasSubevent" or relation == "HasLastSubevent":
            prompt = "While {}, you would ".format(vp_present_participle(head))
        elif relation == "InheritsFrom":
            prompt = "{} inherits from".format(head)
        elif relation == "PartOf":
            prompt = "{} {} is a part of {} ".format(article(head).upper(), head, article(tail))
        elif relation == "IsA":
            prompt = "{} is {} ".format(head, article(tail))
        elif relation == "InstanceOf":
            prompt = "{} is an instance of".format(head)
        elif relation == "MadeOf":
            prompt = "{} is made of".format(head)
        elif relation == "DefinedAs":
            prompt = "{} is defined as ".format(head)
        elif relation == "NotCapableOf":
            prompt = "{} is not capable of".format(head)
        elif relation == "NotDesires":
            prompt = "{} {} does not desire".format(article(head), head)
        elif relation == "NotHasA":
            prompt = "{} does not have a".format(head)
        elif relation == "NotHasProperty" or relation == "NotIsA":
            prompt = "{} is not".format(head)
        elif relation == "NotMadeOf":
            prompt = "{} is not made of".format(head)
        elif relation == "SymbolOf":
            prompt = "{} is a symbol of".format(head)
        else:
            raise Exception(relation)
    elif kg == "atomic" or kg == "atomic2020":
        if relation == "AtLocation":
            prompt = "You are likely to find {} {} in {} ".format(
                article(head), head, article(tail)
            )
        elif relation == "CapableOf":
            prompt = "{} can ".format(head)
        elif relation == "Causes":
            prompt = "Sometimes {} causes ".format(head)
        elif relation == "Desires":
            prompt = "{} {} desires".format(article(head), head)
        elif relation == "HasProperty":
            prompt = "{} is ".format(head)
        elif relation == "HasSubEvent":
            prompt = "While {}, you would ".format(vp_present_participle(head))
        elif relation == "HinderedBy":
            prompt = "{}. This would not happen if"
        elif relation == "MadeUpOf":
            prompt = "{} {} contains".format(article(head), head)
        elif relation == "NotDesires":
            prompt = "{} {} does not desire".format(article(head), head)
        elif relation == "ObjectUse":
            prompt = "{} {} can be used for".format(article(head), head)
        elif relation == "isAfter":
            prompt = "{}. Before that, ".format(head)
        elif relation == "isBefore":
            prompt = "{}. After that, ".format(head)
        elif relation == "isFilledBy":
            prompt = "{} is filled by".format(head) #TODO
        elif relation == "oEffect":
            prompt = "{}. The effect on others will be".format(head)
        elif relation == "oReact":
            prompt = "{}. As a result, others feel".format(head)
        elif relation == "oWant":
            prompt = "{}. After, others will want to".format(head)
        elif relation == "xAttr":
            prompt = "{}. PersonX is".format(head)
        elif relation == "xEffect":
            prompt = "{}. The effect on PersonX will be".format(head)
        elif relation == "xIntent":
            prompt = "{}. PersonX did this to".format(head)
        elif relation == "xNeed":
            prompt = "{}. Before, PersonX needs to".format(head)
        elif relation == "xReact":
            prompt = "{}. PersonX will be".format(head)
        elif relation == "xReason":
            prompt = "{}. PersonX did this because".format(head)
        elif relation == "xWant":
            prompt = "{}. After, PersonX will want to".format(head)
    else:
        raise Exception("Invalid KG")

    return prompt.strip()


def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start if start != -1 else None


def main():
    wandb.init(project="gpt2_zeroshot")

    config = wandb.config
    config.GPT2_MODEL = str(os.environ.get('GPT2_MODEL', "gpt2"))
    config.OUTPUT_DIR = str(os.environ.get('OUTPUT_DIR', "data/gpt2-zeroshot/output/"))
    config.MODEL_SAVE_LOCATION = str(os.environ.get('MODEL_SAVE_LOCATION', "models/"))
    config.SEED = int(os.environ.get("SEED", 42))
    config.IN_LEN = 75
    config.SUMMARY_LEN = 35
    config.DATA_PATH = "data/gpt2-zeroshot" if "DATA_PATH" not in os.environ else os.environ["DATA_PATH"]
    config.TOP_K = int(os.environ.get('TOP_K', "1"))
    config.TOP_P = float(os.environ.get('TOP_P', "0.9"))
    config.NUM_BEAMS = int(os.environ.get('NUM_BEAMS', "10"))
    config.NUM_SEQUENCES = int(os.environ.get('NUM_SEQUENCES', "10"))
    config.STOP_TOKEN = "."

    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    model_name = config.GPT2_MODEL

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    wandb.watch(model, log="all")

    for kg in KGS_TO_EVAL:
        fname = os.path.join(config.DATA_PATH, kg, "test_sampled_prefixes.jsonl")
        print("\n\n===== Evaluating {} ===== {} \n\n".format(kg, fname))
        generations_for_fact = []
        with open(fname) as f:
            for line_idx, fact in tqdm.tqdm(enumerate(f)):
                prompt = fact_to_prompt(kg, json.loads(fact))
                input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                generations = model.generate(
                    input_ids=input_ids.to(device),
                    max_length=input_ids.size(1) + 10,
                    temperature=1.0,
                    top_k=config.TOP_K,
                    top_p=config.TOP_P,
                    repetition_penalty=1.0,
                    do_sample=True,
                    num_return_sequences=config.NUM_SEQUENCES,
                    num_beams=config.NUM_BEAMS
                )

                if len(generations.shape) > 2:
                    generations.squeeze_()

                text_generations = []
                for gen in generations:
                    gen = gen.tolist()
                    text = tokenizer.decode(gen, clean_up_tokenization_spaces=True)
                    text = text[:find_nth(text, config.STOP_TOKEN, 1)] if config.STOP_TOKEN not in prompt else text[:find_nth(text, config.STOP_TOKEN, 2)]
                    text_generations.append(text)

                generations_for_fact.append({
                    "idx": line_idx,
                    "fact": json.loads(fact),
                    "prompt": prompt,
                    "generations": text_generations
                })

        write_items([json.dumps(r) for r in generations_for_fact], os.path.join(config.OUTPUT_DIR, "{}-zeroshot-generations.jsonl".format(kg)))

    model.save_pretrained(config.MODEL_SAVE_LOCATION)


if __name__ == '__main__':
    main()
