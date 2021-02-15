import pandas as pd
import transformers
import os

from transformers import T5Tokenizer

train_dataset = pd.read_csv('../data/atomic2020_train.tsv', encoding='latin-1', sep="\t")
train_dataset = train_dataset[['head_event','tail_event','relation']]
train_dataset.head_event = train_dataset.head_event + ' ' + train_dataset.relation + " [EOS]"
train_dataset.tail_event = train_dataset.tail_event + ' [EOS]'


tokenizer = T5Tokenizer.from_pretrained('t5-large')
tokenizer.add_special_tokens({'eos_token': '[EOS]', 'additional_special_tokens': ['LocationOfAction', 'HinderedBy', 'HasFirstSubevent', 'NotHasProperty', 'NotHasA', 'HasA', 'AtLocation', 'NotCapableOf', 'CausesDesire', 'HasPainCharacter', 'NotDesires', 'MadeUpOf', 'InstanceOf', 'SymbolOf', 'xReason', 'isAfter', 'HasPrerequisite', 'UsedFor', 'MadeOf', 'MotivatedByGoal', 'Causes', 'oEffect', 'CreatedBy', 'ReceivesAction', 'NotMadeOf', 'xWant', 'PartOf', 'DesireOf', 'HasPainIntensity', 'xAttr', 'DefinedAs', 'oReact', 'xIntent', 'HasSubevent', 'oWant', 'HasProperty', 'IsA', 'HasSubEvent', 'LocatedNear', 'Desires', 'isFilledBy', 'isBefore', 'InheritsFrom', 'xNeed', 'xEffect', 'xReact', 'HasLastSubevent', 'RelatedTo', 'CapableOf', 'NotIsA', 'ObjectUse']})

writer_source = open('lens_source.txt', 'w')
writer_target = open('lens_target.txt', 'w')

lens_source = []
lens_target = []

for _, row in train_dataset.iterrows():
    text = str(row["head_event"])
    text = ' '.join(text.split())

    ctext = str(row["tail_event"])
    ctext = ' '.join(ctext.split())

    source = tokenizer.batch_encode_plus([text])
    target = tokenizer.batch_encode_plus([ctext])
    writer_source.write(str(len(source["input_ids"][0])) + "\n")
    writer_target.write(str(len(target["input_ids"][0])) + "\n")

    lens_source.append(len(source["input_ids"][0]))
    lens_target.append(len(target["input_ids"][0]))

for lens_list in [lens_source, lens_target]:
    results = {
        "mean": mean(lens_list),
        "min": min(lens_list),
        "stdev": stdev(lens_list),
        "max": max(lens_list)
    }

    print(results)