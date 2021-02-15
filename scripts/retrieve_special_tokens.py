import csv

relations = set()
with open('../old_data/atomic2020_train.tsv') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        relations.add(row['relation'])

with open('../old_data/atomic_train.tsv') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        relations.add(row['relation'])

with open('../old_data/conceptnet_train.tsv') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        relations.add(row['relation'])

with open('../data/transomcs_train.tsv') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        relations.add(row['relation'])

print({'additional_special_tokens': list(relations)})