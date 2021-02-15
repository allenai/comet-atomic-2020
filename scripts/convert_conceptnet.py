import sys
import csv

writer = csv.DictWriter(open('../data/conceptnet_dev.tsv', 'w'), delimiter='\t', fieldnames=['relation', 'head', 'tail'])
writer.writeheader()

with open(sys.argv[1]) as file:
    reader = csv.DictReader(file, delimiter='\t', fieldnames=['relation', 'head', 'tail', 'score'])
    for row in reader:
        writer.writerow({'relation': row['relation'], 'head': row['head'], 'tail': row['tail']})