import sys
import csv

writer = csv.DictWriter(open('../data/atomic_test.tsv', 'w'), delimiter='\t', fieldnames=['relation', 'head_event', 'tail_event'])
writer.writeheader()

with open(sys.argv[1]) as file:
    reader = csv.DictReader(file, delimiter='\t', fieldnames=['head', 'relation', 'tail', 'id1', 'id2', 'score'])
    for row in reader:
        writer.writerow({'relation': row['relation'], 'head_event': row['head'], 'tail_event': row['tail']})