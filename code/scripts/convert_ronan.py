import sys
import csv

writer = csv.DictWriter(open(f'../data/{sys.argv[1].split("/")[-1].split("_")[0]}_{sys.argv[1].split("/")[-1].split("_")[-1].split(".")[0]}.tsv', 'w'), delimiter='\t', fieldnames=['relation', 'head_event', 'tail_event'])
writer.writeheader()

with open(sys.argv[1]) as file:
    reader = csv.DictReader(file, delimiter='\t', fieldnames=['head', 'relation', 'tail'])
    for row in reader:
        writer.writerow({'relation': row['relation'], 'head_event': row['head'], 'tail_event': row['tail']})