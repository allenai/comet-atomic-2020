#! /bin/sh

# edit the directory
# data file must be in tab separated files
#   conceptnet.tsv
#   atomic.tsv
#   atomic2020.tsv
#   transomcs.tsv
# 3 columns expected (order inconsequential): head relation tail
# header expected

python3 preprocess_kb_triples_part1.py data/
python3 preprocess_kb_triples_part2.py data/
python3 calculate_coverage.py data/
