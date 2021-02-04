#!/bin/bash

set -x

DATASETS=( conceptnet atomic atomic2020 conceptnet transomcs )
for DATASET in "${DATASETS[@]}"
do
    python3 ./split/sample_prefixes.py --input-file "./split/data/${DATASET}/test.tsv" --output-file "./split/data/${DATASET}/test_sampled_prefixes.jsonl" --sample-size 5000
done