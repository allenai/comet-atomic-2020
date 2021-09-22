# Evaluation Metrics for COMET-ATOMIC 2020

The second step in reproducting the result in the paper is running evaluation metrics. This folder contains example files, scripts, and installation information for running evaluation of COMET generations from all models.

## Installation Instructions

To install, run `pip install -r requirements.txt`. Our codebase was compiled with Python 3.6, although earlier versions may work as well. We recommend using `conda` to handle dependencies.

## Running Evaluation

The following are instructions for running evaluation metrics on COMET-ATOMIC 2020. We write out the instructions w.r.t the example file provided in `BART/` -- to check that your evaluation framework is set up properly, your metrics generated should match the metrics reported in the paper.

1. Run `python automatic_eval.py --input_file YOUR_INPUT_FILE`. For example,  `python automatic_eval.py --input_file BART-atomic_2020.json` will run on the results for COMET-BART 2020.

Some notes about the evaluation script:

- Please use the evaluation script here and NOT `nltk` or another evaluation metric generator.

- The script will create a generation file and a scores file, and save scores in `./results`.

- The scripts expects generations for the 5000 example `test.tsv` file, not the whole test file. The script won't directly work for evaluation on the entire test file or on the development set. You'll need to edit `test.tsv` to contain tuples for the evaluation set you decide to evaluate on, and the order of the tuples in `test.tsv` and your generations `jsonl` file must match line by line.