# COMET2020 Codebase

This folder contains all the code needed to reproduce expirements in the COMET2020 paper.

# Setup

Run `pip install -r requirements.txt` to install requirements for your Python instance. We recommend [Conda](https://www.anaconda.com/) to manage Python installs. Our codebases is on Python 3.

The code for modeling is located in `mosaic/modeling`. `mosaic/KGDataset` is used to convert the ATOMIC2020 CSV into an HuggingFace `Datasets` object.

# Directory Overview

`beaker_exp`: Contains files needed to run expirements using `Beaker` (https://beaker.org/) instead of on your local machine.

`human_eval`: Contains HTML files for human evaluation on Amazon MTurk, as described in the AAAI paper.

`models`: Contains additional modeling files to reproduce the GPT2 and BART expirements. `models/comet_atomic2020_bart` contains a README and code to run COMET-BART2020.

`scripts`: Contains additional scripts (e.g. utils.py) used during expirements in the COMET-ATOMIC2020 paper.

`split`: Contains code used to make the test, train, and dev splits of ATOMIC2020 with Stratified Random Sampling.

`system_eval`: Contains code for automatic evaluation of generated entities.