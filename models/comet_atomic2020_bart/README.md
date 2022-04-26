# COMET-ATOMIC2020 with BART

## experiment set up

1. install required packages

        pip install -r requirements.txt
        
        (We also checked that `pytorch==1.8.1 and transformers==4.1.1` works.)

2. train the model

        bash run.sh

3. the results are stored at `./results/`


## use our model

1. download the model file

        bash download_model.sh

2. sample usage 

        python ./generation_example.py
