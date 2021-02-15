
# This is an example for train (finetune) COMET-ATOMIC2020 with BART
# Please specify --data_dir that contains [train|val|test].[source|target].

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --task summarization \
    --num_workers 2 \
    --learning_rate=1e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 100 \
    --val_check_interval 1.0 \
    --sortish_sampler \
    --data_dir ./data/atomic_2020_samples \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --output_dir ./results/ \
    --num_train_epochs 1 \
    --model_name_or_path facebook/bart-large \
    --atomic 
