#!/usr/bin/env bash

BERT_BASE_DIR='/data/checkpoints/uncased_L-24_H-1024_A-16'
DATASETS="$HOME/.data/glue_data"
DATASETS='../../../Data/imdb'

tensorboard --logdir data/outputs &

for NUMBER in {1001..2000..4}
do
  echo $NUMBER
python run_classifier.py \
  --task_name=IMDB \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$DATASETS/$NUMBER \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
 --num_train_epochs=3 \
  --output_dir=data/outputs/$NUMBER \
  --model_dir=data/outputs/imdb \
  --gpu_number=4 \
  $@
done