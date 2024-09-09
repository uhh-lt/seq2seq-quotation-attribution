#!/bin/bash
BASE_NAME=$1
DATA_DIR=$2
MODEL_SAVE_DIR=$3
PREDICT_SAVE_DIR=$4
LOG_DIR=$5
lr=$6
epochs=$7
eval_step=${8}
log_step=${9}
eval_delay=${10}
train_bsz=${11}
eval_bsz=${12}
port=${13}
original_input_dir=${14}

# lr=5e-4 for T5-base/large, lr=5e-5 for T0_3B, lr=3e-5 for T0_3B integer-free
# lr=1e-5/3e-5 for T0pp
# min_num_mentions=2 for ontonotes, 1 for others
# ontonotes: eval_len_out=4096 PreCo: eval_len_out=2560 LB: eval_len_out=6170, tuba: eval_len_out=5161/5388, quote: eval_len_out=2944(mt5)/2881(t5)/2072(t5-short)
# ontonotes: epochs=100 PreCO: epochs=10  LB: epochs=100
# For OntoNotes: eval_step=800, save_step=800, eval_delay=30000, log_step=100
# For PreCo: eval_step=3200, save_step=15200, eval_delay=30000, log_step=100
# For LB: eval_step=100, save_step=100, eval_delay=1500, log_step=10
# ontonotes and lb: eval_bsz=1  preco: eval_bsz=2

# SEQ_TYPE meaning:
# action: copy_action, short_seq: partial linearization, full_seq: token action
# tagging: copy action as decoder_input,
# input_feed: token action sequence+copy action sequence as decoder_input

# ACTION_TYPE meaning:
# integer: integer cluster identity representation
# non_integer: integer-free cluster identity representation


#MODEL_NAME="bigscience/T0_3B"
#MODEL_NAME="google/flan-t5-xl"
#epochs=100
gpus="0,1,2,3,4,5,6,7"
warmup=0.1
train_len=2048
train_len_out=4096
#2310
eval_len=4096
#eval_len_out=4096
num_beams=4
#log_step=100
#eval_step=800
#save_step=800
#eval_delay=30000

weight_decay=0.01
n_gpu=1
ds_config_dir="ds_configs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,4
# export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRITON_CACHE_DIR=/mnt/data1/users/fschroeder/.cache/triton

declare -A eval_len_map=( [action]=2881 [short_seq]=2072 )

models="google-t5/t5"
seq_types="short_seq action"
sizes="3b"

for m in $models; do
    m_name="${m//\//_}"
    echo $m_name   
    for seq_type in $seq_types; do
        eval_len_out=${eval_len_map[$seq_type]}
        for size in $sizes; do
            model="$m-$size"
            combination="$BASE_NAME-$m_name-$size-$seq_type"
            input_dir="$DATA_DIR/$BASE_NAME-$seq_type"
            model_save_dir="$MODEL_SAVE_DIR/$combination"
            log_dir="$LOG_DIR/$combination"
            predict_dir="$PREDICT_SAVE_DIR/$combination"
            if [ -f "$predict_dir/test-results.json" ]; then
                echo Skipping combination $combination: already done.
                continue
            fi
            echo Running combination $combination: input from $input_dir, save model in $model_save_dir, log to $log_dir, predict in $predict_dir
            deepspeed --master_port $port main_trainer.py \
                --output_dir $model_save_dir  \
                --model_name_or_path $model \
                --do_train True \
                --save_strategy steps  \
                --load_best_model_at_end True \
                --metric_for_best_model quote_f1_joint \
                --evaluation_strategy steps \
                --logging_steps $log_step \
                --eval_steps $eval_step \
                --original_input_dir $original_input_dir \
                --data_dir $input_dir \
                --language german \
                --save_dir $predict_dir \
                --per_device_train_batch_size $train_bsz  \
                --per_device_eval_batch_size $eval_bsz \
                --learning_rate $lr \
                --num_train_epochs $epochs \
                --logging_dir $log_dir \
                --remove_unused_columns False \
                --overwrite_output_dir True \
                --dataloader_num_workers 0 \
                --predict_with_generate True \
                --warmup_ratio $warmup \
                --max_train_len $train_len \
                --max_train_len_out $train_len_out \
                --max_eval_len $eval_len \
                --max_eval_len_out $eval_len_out \
                --generation_num_beams $num_beams \
                --generation_max_length $eval_len_out \
                --weight_decay $weight_decay \
                --save_predicts True \
                --do_predict True \
                --bf16 True \
                --save_total_limit 2 \
                --save_steps $eval_step \
                --eval_delay $eval_delay \
                --gradient_checkpointing True \
                --seq2seq_type $seq_type \
                --mark_sentence True \
                --action_type integer \
                --align_mode l \
                --min_num_mentions 1 \
                --add_mention_end False \
                --deepspeed $ds_config_dir/ds_stage2.json
        done
    done
done



