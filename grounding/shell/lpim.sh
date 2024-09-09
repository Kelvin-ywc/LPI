#!/bin/bash

cd /root/workspace/grounding/prompt_grounding

python tools/finetune.py       --config-file configs/refcoco+/finetune_A_decompose_layer_task.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/refcoco+_layer_task.txt

python tools/finetune.py       --config-file configs/refcocog/finetune_A_decompose_layer_task.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/refcocog_layer_task.txt