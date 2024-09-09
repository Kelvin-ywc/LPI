#!/bin/bash

cd /root/workspace/grounding/prompt_grounding

python tools/finetune.py --config-file configs/refcoco+/finetune_A_decompose_interact_layer_task.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./log_new/refcoco+_interact_layer_task.txt

python tools/finetune.py --config-file configs/refcoco+/finetune_A_decompose_layer_task.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./log_new/refcoco+_layer_task.txt