#!/bin/bash

cd /root/workspace/grounding/prompt_grounding

python tools/finetune.py --config-file configs/ablation/prompt_depth/finetune_A_decompose_interact_layer_task_d8.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./log_new/refcoco_interact_layer_task_d8.txt

python tools/finetune.py --config-file configs/ablation/prompt_depth/finetune_A_decompose_interact_layer_task_d10.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./log_new/refcoco_interact_layer_task_d10.txt

python tools/finetune.py --config-file configs/ablation/prompt_depth/finetune_A_decompose_interact_layer_task_d12.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./log_new/refcoco_interact_layer_task_d12.txt