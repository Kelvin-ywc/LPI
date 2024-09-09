#!/bin/bash

cd /root/workspace/grounding/prompt_grounding
#python tools/finetune.py       --config-file configs/refcoco/finetune_A_decompose_task_layer_interact.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1

python tools/finetune.py       --config-file configs/refcoco/finetune_A_decompose_task_interact.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1

python tools/finetune.py       --config-file configs/refcoco/finetune_A_decompose_layer_interact.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1

python tools/finetune.py       --config-file configs/refcoco/finetune_A_decompose_layer_task.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1

#python tools/finetune.py       --config-file configs/refcoco/finetune_A_decompose_task.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1
#
#python tools/finetune.py       --config-file configs/refcoco/finetune_A_decompose_layer.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1
#
#python tools/finetune.py       --config-file configs/refcoco/finetune_A_decompose_interact.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1