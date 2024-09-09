#!/bin/bash

cd /root/workspace/grounding/prompt_grounding

python tools/finetune.py       --config-file configs/sprompt/finetune_A_decompose.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/sprompt_vis_refcoco.txt

mv visualize visualize_sprompts
mkdir visualize
cd visualize
mkdir 0
mkdir 1
mkdir 2
mkdir 3
mkdir 4
mkdir 5
mkdir 6
mkdir 7
mkdir 8
mkdir 9
mkdir 10
mkdir 11
cd ..

python tools/finetune.py       --config-file configs/refcoco/val/finetune_A_decompose_interact_layer_task.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/lpi_vis_refcoco.txt
