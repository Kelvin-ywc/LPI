#!/bin/bash

cd /root/workspace/grounding/prompt_grounding
#python tools/finetune.py --config-file configs/sprompt/finetune_A_decompose.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/prompt_finetune_A_r1.txt

python tools/finetune.py --config-file configs/sprompt/finetune_A_decompose_refcoco+.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/sprompt_refcoco+.txt

python tools/finetune.py --config-file configs/sprompt/finetune_A_decompose_refcocog.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/sprompt_refcocog.txt

#python tools/finetune.py --config-file configs/ablation/prompt/finetune_A_r16.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/prompt_finetune_A_r16.txt