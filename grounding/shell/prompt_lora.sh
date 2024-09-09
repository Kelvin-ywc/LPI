#!/bin/bash

cd /root/workspace/grounding/prompt_grounding
python tools/finetune.py --config-file configs/ablation/prompt/finetune_A_r1.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/prompt_finetune_A_r1.txt

python tools/finetune.py --config-file configs/ablation/prompt/finetune_A_r2.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/prompt_finetune_A_r2.txt

python tools/finetune.py --config-file configs/ablation/prompt/finetune_A_r8.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/prompt_finetune_A_r8.txt

python tools/finetune.py --config-file configs/ablation/prompt/finetune_A_r16.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/prompt_finetune_A_r16.txt