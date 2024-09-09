#!/bin/bash

cd /root/workspace/grounding/prompt_grounding

#python tools/finetune.py       --config-file configs/maple/finetune_A_decompose.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/maple_refcoco.txt

python tools/finetune.py       --config-file configs/l2p/finetune_A_decompose_refcoco+.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/l2p_refcoco+.txt

python tools/finetune.py       --config-file configs/l2p/finetune_A_decompose_refcocog.yaml --skip-test       --custom_shot_and_epoch_and_general_copy 0_10_1 > ./logs/l2p_refcocog.txt
