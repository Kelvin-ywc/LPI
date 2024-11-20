#!/bin/bash

cd /home1/yanweicai/workspace/prompt/lpi/grounding

python tools/finetune.py --config-file configs/refcoco+/finetune_A_decompose_interact_layer_task.yaml --skip-test --custom_shot_and_epoch_and_general_copy 0_10_1