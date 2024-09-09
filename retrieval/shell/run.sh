#!/bin/bash
cd /root/workspace0401/S-Prompts-hpi
python main.py > ./logss/hip.txt
cd /root/workspace0401/S-Prompts-sprompt
python main.py > ./logss/sprompt.txt