import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    # param = load_json('configs/coco_org_sprompt.json')
    '''
    configs:
    clip: configs/lpi/coco_clip.json
    l2p: configs/lpi/coco_l2p.json
    S-prompts: configs/lpi/coco_sprompts.json
    lpi(ours): configs/lpi/coco_sprompts.json
    '''
    # param = load_json('configs/lpi/coco_org_sprompt.json')

    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--local_rank',  default=-1)
    return parser


if __name__ == '__main__':
    main()
