import gradio as gr

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse

import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import yaml
import json
import pdb
import os
import random
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
# from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.inference_one_image import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.stats import get_model_complexity_info
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.structures.image_list import ImageList
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import os
import functools
import io
import os
import datetime

import torch
import torch.distributed as dist



def load_visualizer(config_file=None):
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config-file",
        # default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        default='./configs/refcoco/val/finetune_A_decompose_interact_layer_task.yaml',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        # default=None,
        default='./best_model/model_lpi.pth',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--task_config", default='./configs/refcoco/val/finetune_A_decompose_interact_layer_task.yaml')

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--box_pixel', default=3, type=int)
    parser.add_argument('--text_size', default=1, type=float)
    parser.add_argument('--text_pixel', default=1, type=int)
    parser.add_argument('--image_index', default=0, type=int)
    parser.add_argument('--threshold', default=0.6, type=float)
    parser.add_argument("--text_offset", default=10, type=int)
    parser.add_argument("--text_offset_original", default=4, type=int)
    parser.add_argument("--color", default=255, type=int)

    args = parser.parse_args()

    if config_file != None:
        args.config_file = config_file
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )
        init_distributed_mode(args)
        print("Passed distributed init")

    cfg.defrost()
    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    log_dir = cfg.OUTPUT_DIR
    if args.weight:
        log_dir = os.path.join(log_dir, "eval", os.path.splitext(os.path.basename(args.weight))[0])
    if log_dir:
        mkdir(log_dir)

    logger = setup_logger("maskrcnn_benchmark", log_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = build_detection_model(cfg)
    try:
        model.to(cfg.MODEL.DEVICE)
    except:
        cfg.defrost()
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()


    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    if args.weight:
        _ = checkpointer.load(args.weight, force=True)
    else:
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

    all_keys = torch.load('./all_key/all_key.pth')
    model.all_keys = all_keys

    visualizer = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        load_model=True,
        model=model
    )
    return visualizer

def retrieval(model_name, caption, image, metric, show_score=True):
    # image_tensor = torch.tensor(image)
    # image_tensor = image_tensor.permute(2,0,1)
    # image_tensor = image_tensor.unsqueeze(0)
    # bs, c, w, h = image_tensor.shape
    # image_tensor_shape = [torch.tensor([w, h])]
    # image_list = ImageList(image_tensor, image_tensor_shape)
    # print(show_score)
    global global_model
    global visualizer

    if global_model != model_name:
        global_model = model_name
        if model_name == 'GLIP':
            visualizer = load_visualizer(config_file_dict['GLIP'])
        elif model_name == 'LPI':
            visualizer = load_visualizer(config_file_dict['LPI'])
        # reload the model
    # visualizer.model.cfg.defrost()
    # if model_name == 'GLIP':
    #     visualizer.model.cfg.LPAI.VISUAL_PROMPT = False
    #     visualizer.model.cfg.LPAI.TEXTUAL_PROMPT = False
    #     visualizer.model.cfg.LPAI.INTERACT = False
    #     visualizer.model.cfg.LPAI.PROMPT_LORA = False
    # elif model_name == 'LPI':
    #     visualizer.model.cfg.LPAI.VISUAL_PROMPT = True
    #     visualizer.model.cfg.LPAI.TEXTUAL_PROMPT = True
    #     visualizer.model.cfg.LPAI.INTERACT = True
    #     visualizer.model.cfg.LPAI.PROMPT_LORA = True
    # visualizer.model.cfg.freeze()
    caption = [caption]

    prediction = visualizer.compute_prediction(image, caption)

    # image = visualizer.visualize_with_predictions(image, prediction)

    color = 0
    threshold = 0.0
    alpha = 0.9
    text_size = 0.6
    box_pixel = 3
    text_pixel = 2
    # TODO
    result, _ = visualizer.visualize_with_predictions_metric(
        image,
        prediction,
        thresh=threshold,
        alpha=alpha,
        box_pixel=box_pixel,
        text_size=text_size,
        text_pixel=text_pixel,
        # text_offset=text_offset,
        # text_offset_original=text_offset_original,
        color=color,
        metric=metric,
        show_score=show_score
    )

    return result


def show_gallery(text):
    return None


def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    #args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(0, 7200)
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# def load_model(args):
#     model = build_detection_model(cfg)
#     try:
#         model.to(cfg.MODEL.DEVICE)
#     except:
#         cfg.defrost()
#         cfg.MODEL.DEVICE = "cpu"
#         cfg.freeze()
#
#     checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
#     if args.weight:
#         _ = checkpointer.load(args.weight, force=True)
#     else:
#         _ = checkpointer.load(cfg.MODEL.WEIGHT)
#
#     all_keys = torch.load('./all_key/all_key.pth')
#     model.all_keys = all_keys
#     return model

config_file_dict = {
    'LPI': './configs/refcoco/val/finetune_A_decompose_interact_layer_task.yaml',
    'GLIP': './configs/refcoco/val/GLIP_A.yaml'
}

global_model = 'LPI'
config_file = None
visualizer = load_visualizer(config_file)

iface_first = gr.Interface(
    fn=retrieval,
    inputs=[
        gr.Dropdown(['GLIP', 'LPI'], value='LPI', label='Model', info='Choice the model for inference.'),
        gr.Textbox(label='Caption'),
        gr.Image(label='Image'),
        gr.Dropdown(["R@1", "R@5", "R@10"],value='R@1', label="Metrics", info="Metrics for results!"),
        gr.Radio(["True", "False"], label="Show score.",value='True', info="Whether to show the score?"),
    ],
    outputs=[gr.Image(label='Bounding box')]
)

iface_second = gr.Interface(
    fn=show_gallery,
    inputs=[gr.Textbox(label='Model')],
    outputs=[gr.Gallery()]
)

tabbed_interface = gr.TabbedInterface([iface_first, iface_second], ['Inference', 'Gallery'])
def main():
    tabbed_interface.launch()


if __name__ == '__main__':
    main()
