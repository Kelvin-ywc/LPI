# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.prompt.prompt import PromptEncoder

from ..backbone import build_backbone
from ..rpn import build_rpn
from ..roi_heads import build_roi_heads

from ..language_backbone import build_language_backbone

from transformers import AutoTokenizer

import random
import timeit
import pdb
from copy import deepcopy
import numpy as np

from typing import List, Optional, Tuple, Union
from matrix.matrix import nt_bxent_loss
from sklearn.cluster import KMeans
import copy

def random_word(input_ids, mask_token_id, vocabs, padding_token_id, greenlight_map):
    """
    greenlight_map, batch_size x 256 (seq_len):
        0 means this location cannot be calculated in the MLM loss
        -1 means this location cannot be masked!!
        1 means this location can be masked and can be calculated in the MLM loss
    """
    output_label = deepcopy(input_ids)
    for j in range(input_ids.size(0)):
        for i in range(input_ids.size(1)):
            prob = random.random()
            # mask token with probability
            ratio = 0.15
            if greenlight_map is not None and greenlight_map[j,i] == -1:
                output_label[j,i] = -100
                continue

            if (not input_ids[j,i] == padding_token_id) and prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input_ids[j,i] = mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input_ids[j,i] = random.choice(vocabs)

            else:
                # no masking token (will be ignored by loss function later)
                output_label[j,i] = -100
            
            if greenlight_map is not None and greenlight_map[j,i] != 1:
                output_label[j,i] = -100 # If this location should not be masked
    return input_ids, output_label


class GeneralizedVLRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        self.cfg = cfg

        # visual encoder
        self.backbone = build_backbone(cfg)

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        self.language_backbone = build_language_backbone(cfg)
        self.encoder = PromptEncoder(self.language_backbone, self.backbone)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.DEBUG = cfg.MODEL.DEBUG

        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER
        if self.cfg.LPAI.PROMPT_LORA:
            if False:
                self.visual_prompt, self.textual_prompt = nn.ModuleList([MaPLePrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 96,768) for i in range(12)])
            else:
        #         visual_stack = []
        #         textual_stack = []
        #         for i in range(12):
        #             tmp_v, tmp_t = DecomposedPrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 96, 768, self.cfg.LPAI.PROMPT_LORA_D)
        #             visual_stack.append(tmp_v)
        #             textual_stack.append(tmp_t)
        #
        # #         self.visual_prompt = nn.ModuleList([DecomposedPrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 96, self.cfg.LPAI.PROMPT_LORA_D) for i in range(12)])
        # #         self.textual_prompt = nn.ModuleList([DecomposedPrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 768, self.cfg.LPAI.PROMPT_LORA_D) for i in range(12)])
        #         self.visual_prompt = nn.ModuleList(visual_stack)
        #         self.textual_prompt = nn.ModuleList(textual_stack)
                self.prompts = nn.ModuleList([DecomposedPrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 96, 768, self.cfg.LPAI.PROMPT_LORA_D) for i in range(12)])
        else:
            if self.cfg.LPAI.INTERACT_TYPE == 'maple':
                self.prompts = nn.ModuleList(
                    [MaPLePrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 96, 768) for i in range(12)])
            elif self.cfg.LPAI.INTERACT_TYPE == 'l2p':
                self.prompts = L2pPrompt(length=4, embed_dim=96,prompt_pool=True,pool_size=12, top_k=4, batchwise_prompt=True, prompt_key=True)
            else:
                self.visual_prompt = nn.ModuleList(
                    [NormalPrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 96, self.cfg.LPAI.PROMPT_LORA_D)
                     for i in range(12)])
                self.textual_prompt = nn.ModuleList(
                    [NormalPrompt(cfg.LPAI.PROMPT_DEPTH, self.cfg.LPAI.PROMPT_LENGTH, 768, self.cfg.LPAI.PROMPT_LORA_D)
                     for i in range(12)])

        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES

        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if hasattr(self.backbone, 'fpn'):
                assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, 'cls_logits'):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.parameters():
                p.requires_grad = False
        
        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS 
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

        if self.cfg.GLIPKNOW.KNOWLEDGE_FILE:
            from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
            self.class_name_to_knowledge = load_from_yaml_file(self.cfg.GLIPKNOW.KNOWLEDGE_FILE)
            self.class_name_list = sorted([k for k in self.class_name_to_knowledge])

        self.alignment_loss = ClipLoss()
        self.task_id = 0
        self.all_keys = []
        self.register_buffer('all_keys', self.all_keys)

    def getTaskIds(self, inputs):
        with torch.no_grad():
            if isinstance(self, nn.DataParallel):
                feature = self.module.extract_vector(inputs)
            else:
                feature = self.extract_vector(inputs)[-1]
                feature_shape = feature.shape
                feature = feature.view(feature_shape[0], -1)
                feature = feature / feature.norm(dim=-1, keepdim=True)

                taskselection = []
                for task_centers in self.all_keys: # task_centers: [5,512]
                    tmpcentersbatch = []
                    for center in task_centers:
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1)) # List[5] 128
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])

                selection = torch.vstack(taskselection).min(0)[1]

        return selection

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN, self).train(mode)
        if mode:
            for name, p in self.named_parameters():
                p.requires_grad_(False)
                if self.cfg.LPAI.VISUAL_PROMPT or self.cfg.LPAI.TEXTUAL_PROMPT:
                    if self.cfg.LPAI.INTERACT_TYPE == 'maple' or self.cfg.LPAI.INTERACT_TYPE == 'linear':
                        if 'prompts' + '.' + str(self.task_id) + '.' in name:
                            p.requires_grad_(True)
                            print(f'{name}: not frozen')
                    elif self.cfg.LPAI.INTERACT_TYPE == 'l2p' :
                        if 'prompts' in name:
                            p.requires_grad_(True)
                            print(f'{name}: not frozen')
                    else:
                        if 'visual_prompt' + '.' + str(self.task_id) + '.' in name or 'textual_prompt' + '.' + str(self.task_id) + '.' in name:
                            p.requires_grad_(True)
                            print(f'{name}: not frozen')
                if self.cfg.LPAI.INTERACT:
                    if 'interactModuleList' + '.' + str(self.task_id) + '.' in name:
                        p.requires_grad_(True)
                        print(f'{name}: not frozen')

    def update_task_id(self, task_id):
        self.task_id = task_id

    def forward(self, 
        images, 
        targets=None, 
        captions=None, 
        positive_map=None,
        greenlight_map=None,
        task_id=[0]):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # self.task_id = task_id[0]
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images) # [32,3,1280.800]
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device
        bs = images.tensors.shape[0]
        if self.training:
            if self.cfg.LPAI.INTERACT_TYPE == 'l2p':
                visual_prompt_exp = self.prompts
                textual_prompt_exp = None
            else:
                assert self.task_id == task_id[0], f'{self.task_id}, {task_id}'
                task_id = task_id * bs
                if self.cfg.LPAI.INTERACT_TYPE == 'maple' or self.cfg.LPAI.INTERACT_TYPE == 'linear':
                    visual_prompt, textual_prompt = self.prompts[task_id[0]]()
                    # visual_prompt, textual_prompt = self.prompts[task_id[0]]()
                else:
                    visual_prompt = self.visual_prompt[task_id[0]]()
                    textual_prompt = self.textual_prompt[task_id[0]]()
                visual_prompt_exp = visual_prompt.expand(bs, -1,-1,-1)
                textual_prompt_exp = textual_prompt.expand(bs, -1,-1,-1)

        else:
            if self.cfg.LPAI.INTERACT_TYPE == 'l2p':
                visual_prompt_exp = self.prompts
                textual_prompt_exp = None
            elif self.cfg.LPAI.INTERACT_TYPE == 'maple' or self.cfg.LPAI.INTERACT_TYPE == 'linear':
                visual_prompt_exp = torch.stack([i()[0] for i in self.prompts], 0)[task_id, :, :]
                textual_prompt_exp = torch.stack([i()[1] for i in self.prompts], 0)[task_id, :, :]
            else:
                visual_prompt_exp = torch.stack([i() for i in self.visual_prompt], 0)[task_id, :, :]
                textual_prompt_exp = torch.stack([i() for i in self.textual_prompt], 0)[task_id, :, :]
        if self.cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
            language_dict_features, positive_map = self._forward_language_parallel(
                    captions=captions, targets=targets, device=device,
                    positive_map=positive_map)
        else:
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions,
                                                            max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                            padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True,
                                                            return_tensors='pt',
                                                            truncation=True).to(device)
                if self.use_mlm_loss:
                    if not self.mlm_loss_for_only_positives:
                        greenlight_map = None
                    input_ids, mlm_labels = random_word(
                        input_ids=tokenized.input_ids, 
                        mask_token_id=self.tokenizer.mask_token_id,
                        vocabs=self.tokenizer_vocab_ids,
                        padding_token_id=self.tokenizer.pad_token_id,
                        greenlight_map=greenlight_map)
                else:
                    input_ids = tokenized.input_ids #[32, 256]
                    mlm_labels = None
                
                
                tokenizer_input = {"input_ids": input_ids,
                                "attention_mask": tokenized.attention_mask,
                                   "textual_prompt": textual_prompt_exp}

                # if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                #     with torch.no_grad():
                #         language_dict_features = self.language_backbone(tokenizer_input)
                # else:
                #     language_dict_features = self.language_backbone(tokenizer_input)
                
                # ONE HOT
                if self.cfg.DATASETS.ONE_HOT: # False
                    new_masks = torch.zeros_like(language_dict_features['masks'],
                                                device=language_dict_features['masks'].device)
                    new_masks[:, :self.cfg.MODEL.DYHEAD.NUM_CLASSES] = 1
                    language_dict_features['masks'] = new_masks

                # MASK ALL SPECIAL TOKENS
                if self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL: # False
                    language_dict_features["masks"] = 1 - tokenized.special_tokens_mask
                
                # language_dict_features["mlm_labels"] = mlm_labels

        # visual embedding
        swint_feature_c4 = None
        if 'vl' in self.cfg.MODEL.SWINT.VERSION:
            # the backbone only updates the "hidden" field in language_dict_features
            inputs = {"img": images.tensors, "lang": language_dict_features}
            visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
        else:
            vis_input = {
                'images': images.tensors,
                'visual_prompt': visual_prompt_exp
            }
            # visual_features = self.backbone(images.tensors)
            # visual_features = self.backbone(vis_input)

        language_dict_features, visual_features = self.encoder(tokenizer_input, vis_input, task_id)
        language_dict_features["mlm_labels"] = mlm_labels

        # rpn force boxes
        if targets:
            targets = [target.to(device)
                       for target in targets if target is not None]

        if self.force_boxes: # False
        # if True:
            proposals = []
            for t in targets:
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                proposals.append(tb)
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                _, proposal_losses, fused_visual_features = self.rpn(
                    images, visual_features, targets, language_dict_features,
                    positive_map, captions, swint_feature_c4)
            elif self.training:
                null_loss = 0
                for key, param in self.rpn.named_parameters():
                    null_loss += 0.0 * param.sum()
                proposal_losses = {('rpn_null_loss', null_loss)}
        else:
            proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map,
                                              captions, swint_feature_c4)
        if self.roi_heads:
            if self.cfg.MODEL.ROI_MASK_HEAD.PREDICTOR.startswith("VL"):
                if self.training:
                    # "Only support VL mask head right now!!"
                    assert len(targets) == 1 and len(targets[0]) == len(positive_map), "shape match assert for mask head!!"
                    # Not necessary but as a safe guard:
                    # use the binary 0/1 positive map to replace the normalized positive map
                    targets[0].add_field("positive_map", positive_map)
            # TODO: make sure that this use of language_dict_features is correct!! Its content should be changed in self.rpn
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                x, result, detector_losses = self.roi_heads(
                    fused_visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
            else:
                x, result, detector_losses = self.roi_heads(
                    visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
        else:
            # RPN-only models don't have roi_heads
            x = visual_features
            result = proposals
            detector_losses = {}
        # print(detector_losses)
        # print(proposal_losses)
        if self.training:
            # bs,12,16,512 , bs,12,16,96
            # temperature = 0.1
            if self.cfg.LPAI.INTERACT_TYPE == 'l2p':
                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
            else:
                visual_prompt_for_loss = torch.mean(visual_prompt, -1)
                visual_prompt_for_loss = visual_prompt_for_loss / visual_prompt_for_loss.norm(dim=-1, keepdim=True)
                textual_prompt_for_loss = torch.mean(textual_prompt, -1)
                textual_prompt_for_loss = textual_prompt_for_loss / textual_prompt_for_loss.norm(dim=-1, keepdim=True)
                # visual_prompt_for_loss = visual_prompt_for_loss / temperature
                # textual_prompt_for_loss = textual_prompt_for_loss / temperature
                # visual_prompt = torch.nn.functional.normalize(visual_prompt,dim=1)
                # textual_prompt = torch.nn.functional.normalize(textual_prompt, dim=1)
                layer_score = 100 * visual_prompt_for_loss @ textual_prompt_for_loss.t()

                layer_loss = {'alignment_loss': self.alignment_loss(layer_score)*0.1}

                losses = {}
                losses.update(detector_losses)
                # assert detector_losses.require_grad == True
                for itm in proposal_losses:
                    proposal_losses[itm] *= 0.8
                losses.update(proposal_losses)
                # assert proposal_losses.require_grad == True
                if self.cfg.LPAI.LAYER_ALIGNMENT:
                    losses.update(layer_loss)
                if self.cfg.LPAI.TASK_ALIGNMENT:
                    if task_id[0] != 0:
                        task_loss = {'task_loss': self.cal_task_loss(task_id, visual_prompt_for_loss, textual_prompt_for_loss)*0.1}
                        losses.update(task_loss)
            return losses
        return result
    # def cal_dif(self, v1, v2):
    #     return torch.sum((v1-v2)**2)

    # def cal_sim(self, v1, v2):
    #     return torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))

    def interface_visual(self, task_id):
        return self.prompts[task_id]()[0]

    def interface_textual(self, task_id):
        return self.prompts[task_id]()[1]

    def extract_vector(self, images):
        images = to_image_list(images)
        device = images.tensors.device
        vis_input = {
            'images': images.tensors
        }
        visual_features = self.backbone(vis_input)
        return visual_features

    def clustering(self, dataloader,device):
        features = []
        for iteration, (inputs, _, _, _, _, _) in enumerate(dataloader):
            # device = inputs.tensors.device
            inputs = inputs.to(device)
            with torch.no_grad():
                if isinstance(self, nn.DataParallel):
                    feature = self.module.extract_vector(inputs)
                else:
                    feature = self.extract_vector(inputs)[-1]
                    feature_shape = feature.shape
                    feature = feature.view(feature_shape[0], -1)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def cal_task_loss(self, task_id, visual_prompt, textual_prompt): # [12,16], [12,16]
        task_id = task_id[0]
        device = visual_prompt.device
        # import numpy as np
        task_sim_matrix = np.loadtxt('./MID/task_sim_matrix.txt')
        task_sim_matrix = torch.tensor(task_sim_matrix[:task_id+1, :task_id+1]).to(device)
        threshold = 0.4
        task_sim_matrix = (task_sim_matrix>threshold).type(torch.int)
        visual_prompt_stack = torch.stack([self.prompts[i]()[0].view(-1) for i in range(task_id+1)])
        textual_prompt_stack = torch.stack([self.prompts[i]()[1].view(-1) for i in range(task_id+1)])
        # visual_prompt_stack_org = []
        # textual_prompt_stack_org = []
        # for i in range(task_id+1):
        #     vv = self.visual_prompt[i]()
        #     vv = torch.mean(vv, -1)
        #     vv = vv/ vv.norm(dim=-1, keepdim=True)
        #     vv = vv.view(-1)
        #     visual_prompt_stack_org.append(vv)

        #     tt = self.textual_prompt[i]()
        #     tt = torch.mean(tt, -1)
        #     tt = tt / tt.norm(dim=-1, keepdim=True)
        #     tt = tt.view(-1)
        #     textual_prompt_stack_org.append(tt)
        # visual_prompt_stack = torch.stack(visual_prompt_stack_org)
        # textual_prompt_stack = torch.stack(textual_prompt_stack_org)
        # visual_prompt_stack = torch.stack(
        #     [torch.mean(self.visual_prompt[i](), -1).view(-1) for i in range(task_id + 1)])
        # textual_prompt_stack = torch.stack(
        #     [torch.mean(self.textual_prompt[i](), -1).view(-1) for i in range(task_id + 1)])
        # visual_sim = torch.tensor([torch.sum(torch.cosine_similarity(torch.mean(self.visual_prompt[i](), -1), visual_prompt))/12 for i in range(task_id)]).to(device)
        # textual_sim = torch.tensor([torch.sum(torch.cosine_similarity(torch.mean(self.textual_prompt[i](), -1),textual_prompt))/12 for i in range(task_id)]).to(device)

        # visual_sim[torch.eye(visual_sim.size(0)).bool()] = float("inf")
        # textual_sim[torch.eye(textual_sim.size(0)).bool()] = float("inf")
        # mse_loss = torch.nn.MSELoss()
        # visual_prompt_stack = visual_prompt_stack / temperature
        # textual_prompt_stack = textual_prompt_stack / temperature
        temperature = 0.01
        return (nt_bxent_loss(visual_prompt_stack, task_sim_matrix, temperature) + nt_bxent_loss(textual_prompt_stack, task_sim_matrix, temperature)) / 2


    def _forward_language_parallel(self, captions=None, targets=None,
            device=None, positive_map=None):
        ktype = self.cfg.GLIPKNOW.KNOWLEDGE_TYPE
        def _construct_captions_from_class_names(class_names):
            captions = []
            for c in class_names:
                try:
                    info = self.class_name_to_knowledge[c]
                    cap = info['clean_name']

                    # combine wiki and gpt3 knowledge
                    if self.cfg.GLIPKNOW.WIKI_AND_GPT3:
                        ktype = 'def_wiki'
                        know_seq = info[ktype]

                        ktype = 'gpt3'
                        if ktype == 'gpt3' or type(info[ktype]) == list:
                            know_seq += ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM] ])

                        cap += ': ' + know_seq

                    # only one knoweldge source is used        
                    else:
                        if ktype and ktype in info and info[ktype]:
                            if ktype == 'gpt3' or type(info[ktype]) == list:
                                know_seq = ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM] ])
                            else: 
                                know_seq = info[ktype]
                            cap += ': ' + know_seq
                except:
                    cap = c
                    print(f'cap {cap}, c {c}')
                    
                    
                captions.append(cap)
            return captions

        if self.training:
            assert captions is None
            assert targets is not None

            max_classes_per_batch = self.cfg.GLIPKNOW.MAX_NUM_CLASSES_PER_BATCH_TRAIN
            if max_classes_per_batch >= len(self.class_name_list):
                shuffled_class_names = self.class_name_list.copy()
                random.shuffle(shuffled_class_names)
                if max_classes_per_batch > len(shuffled_class_names):
                    shuffled_class_names.extend(shuffled_class_names[:max_classes_per_batch
                        -len(shuffled_class_names)])
                    random.shuffle(shuffled_class_names)
            else:
                label_list = []
                label_to_idx = {}
                for target_per_im in targets:
                    labels_per_im = target_per_im.get_field('label_names')
                    for label in labels_per_im:
                        if label not in label_to_idx:
                            label_to_idx[label] = len(label_list)
                            label_list.append(label)

                label_list = label_list[:max_classes_per_batch]
                if len(label_list) < max_classes_per_batch:
                    all_neg_classes = [c for c in self.class_name_list if c not
                            in label_to_idx]
                    neg_label_list = random.sample(all_neg_classes,
                            max_classes_per_batch - len(label_list))
                    label_list.extend(neg_label_list)
                random.shuffle(label_list)
                shuffled_class_names = label_list

            label_to_shuffled_idx = {l: i for i, l in
                    enumerate(shuffled_class_names)}
            total_boxes = sum(len(t) for t in targets)
            positive_map = torch.zeros((total_boxes, max_classes_per_batch+1),
                device=device)
            offset = 0
            for target_per_im in targets:
                labels_per_im = target_per_im.get_field('label_names')
                for label in labels_per_im:
                    j = label_to_shuffled_idx.get(label, -1)
                    if j >= 0:
                        positive_map[offset, j] = 1
                    offset += 1
            captions = _construct_captions_from_class_names(shuffled_class_names)
            captions.append('') # onobj at the end, onedet/modeling/rpn/loss.py:719
            batch_size = len(targets)

        else:
            assert captions is not None
            batch_size = 1
            assert len(captions) == 1
            class_names = captions[0]
            max_classes_per_batch = len(class_names)
            captions = _construct_captions_from_class_names(class_names)
            captions.append('') # onobj at the end, onedet/modeling/rpn/loss.py:719

        tokenized = self.tokenizer.batch_encode_plus(captions,
                                                     max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                     padding="longest",
                                                     return_special_tokens_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True).to(device)
        assert not self.use_mlm_loss
        tokenizer_input = {"input_ids": tokenized.input_ids,
                           "attention_mask": tokenized.attention_mask}

        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)
        else:
            language_dict_features = self.language_backbone(tokenizer_input)

        assert not self.cfg.DATASETS.ONE_HOT
        assert not self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL

        agg_type = self.cfg.GLIPKNOW.LAN_FEATURE_AGG_TYPE
        agg_feats = language_dict_features['hidden']
        agg_emb = language_dict_features['embedded']
        if agg_type == 'first':
            agg_feats = agg_feats[:, 0, :]
            agg_emb = agg_emb[:, 0, :]
        elif agg_type == 'mean':
            attn_mask = language_dict_features['masks']
            seq_len = attn_mask.sum(-1).unsqueeze(-1).float()
            agg_feats = agg_feats * attn_mask.unsqueeze(-1).float()
            agg_feats = agg_feats.sum(1) / seq_len
            agg_emb = agg_emb * attn_mask.unsqueeze(-1).float()
            agg_emb = agg_emb.sum(1) / seq_len
        else:
            raise ValueError('not supported GLIPKNOW.LAN_FEATURE_AGG_TYPE: {}'.format(agg_type))

        expanded_features = agg_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        expanded_embedding = agg_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        lang_dict = {}
        lang_dict["mlm_labels"] = None
        lang_dict["aggregate"] = None
        lang_dict["embedded"] = expanded_embedding
        lang_dict['hidden'] = expanded_features
        lang_dict["masks"] = torch.ones((batch_size, max_classes_per_batch+1),
                device=device, dtype=language_dict_features['masks'].dtype)
        # in GLIP setting, the token at the end of seqence is usually [PAD], and is masked out
        # if [noobj] is not masked out, the loss sum is very big, as most
        # anchors are matched to [noobj]
        lang_dict["masks"][:,-1] = 0
        return lang_dict, positive_map

import math
class DecomposedPromptTest(nn.Module):
    def __init__(self, layer_num, prompt_num, prompt_depth_vis, prompt_depth_text, r=4):
        super().__init__()
        self.d = r

        self.decomposed_prompt_visual = nn.Parameter(torch.randn(layer_num,prompt_num, prompt_depth_vis))
        self.decomposed_prompt_textual = nn.Parameter(torch.randn(layer_num,prompt_num, prompt_depth_text))

        nn.init.normal_(self.decomposed_prompt_visual, std=0.125)
        nn.init.normal_(self.decomposed_prompt_visual, std=0.125)
        # self.layerNorm = nn.LayerNorm(prompt_depth)
        self.scale = 1

    def forward(self):
        return self.decomposed_prompt_visual, self.decomposed_prompt_textual

class DecomposedPrompt(nn.Module):
    def __init__(self, layer_num, prompt_num, prompt_depth_vis, prompt_depth_text, r=4):
        super().__init__()
        self.d = r
        # d1 = torch.randn(layer_num,1,1, self.d)
        # d2 = torch.randn(1, prompt_num,1,self.d)
        # d3 = torch.rand(1,1,prompt_depth, self.d)
        d1_share = torch.randn(layer_num, self.d)
        d2_visual = torch.randn(prompt_num,self.d)
        d2_textual = torch.randn(prompt_num,self.d)
        d3_visual = torch.rand(prompt_depth_vis, self.d)
        d3_textual = torch.rand(prompt_depth_text, self.d)
        # nn.init.xavier_uniform_(d1)
        # nn.init.xavier_uniform_(d2)
        # nn.init.xavier_uniform_(d3)

        # torch.nn.init.kaiming_uniform_(d2)
        # torch.nn.init.kaiming_uniform_(d3)

        self.dim_1_share = nn.Parameter(d1_share)
        self.dim_2_visual = nn.Parameter(d2_visual)
        self.dim_2_textual = nn.Parameter(d2_textual)
        self.dim_3_visual = nn.Parameter(d3_visual)
        self.dim_3_textual = nn.Parameter(d3_textual)

        # torch.nn.init.kaiming_uniform_(self.dim_1, a=math.sqrt(5))
        # torch.nn.init.kaiming_uniform_(self.dim_2, a=math.sqrt(5))
        # nn.init.zeros_(self.dim_2)
        # nn.init.zeros_(self.dim_3)
        # torch.nn.init.kaiming_uniform_(self.dim_1, a=3)
        # torch.nn.init.kaiming_uniform_(self.dim_2, a=3)
        # torch.nn.init.kaiming_uniform_(self.dim_3, a=3)
        nn.init.normal_(self.dim_1_share, std=0.5)
        nn.init.normal_(self.dim_2_visual, std=0.5)
        nn.init.normal_(self.dim_2_textual, std=0.5)
        nn.init.normal_(self.dim_3_visual, std=0.5)
        nn.init.normal_(self.dim_3_textual, std=0.5)
        # self.layerNorm = nn.LayerNorm(prompt_depth)
        self.scale = 1
        # self.scale = 1

    def interface(self):
        dim_1 = self.dim_1.view(-1,1,1,self.d)
        dim_2 = self.dim_2.view(1, -1, 1, self.d)
        dim_3 = self.dim_3.view(1, 1, -1, self.d)
        decomposed_prompt = torch.mul(torch.mul(dim_1, dim_2), dim_3)
        decomposed_prompt = torch.mean(decomposed_prompt, dim=3)
        # decomposed_prompt = self.layerNorm(decomposed_prompt)
        return decomposed_prompt

    def forward(self):
        # d1
        dim_1_share = self.dim_1_share.view(-1,1,1,self.d)
        # d2
        dim_2_visual = self.dim_2_visual.view(1, -1, 1, self.d)
        dim_2_textual = self.dim_2_textual.view(1, -1, 1, self.d)
        # d3
        dim_3_visual = self.dim_3_visual.view(1, 1, -1, self.d)
        dim_3_textual = self.dim_3_textual.view(1, 1, -1, self.d)

        decomposed_prompt_visual = torch.mul(torch.mul(dim_1_share, dim_2_visual), dim_3_visual)
        decomposed_prompt_visual = torch.mean(decomposed_prompt_visual, dim=3)*self.scale

        decomposed_prompt_textual = torch.mul(torch.mul(dim_1_share, dim_2_textual), dim_3_textual)
        decomposed_prompt_textual = torch.mean(decomposed_prompt_textual, dim=3)*self.scale
        # decomposed_prompt = decomposed_prompt * self.scale
        # norm_flag = False
        # if norm_flag:
        #     decomposed_prompt = self.layerNorm(decomposed_prompt)
        return decomposed_prompt_visual, decomposed_prompt_textual

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MaPLePrompt(nn.Module):
    def __init__(self, layer_num, prompt_num, vis_depth, text_depth):
        super().__init__()
        self.textual_prompt = nn.ParameterList([nn.Parameter(torch.empty(prompt_num, text_depth))
                                                      for _ in range(layer_num)])
        for single_text_prompt in self.textual_prompt:
            nn.init.normal_(single_text_prompt, std=0.02)

        self.visual_prompt = []
        single_layer = nn.Linear(text_depth, vis_depth)
        self.compound_prompt_projections = _get_clones(single_layer, layer_num)


        # self.scale = 1
    def interface(self):
        # dim_1 = self.dim_1.view(-1,1,1,self.d)
        # dim_2 = self.dim_2.view(1, -1, 1, self.d)
        # dim_3 = self.dim_3.view(1, 1, -1, self.d)
        # decomposed_prompt = torch.mul(torch.mul(dim_1, dim_2), dim_3)
        # decomposed_prompt = torch.mean(decomposed_prompt, dim=3)
        # decomposed_prompt = self.layerNorm(decomposed_prompt)
        return [self.textual_prompt, self.visual_prompt]

    def forward(self):
        visual_stack = []
        textual_stack = []
        for index, layer in enumerate(self.compound_prompt_projections):
            # self.visual_prompt.append(layer(self.textual_prompt[index]))
            visual_stack.append(layer(self.textual_prompt[index]))
            textual_stack.append(self.textual_prompt[index])
        # decomposed_prompt = decomposed_prompt * self.scale
        # norm_flag = False
        # if norm_flag:
        #     decomposed_prompt = self.layerNorm(decomposed_prompt)
        return torch.stack(visual_stack, dim=0), torch.stack(textual_stack, dim=0)
class NormalPrompt(nn.Module):
    def __init__(self, layer_num, prompt_num, prompt_depth, r=4):
        super().__init__()

        self.visual_prompt = nn.Parameter(torch.randn(layer_num, prompt_num, prompt_depth))

        nn.init.normal_(self.visual_prompt, std=0.02)

        # self.layerNorm = nn.LayerNorm(prompt_depth)
        self.scale = 1
        # self.scale = 1
    def interface(self):
        # decomposed_prompt = self.layerNorm(decomposed_prompt)
        return self.visual_prompt

    def forward(self):
        # decomposed_prompt = decomposed_prompt * self.scale
        # norm_flag = False
        # if norm_flag:
        #     decomposed_prompt = self.layerNorm(decomposed_prompt)
        return self.visual_prompt

class L2pPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', ):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        device = x_embed.device
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
            self.to(device)
            # self.prompt_key.to(device)
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)
            prompt_norm = prompt_norm.to(device)# Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size

            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],),
                                                                     torch.min(idx.flatten()),
                                                                     device=device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0,
                                                                     device=device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        # out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        x_embed[:,:out['total_prompt_len']] = batched_prompt
        out['prompted_embedding'] = x_embed
        return out

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        self.margin = 0.2
        self.max_violation = True

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, logits):
        device = logits.device
        logits_per_image = logits
        logits_per_text = logits.transpose(0, 1)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2


        return total_loss


