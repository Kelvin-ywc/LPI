import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import threading
import queue
from typing import Optional, List, Union, Tuple
from maskrcnn_benchmark.modeling.bert.modeling_bert import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions


class PromptEncoder(nn.Module):
    def __init__(self, language_backbone, backbone):
        super().__init__()
        self.language_backbone = language_backbone
        self.backbone = backbone

        # language attribute
        self.language_encoder = language_backbone.body # BertEncoder
        self.cfg = self.language_encoder.cfg
        self.textual_num_layers = self.language_encoder.num_layers
        self.model = self.language_encoder.model

        # bert model attribute
        self.config = self.language_encoder.model.config
        self.embeddings = self.language_encoder.model.embeddings
        self.encoder = self.language_encoder.model.encoder
        self.pooler = self.language_encoder.model.pooler
        # visual attribute
        self.visual_encoder = backbone.body
        self.patch_embed = self.visual_encoder.patch_embed
        self.ape = self.visual_encoder.ape
        # self.absolute_pos_embed = self.visual_encoder.absolute_pos_embed
        self.pos_drop = self.visual_encoder.pos_drop
        self.num_layers = self.visual_encoder.num_layers
        self.layers = self.visual_encoder.layers
        self.out_features = self.visual_encoder.out_features
        self.num_features = self.visual_encoder.num_features

        for i_layer in range(self.num_layers):
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, getattr(self.visual_encoder, layer_name))
        self.fpn = backbone.fpn

    def language_forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]
        textual_prompt = x["textual_prompt"]
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS: # True

            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask,
                output_hidden_states=True,
                textual_prompt=textual_prompt
            )
            # outputs = self.bert_forward(input_ids=input,attention_mask=mask, output_hidden_states=True)
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]
            features = None
            features = torch.stack(encoded_layers[-self.textual_num_layers:], 1).mean(1)

            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.textual_num_layers # 32, 256, 768

            embedded = features * mask.unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        else:
            # without padding, only consider positive_tokens
            max_len = (input != 0).sum(1).max().item()
            outputs = self.model(
                input_ids=input[:, :max_len],
                attention_mask=mask[:, :max_len],
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]

            features = None
            features = torch.stack(encoded_layers[-self.textual_num_layers:], 1).mean(1)
            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.textual_num_layers

            embedded = features * mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate, # [32, 768]
            "embedded": embedded, # [32, 256, 768]
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret


    def visual_forward(self, x):
        Wh, Ww, visual_prompt, x = self.visual_prehandle(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, visual_prompt)
            name = f'stage{i + 2}'
            if name in self.out_features:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def visual_prehandle(self, x):
        visual_prompt = x['visual_prompt']
        x = x['images']
        """Forward function."""
        x = self.patch_embed(x)  # [32, 3, 1280, 800] => [32, 96, 320, 200]
        Wh, Ww = x.size(2), x.size(3)  # 320, 200
        if self.ape:  # False
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)  # [32, 64000, 96]
        x = self.pos_drop(x)
        return Wh, Ww, visual_prompt, x

    def forward2(self, textual_input, visual_input):
        t1 = threading.Thread(target=self.visual_forward, args=(visual_input))
        t2 = threading.Thread(target=self.language_forward, args=(textual_input))
        # visual_feature = self.visual_forward(visual_input)
        # textual_feature = self.language_forward(textual_input)
        t1.start()
        t2.start()
        visual_feature = t1.join()
        textual_feature = t2.join()
        return visual_feature, textual_feature

    def forward(self, textual_input=None, visual_input=None, task_id=None):
        # visual_feature = self.fpn(self.visual_forward(visual_input))

        # visual_feature = self.visual_forward(visual_input)
        # visual pre-handle



        # textual_feature = self.language_backbone.body(textual_input)
        # textual_feature = self.language_forward(textual_input)
        input = textual_input["input_ids"]
        mask = textual_input["attention_mask"]
        textual_prompt = textual_input["textual_prompt"]

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS: # True
            # visual
            outputs, outs = self.forward_interact(textual_input, visual_input, task_id)
            # outputs = self.bert_forward(input_ids=input,attention_mask=mask, output_hidden_states=True)
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]
            features = None
            features = torch.stack(encoded_layers[-self.textual_num_layers:], 1).mean(1)

            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.textual_num_layers # 32, 256, 768

            embedded = features * mask.unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        else:
            # without padding, only consider positive_tokens
            max_len = (input != 0).sum(1).max().item()
            outputs = self.model(
                input_ids=input[:, :max_len],
                attention_mask=mask[:, :max_len],
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]

            features = None
            features = torch.stack(encoded_layers[-self.textual_num_layers:], 1).mean(1)
            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.textual_num_layers

            embedded = features * mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        textual_feature = {
            "aggregate": aggregate, # [32, 768]
            "embedded": embedded, # [32, 256, 768]
            "masks": mask,
            "hidden": encoded_layers[-1]
        }

        # for i in range(self.num_layers):
        #     layer = self.layers[i]
        #     x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, visual_prompt)
        #     name = f'stage{i + 2}'
        #     if name in self.out_features:
        #         norm_layer = getattr(self, f'norm{i}')
        #         x_out = norm_layer(x_out)
        #         out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
        #         outs.append(out)

        visual_feature = self.fpn(outs)
        # visual_feature = self.backbone(visual_input)
        # visual_feature = self.backbone.fpn(visual_feature)
        return textual_feature, visual_feature

    def forward_interact(self, textual_input, visual_input, task_id):
        # visual_output = self.visual_forward(visual_input)
        input = textual_input["input_ids"]
        mask = textual_input["attention_mask"]
        textual_prompt = textual_input["textual_prompt"]
        textual_output, visual_output = self.model(
            input_ids=input,
            attention_mask=mask,
            output_hidden_states=True,
            textual_prompt=textual_prompt,
            visual_encoder=self.visual_encoder,
            visual_input=visual_input,
            task_id=task_id
        )

        return textual_output, visual_output


