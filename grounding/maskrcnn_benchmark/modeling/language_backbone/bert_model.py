from copy import deepcopy
import numpy as np
import torch
from torch import nn

# from pytorch_pretrained_bert.modeling import BertModel
from transformers import RobertaConfig, RobertaModel
from maskrcnn_benchmark.modeling.bert.configuration_bert import BertConfig
from maskrcnn_benchmark.modeling.bert.modeling_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()
        self.cfg = cfg
        self.bert_name = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT)

        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            config.visual_prompt = self.cfg.LPAI.VISUAL_PROMPT
            config.textual_prompt = self.cfg.LPAI.TEXTUAL_PROMPT
            config.interact = self.cfg.LPAI.INTERACT
            config.prompt_depth = self.cfg.LPAI.PROMPT_DEPTH
            config.lora_r = self.cfg.LPAI.INTERACT_LORA_D
            config.interact_depth = self.cfg.LPAI.INTERACT_DEPTH
            config.interact_type = self.cfg.LPAI.INTERACT_TYPE
            self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        # bert attr
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.pooler = self.model.pooler


    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]
        textual_prompt = x["textual_prompt"]
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:

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
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers # 32, 256, 768

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
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)
            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate, # [32, 768]
            "embedded": embedded, # [32, 256, 768]
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret
