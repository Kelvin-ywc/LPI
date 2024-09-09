import torch
import torch.nn.functional as F
from torch import nn

# from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
# from models.modeling.backbone import build_backbone, MLP

# from transformers import RobertaModel, BertModel
# from models.reftr import build_vl_transformer
# from models.criterion import CriterionVGOnePhrase, CriterionVGMultiPhrase
# from models.post_process import PostProcessVGOnePhrase, PostProcessVGMultiPhrase


import copy
import torch
import torch.nn.functional as F

from typing import Optional, List
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch import nn, Tensor
from models.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, \
    TransformerEncoderLayer


class VLTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_feature_levels=1,
                 num_queries=1, return_intermediate_dec=False, max_lang_seq=128):
        super().__init__()
        # Positional embedding and feat type embedding
        # token type embedding to indicate image feature vs language feature
        self.max_lang_seq = max_lang_seq
        self.num_queries = num_queries
        self.d_model = d_model
        self.nhead = nhead
        self.lang_pos_embeddings = nn.Embedding(max_lang_seq, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)

        # Transformer Encoder as encoder
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # if num_decoder_layers < 0, no decoder is used
        self.use_decoder = num_decoder_layers > 0
        if self.use_decoder:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)
        else:
            print("No decoder is used!")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embed)

    def process_img_feat(self, img_srcs, img_masks, img_pos_embeds):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(img_srcs, img_masks, img_pos_embeds)):
            bs, c, h, w = src.shape
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        img_src_flatten = torch.cat(src_flatten, 1)  # .transpose(0, 1)
        img_mask_flatten = torch.cat(mask_flatten, 1)  # .transpose(0, 1)
        img_lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # .transpose(0, 1)

        # Add token type embedding if available
        bsz, seq_length, dim = img_src_flatten.shape
        if self.token_type_embeddings is not None:
            token_type_ids = torch.ones((bsz, seq_length), dtype=torch.long, device=img_src_flatten.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            img_lvl_pos_embed_flatten = img_lvl_pos_embed_flatten + token_type_embeddings

        return img_mask_flatten, \
               img_src_flatten.transpose(0, 1), \
               img_lvl_pos_embed_flatten.transpose(0, 1)

    def process_lang_feat(self, lang_srcs, lang_masks):
        bsz, seq_length, dim = lang_srcs.shape
        assert seq_length <= self.max_lang_seq
        position_ids = torch.arange(seq_length, dtype=torch.long, device=lang_srcs.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.lang_pos_embeddings(position_ids)

        if self.token_type_embeddings is not None:
            token_type_ids = torch.zeros((bsz, seq_length), dtype=torch.long, device=lang_srcs.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            position_embeddings = position_embeddings + token_type_embeddings

        # Non-zero area is ignored
        lang_masks = lang_masks.logical_not()
        assert (lang_masks[:, 0] == False).all()

        return lang_masks, \
               lang_srcs.transpose(0, 1), \
               position_embeddings.transpose(0, 1)

    def encode(self, img_srcs, img_masks, img_pos_embeds,
               lang_srcs, lang_masks, category=None):
        # create image feature and mask & pos info

        # img_masks, img_srcs, img_pos_embeds = \
        #     self.process_img_feat(img_srcs, img_masks, img_pos_embeds)
        #
        # lang_masks, lang_srcs, lang_pos_embeds = \
        #     self.process_lang_feat(lang_srcs, lang_masks)
        #
        #
        img_masks = torch.ones([img_srcs.size(0), img_srcs.size(1)]).cuda()
        masks = torch.cat([lang_masks, img_masks], dim=1)
        lang_srcs = lang_srcs.transpose(0, 1)
        img_srcs = img_srcs.transpose(0, 1)
        srcs = torch.cat([lang_srcs, img_srcs], dim=0)
        # pos_embeds = torch.cat([lang_pos_embeds, img_pos_embeds], dim=0)

        memory = self.encoder(srcs,src_key_padding_mask=masks).transpose(0, 1)#, src_key_padding_mask=masks, pos=pos_embeds, category=category)
        return memory#, masks, pos_embeds

    def forward(self, img_srcs, img_masks, img_pos_embeds,
                lang_srcs, lang_masks,
                query=None, query_mask=None, query_pos=None, category=None):

        memory = \
            self.encode(img_srcs, img_masks, img_pos_embeds, lang_srcs, lang_masks, category)

        if False and self.use_decoder:
            # TODO here
            hs = self.decoder(query, memory,
                              memory_key_padding_mask=masks,
                              tgt_key_padding_mask=query_mask,
                              pos=pos_embeds, query_pos=query_pos)
        else:
            hs = memory
        return hs


def mlp_mapping(input_dim, output_dim):
    return torch.nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(output_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
    )


class QueryEncoder(nn.Module):
    def __init__(self, num_queries_per_phrase, hidden_dim, ablation):
        super(QueryEncoder, self).__init__()
        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries_per_phrase, hidden_dim*2)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.fuse_encoder_query = mlp_mapping(hidden_dim*2, hidden_dim)
        self.context_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, lang_context_feat, lang_query_feat, mask_query_context):
        learnable_querys = self.query_embed.weight
        bs, n_ph, _ = lang_query_feat.shape
        n_q = learnable_querys.size(0)
        # n_context = lang_context_feat.size(1)

        # attended reduce
        k = self.linear1(lang_context_feat[:, 0:1, :])
        q = self.linear2(lang_context_feat).transpose(1, 2)
        v = self.linear3(lang_context_feat).unsqueeze(1)                     # b, 1, n_context, -1
        att_weight = torch.bmm(k, q)
        att_weight = att_weight.expand(-1, n_ph, -1)
        att_weight = att_weight.masked_fill(mask_query_context, float('-inf'))
        att_weight_normalized = F.softmax(att_weight, dim=-1).unsqueeze(-1)  # b, n_ph, n_context, -1
        context_feats = self.context_out((v * att_weight_normalized).sum(dim=-2))              # b, n_ph, -1

        # residual connection
        context_feats = lang_context_feat[:, None, 0, :] + context_feats

        lang_query_feat = torch.cat([context_feats, lang_query_feat], dim=-1)
        lang_query_feat = self.fuse_encoder_query(lang_query_feat)
        phrase_queries = lang_query_feat.view(bs, n_ph, 1, -1).repeat(1, 1, 1, 2) +\
            learnable_querys.view(1, 1, n_q, -1)
        phrase_queries = phrase_queries.view(bs, n_ph*n_q, -1).transpose(0, 1)

        return torch.split(phrase_queries, self.hidden_dim, dim=-1)


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights
