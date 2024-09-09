import torch
from torch import nn

class DecomposedPrompt(nn.Module):
    def __init__(self, layer_num, prompt_num, prompt_depth_vis, prompt_depth_text, r=4):
        super().__init__()
        self.d = r

        d1_share = torch.randn(layer_num, self.d)
        d2_visual = torch.randn(prompt_num,self.d)
        d2_textual = torch.randn(prompt_num,self.d)
        d3_visual = torch.rand(prompt_depth_vis, self.d)
        d3_textual = torch.rand(prompt_depth_text, self.d)

        self.dim_1_share = nn.Parameter(d1_share)
        self.dim_2_visual = nn.Parameter(d2_visual)
        self.dim_2_textual = nn.Parameter(d2_textual)
        self.dim_3_visual = nn.Parameter(d3_visual)
        self.dim_3_textual = nn.Parameter(d3_textual)

        nn.init.normal_(self.dim_1_share, std=0.5)
        nn.init.normal_(self.dim_2_visual, std=0.5)
        nn.init.normal_(self.dim_2_textual, std=0.5)
        nn.init.normal_(self.dim_3_visual, std=0.5)
        nn.init.normal_(self.dim_3_textual, std=0.5)
        # self.layerNorm = nn.LayerNorm(prompt_depth)
        self.scale = 1

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


class NormalPrompt(nn.Module):
    def __init__(self, layer_num, prompt_num, visual_prompt_depth, textual_prompt_depth):
        super().__init__()

        self.visual_prompt = nn.Parameter(torch.randn(layer_num, prompt_num, visual_prompt_depth))
        self.textual_prompt = nn.Parameter(torch.randn(layer_num, prompt_num, textual_prompt_depth))
        nn.init.normal_(self.visual_prompt, std=0.02)
        nn.init.normal_(self.textual_prompt, std=0.02)

    def forward(self):
        return self.visual_prompt, self.textual_prompt


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