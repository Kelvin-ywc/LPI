import torch
import torch.nn as nn

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg['backbonename']
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {}

    if cfg["net_type"] == "slip_cmpa" or cfg["net_type"] == "slip_ampl":
        design_details = {
            "trainer": 'CMPA',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "cmpa_length": cfg['cmpa_length'],
            "fusing": cfg['fusing'],
            "parameter_sharing": cfg['parameter_sharing'],
            "total_sessions": cfg['total_sessions']
        }

    model = clip.build_model(state_dict or model.state_dict(),cfg, design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, textual_prompt):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, textual_prompt)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        # n_cls = len(classnames)
        n_cls = 1

        n_ctx = cfg.NCTX # number of context vectors
        ctx_init = cfg.CTXINIT
        self.clip_model = clip_model
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = clip_imsize
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        device = clip_model.token_embedding.weight.device
        self.device = device
        self.ctx = nn.Parameter(ctx_vectors).to(device)  # to be optimized

        self.n_ctx = n_ctx


        self.n_cls = None
        self.caption_lens = None
        self.name_lens = None
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def extract_vector(self, captions):
        self.n_cls = len(captions)
        self.caption_lens = [len(_tokenizer.encode(caption)) for caption in captions]
        prompts = [self.prompt_prefix + " " + caption + "." for caption in captions ]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        self.name_lens = self.caption_lens
        return embedding, tokenized_prompts

    def forward(self, captions, ctx):
        self.n_cls = len(captions)
        self.caption_lens = [len(_tokenizer.encode(caption)) for caption in captions]
        prompts = [self.prompt_prefix + " " + caption + "." for caption in captions ]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.name_lens = self.caption_lens

        if ctx == None:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # FIXME: this is a hack to make it work with DDP
        # rank = os.environ['LOCAL_RANK']
        # the following line requires multi-gpu in 
        rank = torch.cuda.current_device()
        total_rank = torch.cuda.device_count()
        bs = embedding.shape[0] // total_rank
        # embedding = embedding[rank*bs: min((rank+1)*bs, embedding.shape[0])]
        # tokenized_prompts = tokenized_prompts[rank*bs: min((rank+1)*bs, tokenized_prompts.shape[0])]

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx:, :]

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == 'none':
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, tokenized_prompts


class cfgc(object):
    backbonename = 'ViT-B/16'
    NCTX = 16
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'