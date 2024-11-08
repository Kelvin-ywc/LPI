import torch
import torch.nn as nn
import copy

from models.clip.prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames
import numpy as np
import torch.nn.functional as F

from loss.loss import nt_bxent_loss, ClipLoss
from models.prompts.prompts import L2pPrompt, NormalPrompt, DecomposedPrompt
class SliNet(nn.Module):

    def __init__(self, args):
        super(SliNet, self).__init__()
        self.cfg = cfgc()
        self.args = args
        self.cfg.backbonename = args["backbonename"]
        self.cfg.NCTX = args["NCTX"]
        self.cfg.CTXINIT = args["CTXINIT"]
        self.cfg.CSC = args["CSC"]
        self.cfg.CLASS_TOKEN_POSITION = args["CLASS_TOKEN_POSITION"]

        clip_model = load_clip_to_cpu(args)
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.device = self.logit_scale.device
        if args['prompt_type'] == 'sprompts':

            self.prompts = nn.ModuleList([
                NormalPrompt(1,args["prompt_length"], args["visual_dim"], args["textual_dim"])
                 for i in range(args["total_sessions"])
            ])
            # self.textual_prompts = nn.ModuleList([
            #     NormalPrompt(1,args["prompt_length"], args["textual_dim"]) for i in range(args["total_sessions"])
            # ])
        elif args['prompt_type'] == 'l2p':
            self.prompts = L2pPrompt(length=4, embed_dim=96, prompt_pool=True, pool_size=12, top_k=4,
                                     batchwise_prompt=True, prompt_key=True)
        elif args['prompt_type'] == 'lpi':
            self.prompts = nn.ModuleList([
                DecomposedPrompt(9,args["prompt_length"], args["visual_dim"], args["textual_dim"]) for i in range(args["total_sessions"])
            ])

        self.class_num = 1
        if True or args["dataset"] == "cddb":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(domainnet_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 50
        # else:
        #     raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        # self.prompt_pool = nn.ModuleList([
        #     nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
        #     for i in range(args["total_sessions"])
        # ])

        self.numtask = 0 #0
        from methods.sprompt import ClipLoss
        self.loss = ClipLoss()
        self.alignment_loss = ClipLoss()
        self.all_keys = []
    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def extract_visual_vector(self, image):
        if self.args['prompt_type'] == 'l2p':
            image_features = self.image_encoder(image.type(self.dtype), self.prompts)
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def extract_vector(self, image):
        if self.args['prompt_type'] == 'l2p':
            image_features = self.image_encoder(image.type(self.dtype), self.prompts)
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def extract_textual_vector(self, text):
        text_prompts, tokenized_prompts = self.classifier_pool[self.numtask-1].extract_vector(text)
        textual_features = self.text_encoder(text_prompts, tokenized_prompts, None)
        textual_features = textual_features / textual_features.norm(dim=-1, keepdim=True)
        return textual_features

    def forward(self, image, text):


        if self.args['prompt_type'] == 'l2p':
            visual_prompt_exp = self.prompts
            textual_prompt = None
        else:
            visual_prompt, textual_prompt = self.prompts[self.numtask - 1]()
            visual_prompt = visual_prompt.type(self.dtype)
            bs = image.shape[0]
            visual_prompt_exp = visual_prompt.expand(bs, -1, -1, -1)

        image_features = self.image_encoder(image.type(self.dtype), visual_prompt_exp)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.classifier_pool[self.numtask-1]
        if self.args['prompt_type'] == 'l2p':
            text_prompts, tokenized_prompts = prompts(text, None)
        else:
            textual_prompt = textual_prompt.type(self.dtype)
            textual_prompt_exp = textual_prompt.expand(bs, -1,-1,-1)
            text_prompts, tokenized_prompts = prompts(text, textual_prompt_exp[:,0])

        text_features = self.text_encoder(text_prompts, tokenized_prompts, textual_prompt_exp)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, visual_prompt_exp, textual_prompt_exp
    
    def cal_loss(self, image_featuers, text_features, visual_prompt, textual_prompt):
        logit_scale = self.logit_scale.exp()
        item_score = logit_scale * image_featuers @ text_features.t()
        losses = {}
        loss = self.loss(logits=item_score)
        base_loss = {'base_loss': loss}
        losses.update(base_loss)
        if self.args['prompt_type'] == 'lpi':
            temperature = 0.01
            visual_prompt_for_loss = torch.mean(visual_prompt, -1)
            if len(visual_prompt_for_loss.shape) == 3:
                visual_prompt_for_loss = torch.mean(visual_prompt_for_loss, 0)
            # visual_prompt_for_loss_norm = visual_prompt_for_loss / visual_prompt_for_loss.norm(dim=-1, keepdim=True)
            visual_prompt_for_loss = visual_prompt_for_loss / temperature
            textual_prompt_for_loss = torch.mean(textual_prompt, -1)
            if len(textual_prompt_for_loss.shape) == 3:
                textual_prompt_for_loss = torch.mean(textual_prompt_for_loss, 0)
            textual_prompt_for_loss = textual_prompt_for_loss / temperature
            # textual_prompt_for_loss_norm = textual_prompt_for_loss / textual_prompt_for_loss.norm(dim=-1, keepdim=True)
            # layer_score = visual_prompt_for_loss_norm @ textual_prompt_for_loss_norm.t()
            layer_score = visual_prompt_for_loss @ textual_prompt_for_loss.t()
            alignment_loss = {'alignment_loss': 0.1*self.alignment_loss(layer_score)}
            losses.update(alignment_loss)
        if self.args['prompt_type'] == 'lpi' and self.numtask!=1:
            task_loss = {'task_loss': 0.1*self.cal_task_loss(self.numtask-1, visual_prompt_for_loss, textual_prompt_for_loss)}
            losses.update(task_loss)
        return {
            "loss": losses
        }

    def cal_task_loss(self, task_id, visual_prompt, textual_prompt): # [12,16], [12,16]
        task_id = task_id
        device = visual_prompt.device
        # import numpy as np
        task_sim_matrix = np.loadtxt('./MID/task_sim_matrix.txt')
        task_sim_matrix = torch.tensor(task_sim_matrix[:task_id+1, :task_id+1]).to(device)
        threshold = 0.4
        task_sim_matrix = (task_sim_matrix>threshold).type(torch.int)
        if False:
            visual_prompt_stack = torch.stack([torch.mean(self.prompts[i]()[0], dim=-1).view(-1) for i in range(task_id+1)])
            textual_prompt_stack = torch.stack([torch.mean(self.prompts[i]()[1], dim=-1).view(-1) for i in range(task_id+1)])
        else:
            visual_prompt_stack = torch.stack([self.prompts[i]()[0].view(-1) for i in range(task_id+1)])
            textual_prompt_stack = torch.stack([self.prompts[i]()[1].view(-1) for i in range(task_id+1)])

        temperature = 0.001
        return (nt_bxent_loss(visual_prompt_stack, task_sim_matrix, temperature) + nt_bxent_loss(textual_prompt_stack, task_sim_matrix, temperature)) / 2

    def textual_interface(self, text, text_category):
        if self.training:
            prompts = self.classifier_pool[self.numtask-1]
            # tokenized_prompts = prompts.tokenized_prompts
            a, b = prompts(text)
        else:

            a = []
            b = []
            c = []
            for bid in range(len(text)):
                if self.args['prompt_type'] == 'lpi':
                    textual_prompt = self.prompts[text_category[bid].item()]()[1]
                else:
                    textual_prompt = self.textual_prompts[text_category[bid].item()]()
                textual_prompt = textual_prompt.type(self.dtype)
                a_, b_ = self.classifier_pool[text_category[bid]]([text[bid]], textual_prompt[0])
                a.append(a_)
                b.append(b_)
                c.append(textual_prompt)
            a = torch.stack(a, dim=0).squeeze(1)
            b = torch.stack(b, dim=0).squeeze(1)
            c = torch.stack(c,dim=0).squeeze(1)
        text_features = self.text_encoder(a, b,c)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def visual_interface(self, image, image_category):

        if self.args['prompt_type'] == 'lpi':
            instance_batch = torch.stack([i()[0] for i in self.prompts], 0)[image_category, :, :]
        else:
            instance_batch = torch.stack([i() for i in self.visual_prompts], 0)[image_category, :, :]
        image_features = self.image_encoder(image.type(self.dtype), instance_batch.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


    def update_fc(self, nb_classes):
        self.numtask += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self



def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


