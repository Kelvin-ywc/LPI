import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
# from utils.tsne import tsne

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy_domain
from models.sinet import SiNet
from models.slinet import SliNet

from loss.loss import ClipLoss
import torch.nn as nn

from utils.data import Coco, CocoEval

from loguru import logger
from datetime import datetime
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class SPrompts(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "slip":
            self._network = SliNet(args)
        elif args["net_type"] == "sip":
            self._network = SiNet(args)
        # elif args["net_type"] == "slip_cmpa":
        #     self._network = SliNet_CMPA(args)
        # elif args["net_type"] == "slip_ampl":
        #     self._network = SliNet_AMPL(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))

        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]


        self.topk = 2  # origin is 5
        self.class_num = self._network.class_num

        self.all_keys = []
        self.textual_all_keys = []
        self.loss = ClipLoss()

        self.cur_id = 0
        log_fn = f'./logs/{datetime.now()}.txt'
        logger.add(log_fn)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self):
        self._acc_task = []
        replay_list = []
        final_res = {}
        for i, id in enumerate(np.arange(12)):
            self._cur_task = []
            self._cur_task.append(id)
            self._acc_task.append(id)
            self.cur_id = i
            # self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
            self._network.update_fc(self._total_classes)


            train_dataset = Coco(image_root=self.args['image_root'], ann_file=self.args['annotation_train_root'], tasks=self._cur_task)


            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

            test_dataset = CocoEval(image_root=self.args['image_root'], ann_file=self.args['annotation_val_root'], tasks=np.arange(0, self.cur_id+1))
            self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                                          num_workers=self.num_workers)

            # if i == 0:
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)

            tmp_res = self._train(self.train_loader, self.test_loader)
            # self.clustering(self.train_loader)
            final_res[i] = tmp_res
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

        from datetime import datetime
        file_path = f'./res/{datetime.now()}.json'
        logger.info(final_res)
        self.save_dict(final_res, file_path)




    def save_dict(self, dictionary, file_path):
        import json
        with open(file_path, 'w') as file:
            json.dump(dictionary, file)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if isinstance(self._network, torch.nn.DataParallel):
            network = self._network.module 
        else:
            network = self._network
        if len(self.args['device']) > 1:
            for name, param in network.named_parameters():
                param.requires_grad_(False)
                if "classifier_pool" + "." + str(network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                    logger.info(f'{name}: not frozen')
                if "prompt_pool" + "." + str(network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                    logger.info(f'{name}: not frozen')
                # if "prompt_attn" in name and self._network.numtask-1==0:
                #     param.requires_grad_(True)
                if "prompt_attn" + "."+str(network.numtask - 1)+"." in name:
                    param.requires_grad_(True)
                    logger.info(f'{name}: not frozen')
                if "visual_prompts" + "." + str(network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                    logger.info(f'{name}: not frozen')
                if "textual_prompts" + "."+str(network.numtask - 1)+"." in name:
                    param.requires_grad_(True)
                    logger.info(f'{name}: not frozen')
                if "prompts" + "." + str(network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                    logger.info(f'{name}: not frozen')
        else:
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                # if "classifier_pool" + "." + str(self._network.numtask - 1) + "." in name:
                #     param.requires_grad_(True)
                #     logger.info(f'{name}: not frozen')
                if "prompts" + "." + str(network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                    logger.info(f'{name}: not frozen')
                # print(name)
            # exit(0)
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if self._cur_task==0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            self.run_epoch = self.init_epoch
            return self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            self.run_epoch = self.epochs
            return self.train_function(train_loader, test_loader, optimizer, scheduler)


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

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        # self._evaluate_retrieval(test_loader, 0)


        prog_bar = tqdm(range(self.run_epoch))
        import collections
        loss_meter = collections.defaultdict(lambda: AverageMeter())
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            for i,  (images, captions, _, _) in enumerate(train_loader):
                images = images.cuda()
                captions = list(captions)
                model_out = self._network(images, captions)
                # loss = model_out['loss'].mean()
                loss = sum(loss for loss in model_out['loss'].values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                # loss_meter['loss'].update(loss.item())
                for val in model_out['loss']:
                    loss_meter[val].update(model_out['loss'][val])
                if i % 50 == 0:
                    info = 'Task {}, Epoch {}/{}, Batch {}, lr {:.4f} =>, '.format(
                        self.cur_id, epoch + 1, self.run_epoch, i, optimizer.param_groups[0]["lr"])
                    for k, v in loss_meter.items():
                        info += '{} = {:.4f}, '.format(k, v.avg)
                        v.reset()
                    logging.info(info)
            scheduler.step()

            # print("eval")
            # self._evaluate_retrieval(test_loader)
        self.clustering(dataloader=train_loader)
        # for i in range(len(test_loader)):
        #     tmp_test_loader = test_loader[i]
        # # for i, tmp_test_loader in test_loader:
        #     print(f'Evaluating at Task [{i}]')
        _,_, final_res = self._evaluate_retrieval(test_loader)
        return final_res

    def get_visual_task_id(self, inputs):
        with torch.no_grad():
            if isinstance(self._network, nn.DataParallel):
                feature = self._network.module.extract_vector(inputs)
            else:
                feature = self._network.extract_vector(inputs)

            taskselection = []
            for task_centers in self.all_keys:
                tmpcentersbatch = []
                for center in task_centers:
                    tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])

            selection = torch.vstack(taskselection).min(0)[1]
        return selection

    def get_textual_task_id(self, inputs):
        with torch.no_grad():
            if isinstance(self._network, nn.DataParallel):
                feature = self._network.module.extract_textual_vector(inputs)
            else:
                feature = self._network.extract_textual_vector(inputs)

            taskselection = []
            for task_centers in self.textual_all_keys:
                tmpcentersbatch = []
                for center in task_centers:
                    tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])

            selection = torch.vstack(taskselection).min(0)[1]
        return selection

    def clustering(self, dataloader):
        visual_features = []
        textual_features = []
        for i,  (inputs, captions, _, _)  in enumerate(dataloader):
            inputs = inputs.to(self._device)
            # inputs, targets = inputs.to(self._device), targets.to(self._device)
            # mask = (targets >= self._known_classes).nonzero().view(-1)
            # inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    visual_feature = self._network.module.extract_vector(inputs)
                    textual_feature = self._network.module.extract_textual_vector(captions)
                    textual_feature = torch.mean(textual_feature, dim=1)
                else:
                    visual_feature = self._network.extract_vector(inputs)
                    textual_feature = self._network.extract_textual_vector(captions)

            visual_feature = visual_feature / visual_feature.norm(dim=-1, keepdim=True)
            visual_features.append(visual_feature)
            textual_feature = textual_feature / textual_feature.norm(dim=-1, keepdim=True)
            textual_features.append(textual_feature)
        visual_features = torch.cat(visual_features, 0).cpu().detach().numpy()
        textual_features = torch.cat(textual_features, 0).cpu().detach().numpy()
        visual_clustering = KMeans(n_clusters=5, random_state=0).fit(visual_features)
        textual_clustering = KMeans(n_clusters=5, random_state=0).fit(textual_features)

        self.all_keys.append(torch.tensor(visual_clustering.cluster_centers_).to(visual_feature.device))
        self.textual_all_keys.append(torch.tensor(textual_clustering.cluster_centers_).to(textual_feature.device))
        # print(self.all_keys)
        # print(len(self.all_keys))
        # print(self.textual_all_keys)
        # print(len(self.textual_all_keys))
        # print(self.all_keys.shape)
        # print(self.all_keys)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain(y_pred.T[0], y_true, self._known_classes, class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def get_clip_metrics(self, image_features, text_features, logit_scale):
        metrics = {}
        image_features = image_features[:5000]
        text_features = text_features[:5000]
        logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
        logits_per_text = logits_per_image.t().detach().cpu()

        logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)

        return metrics

    @torch.no_grad()
    def _evaluate_retrieval(self, data_loader):
        self._network.eval()
        bs = data_loader.batch_size
        # print('Computing features for evaluation...')
        import time
        start_time = time.time()

        texts = data_loader.dataset.text
        score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).cuda()
        score_matrix_i2t = torch.full(( len(data_loader.dataset.image), len(texts)), -100.0).cuda()
        image_feats = []
        image_embeds = []
        category_i = []

        texts_cat = data_loader.dataset.text_cat
        texts_cat = torch.tensor(texts_cat)
        num_text = len(texts)
        num_img = len(data_loader.dataset.image)
        text_bs = 256  # 256
        idx = -1
        image_feats = torch.tensor([]).to(self._device)
        text_feats = torch.tensor([]).to(self._device)
        for image, img_id, category in data_loader:
            image = image.cuda()
            category = category.cuda()
            idx = idx+1
            if self.args['prompt_type'] == 'clip':
                image_feat = self._network.extract_vector(image)
            elif self.args['prompt_type'] == 'l2p':
                # image_feat = self.prompts(image)['prompted_embedding']
                image_feat = self.extract_vector(image)
            else:
                selection = self.get_visual_task_id(image)
                image_feat = self._network.visual_interface(image, selection)
            # print(f'task acc: {torch.sum(selection==category)/selection.shape[0]}')
            image_feats = torch.cat([image_feats, image_feat])
            # for cnt in range(image.shape[0]):
            for z in category:
                category_i.append(z)


        for i in range(0, num_text, text_bs):
            cur_len = i%text_bs+1
            text = texts[i: min(num_text, i + text_bs)]
            text_cat = texts_cat[i: min(num_text, i + text_bs)]
            # img = image[cnt,:,:].expand(cur_len,-1,-1,-1)
            # cat = category[cnt].expand(cur_len)
            if self.args['prompt_type'] == 'clip':
                text_feat = self._network.extract_textual_vector(text)
            elif self.args['prompt_type'] == 'l2p':
                text_feat = self.extract_textual_vector(text)
            else:
                textual_selection = self.get_textual_task_id(text)
                if len(self._multiple_gpus) > 1:
                    image_feat, text_feat = self._network.module.interface_visual_text(image, category,text, text_cat)
                else:
                    # image_feat, text_feat = self._network.interface_visual_text(image, category, text, text_cat)
                    # image_feat, text_feat = self._network.interface_visual_text(image, selection, text, textual_selection)
                    # print(f'task acc: {torch.sum(textual_selection == text_cat.to(textual_selection.device)) / textual_selection.shape[0]}')
                    print(f'task acc: {torch.sum(textual_selection == text_cat.to(textual_selection.device)) / textual_selection.shape[0]}')
                    text_feat = self._network.textual_interface(text, textual_selection)
            text_feats = torch.cat([text_feats, text_feat])
                # print(category)

                # print(selection)
                # print(text_cat)
                # print(textual_selection)
                # print(category)
                # print(text_cat)
                # print(selection)
                # exit(0)
                # score = torch.sum(image_feat * text_feat, dim=1)

            # score =(image_feat@text_feat.t()).t()
            # score_matrix_t2i[i:min(num_text, i+text_bs), idx*bs:min((idx+1)*bs, num_img)] = score
        score_matrix_t2i = (image_feats@text_feats.t()).t()


        # image_feats = torch.cat(image_feats, dim=0)



        # text_feats = []
        # text_embeds = []
        # text_masks = []
        # for i in range(0, num_text, text_bs):
        #     text = texts[i: min(num_text, i + text_bs)]
        #     text_cat = texts_cat[i: min(num_text, i + text_bs)]
        #     if len(self._multiple_gpus) > 1:
        #         text, text_mask = self._network.module.extract_text_vector(text, text_cat)
        #     else:
        #         text, text_mask = self._network.extract_text_vector(text, text_cat)
        #     text_embeds.append(text)
        #     text_feats.append(text)
        #     text_masks.append(text_mask)
        #
        #     # encoder_output = image_feats
        #     # score = encoder_output @ text.t()
        #     # score = score.transpose(0, 1)
        #
        #     score_matrix_t2i[i: min(num_text, i + text_bs), :] = score.type(torch.float32)


        score_matrix_i2t = score_matrix_t2i.transpose(0, 1)

        total_time = time.time() - start_time
        import datetime
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Evaluation time {}'.format(total_time_str))

        score_matrix_i2t, score_matrix_t2i = score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

        final_res = self.itm_eval(score_matrix_i2t, score_matrix_t2i, data_loader.dataset.txt2img, data_loader.dataset.img2txt, category_i, texts_cat)

        return score_matrix_i2t, score_matrix_t2i, final_res

    @torch.no_grad()
    def itm_eval(self,scores_i2t, scores_t2i, txt2img, img2txt, category_i, category_t):
        # Images->Text
        final_res = {}

        i2t_res = {}
        t2i_res = {}

        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
        logger.info(f'For Task[{self.cur_id}]: score shape: {scores_t2i.shape}')
        # Compute metrics
        tr1 = 0.0
        tr5 = 0.0
        tr10 = 0.0
        task_num = self.cur_id+1
        # task_num = 1
        category_i = torch.tensor(category_i)
        category_t = torch.from_numpy(np.asarray(category_t))
        for task in range(task_num):
            ranks_ = torch.masked_select(torch.from_numpy(ranks), category_i == task)
            ranks_ = ranks_.numpy()
            # print(ranks_)
            tr1_ = 100.0 * len(np.where(ranks_ < 1)[0]) / len(ranks_+ 1e-10)
            tr5_ = 100.0 * len(np.where(ranks_ < 5)[0]) / len(ranks_+ 1e-10)
            tr10_ = 100.0 * len(np.where(ranks_ < 10)[0]) / len(ranks_+ 1e-10)
            logger.info(f'For Task[{task_num-1}] Image-Text -> ranks shape: {len(ranks_)}')
            logger.info(f'For Task[{task_num-1}] Image-Text -> Evaluating at task [{task}], tr1: {tr1_}, tr5: {tr5_}, tr10: {tr10_}')
            tr1 = tr1 + tr1_
            tr5 = tr5 + tr5_
            tr10 = tr10 + tr10_
            i2t_res[task] = [tr1_, tr5_, tr10_]
        tr1 = tr1 / task_num
        tr5 = tr5 / task_num
        tr10 = tr10 / task_num

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 0.0
        ir5 = 0.0
        ir10 = 0.0
        for task in range(task_num):
            ranks_ = torch.masked_select(torch.from_numpy(ranks), category_t == task)
            ranks_ = ranks_.numpy()
            ir1_ = 100.0 * len(np.where(ranks_ < 1)[0]) / len(ranks_)
            ir5_ = 100.0 * len(np.where(ranks_ < 5)[0]) / len(ranks_)
            ir10_ = 100.0 * len(np.where(ranks_ < 10)[0]) / len(ranks_)
            ir1 = ir1 + ir1_
            ir5 = ir5 + ir5_
            ir10 = ir10 + ir10_
            logger.info(f'For Task[{task_num - 1}] Text-Image -> ranks shape: {len(ranks_)}')
            logger.info(f'For Task[{task_num-1}] Text-Image -> Evaluating at task [{task}], tr1: {ir1_}, tr5: {ir5_}, tr10: {ir10_}')
            t2i_res[task] = [ir1_, ir5_, ir10_]
        ir1 = ir1 / task_num
        ir5 = ir5 / task_num
        ir10 = ir10 / task_num

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        eval_result = {'txt_r1': tr1,
                       'txt_r5': tr5,
                       'txt_r10': tr10,
                       'txt_r_mean': tr_mean,
                       'img_r1': ir1,
                       'img_r5': ir5,
                       'img_r10': ir10,
                       'img_r_mean': ir_mean,
                       'r_mean': r_mean}

        info = ('Task[{}]: txt_r1 {:.4f}, txt_r5 {:.4f}, txt_r10 {:.4f}, txt_r_mean {:.4f}, '
                'img_r1 {:.4f}, img_r5 {:.4f}, img_r10 {:.4f}, img_r_mean {:.4f}, r_mean {:.4f} ').format(
            task_num-1, tr1, tr5, tr10, tr_mean, ir1, ir5, ir10, ir_mean, r_mean)
        final_res['mscoco'] = {
            'i2t': i2t_res,
            't2i': t2i_res
        }
        logger.info(final_res)
        logging.info(info)
        # for k, v in eval_result.items():
        #     print(k,v)
        return final_res


    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)

                taskselection = []
                for task_centers in self.all_keys:
                    tmpcentersbatch = []
                    for center in task_centers:
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])

                selection = torch.vstack(taskselection).min(0)[1]

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, selection)
                else:
                    outputs = self._network.interface(inputs, selection)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
