import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50

import json
import re
from PIL import Image

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)


class iCore50(iData):

    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


    def __init__(self, args):
        self.args = args
        class_order = np.arange(8 * 50).tolist()
        self.class_order = class_order


    def download_data(self):
        datagen = CORE50(root=self.args["data_path"], scenario="ni")

        dataset_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i*50
            imglist = imglist.astype(np.uint8)
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list)[:, 0])
        train_y = np.concatenate(np.array(dataset_list)[:, 1])
        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y


class iDomainNet(iData):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


    def __init__(self, args):
        self.args = args
        class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

    def download_data(self):
        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.test_data = np.array(train_x)
        self.test_targets = np.array(train_y)



def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption

from torch.utils.data import Dataset
class CocoEval(Dataset):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, transform=transforms.Compose([*train_trsf, *common_trsf]),
                 image_root=None, ann_file=None, max_words=30 , tasks=[0]):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.text_cat = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.tasks = []
        task1 = [4, 5, 7, 8, 9]
        task2 = [2, 3, 10, 11, 12]
        task3 = [1, 6]

        # 12 setting
        # self.tasks.append([1])
        # self.tasks.append([2])
        # self.tasks.append([3])
        # self.tasks.append([4])
        # self.tasks.append([5])
        # self.tasks.append([6])
        # self.tasks.append([7])
        # self.tasks.append([8])
        # self.tasks.append([9])
        # self.tasks.append([10])
        # self.tasks.append([11])
        # self.tasks.append([12])
        self.tasks.append([11])

        self.tasks.append([6])
        self.tasks.append([3])
        self.tasks.append([10])
        self.tasks.append([5])
        self.tasks.append([12])

        self.tasks.append([7])
        self.tasks.append([9])
        self.tasks.append([2])

        self.tasks.append([8])
        self.tasks.append([4])
        self.tasks.append([1])
        # 3 setting
        # self.tasks.append(task1)
        # self.tasks.append(task2)
        # self.tasks.append(task3)


        txt_id = 0
        # self.ann = self.ann[:200]
        new_annotation = []
        for img_id, ann in enumerate(self.ann):
            category = ann['category']
            for task in tasks:
                if category in self.tasks[task]:
                    new_annotation.append(ann)
        self.ann = new_annotation


        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                new_category = 0
                for z in range(len(self.tasks)):
                    if ann['category'] in self.tasks[z]:
                        new_category = z
                self.text_cat.append(new_category)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1



    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        category = self.ann[index]['category']

        new_category = 0
        for z in range(len(self.tasks)):
            if category in self.tasks[z]:
                new_category = z

        return image, index, new_category

class Coco(Dataset):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, transform=transforms.Compose([*train_trsf, *common_trsf]),
                 image_root=None, ann_file=None,
                 max_words=30, prompt='', tasks=[0], replay_list=[]):
        # self.annotation = []
        # for f in ann_rpath:
        #     self.annotation += json.load(open(f, 'r'))
        self.annotation = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.tasks = []

        self.tasks.append([11])

        self.tasks.append([6])
        self.tasks.append([3])
        self.tasks.append([10])
        self.tasks.append([5])
        self.tasks.append([12])

        self.tasks.append([7])
        self.tasks.append([9])
        self.tasks.append([2])

        self.tasks.append([8])
        self.tasks.append([4])
        self.tasks.append([1])


        self.img_ids = {}
        n = 0
        new_annotation = []
        for ann in self.annotation:
            category = ann['category']
            for task in tasks:
                if category in self.tasks[task]:
                    img_id = ann['image_id']
                    if img_id not in self.img_ids.keys():
                        self.img_ids[img_id] = n
                        n += 1
                    new_annotation.append(ann)
        self.annotation = new_annotation
        if len(replay_list) != 0:
            self.annotation = self.annotation + replay_list

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        category = self.annotation[index]['category']

        new_category = 0
        for z in range(len(self.tasks)):
            if category in self.tasks[z]:
                new_category = z
                            # self.img_ids[ann['image_id']]
        return image, caption, 0, new_category


