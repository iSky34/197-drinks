# -*- coding: utf-8 -*-
import requests
import zipfile
import os

#!pip install -U torchvision

# Commented out IPython magic to ensure Python compatibility.
#!pip install albumentations==0.4.6
import torch
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import cv2
import albumentations as A  # our data augmentation library

import warnings
warnings.filterwarnings("ignore")
from torchvision.utils import draw_bounding_boxes

#!pip install pycocotools
from pycocotools.coco import COCO
from engine import evaluate
from albumentations.pytorch import ToTensorV2

if not os.path.exists("197/drinkscoco/train/_annotations.coco.json"):
    print(f'downloading dataset')
    url='https://github.com/iSky34/197-drinks/releases/download/Dataset/197-20220502T103206Z-001.zip'
    a=requests.get(url)
    open('197-20220502T103206Z-001.zip','wb').write(a.content)
    with zipfile.ZipFile('197-20220502T103206Z-001.zip','r') as a:
        a.extractall()

if not os.path.exists("weights.zip"):
    print(f'downloading weights')
    url='https://github.com/iSky34/197-drinks/releases/download/weights/weights.zip'
    a=requests.get(url)
    open('weights.zip','wb').write(a.content)
    with zipfile.ZipFile('weights.zip','r') as a:
        a.extractall()

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            #A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

class DrinksDetection(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split #train, valid, test
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json")) # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target] # required annotation format for albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        
        image = transformed['image']
        boxes = transformed['bboxes']
        
        new_boxes = [] # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {} # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        #targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        #targ['image_id'] = torch.tensor([index])
        #print([index])
        #print([t['image_id'] for t in target])
        targ['image_id'] = torch.tensor([index])
        
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ # scale images
    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':

    dataset_path = "197/drinkscoco"

    coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
    categories = coco.cats
    n_classes = len(categories.keys())
    categories

    train_dataset = DrinksDetection(root=dataset_path, transforms=get_transforms(True))

    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    device = torch.device("cuda") # use GPU to train

    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1) # lr scheduler

    test_dataset = DrinksDetection(root=dataset_path, split="test", transforms=get_transforms(False))


    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
    model.load_state_dict(torch.load("dd.pth"))
    model.to(device)

    evaluate(model, data_loader_test,device=device)
