import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A


from engine import train_one_epoch, evaluate, validate
import utils
import torchvision.transforms as tra
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import cv2
#import generate_masks as gm
import random
from pycocotools.coco import COCO




def get_box_transform(train):
    if train:
        transforms = A.Compose([
        #A.Resize(200, 300),
        #A.CenterCrop(100, 100),
        #A.RandomCrop(80, 80),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-6, 6), p=0.6),
        #A.ShiftScaleRotate(rotate_limit=50, p=0.6)
        #A.SmallestMaxSize(max_size=1292, p=1)
        #A.augmentations.transforms.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=5, p=0.5)
        A.RandomScale(scale_limit=0.2, p=0.5)
        #A.core.composition.OneOf([A.Resize(2298, 1292, interpolation=1, always_apply=False, p=1), A.Resize(3109, 1748, interpolation=1, always_apply=False, p=1), A.Resize(2568, 1444, interpolation=1, always_apply=False, p=1), A.Resize(2839, 1596, interpolation=1, always_apply=False, p=1)],p=0.5)
        #A.augmentations.transforms.Resize(2298, 1292, interpolation=1, always_apply=False, p=1)
        #A.VerticalFlip(p=0.5),
        #A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        transforms=None
    return transforms

def get_mask_transform(train):
    if train:
        transforms = A.Compose([
        #A.Resize(200, 300),
        #A.CenterCrop(100, 100),
        #A.RandomCrop(80, 80),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-6, 6), p=0.6),
        #A.ShiftScaleRotate(rotate_limit=50, p=0.6)
        #A.SmallestMaxSize(max_size=1292, p=1)
        #A.augmentations.transforms.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=5, p=0.5)
        A.RandomScale(scale_limit=0.2, p=0.5)
        #A.core.composition.OneOf([A.Resize(2298, 1292, interpolation=1, always_apply=False, p=1), A.Resize(3109, 1748, interpolation=1, always_apply=False, p=1), A.Resize(2568, 1444, interpolation=1, always_apply=False, p=1), A.Resize(2839, 1596, interpolation=1, always_apply=False, p=1)],p=0.5)
        #A.augmentations.transforms.Resize(2298, 1292, interpolation=1, always_apply=False, p=1)
        #A.VerticalFlip(p=0.5),
        #A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transforms=None
    return transforms

class COCOInstanceSegmentationDataset(object):
    def __init__(self, root, json_dir, transforms, train):
        self.root = root
        self.json=json_dir
        self.train=train
        self.transforms = transforms
        self.coco = COCO(json_dir)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        # load images and masks

        id = self.ids[idx]

        path = self.coco.loadImgs(id)[0]["file_name"]
        img=Image.open(os.path.join(self.root, path)).convert("RGB")
        coco_dict=self.coco.loadAnns(self.coco.getAnnIds(id))
        width, height = img.size
        img=np.array(img, dtype=np.float32)/255
        # masks=np.zeros((0, height, width), dtype=np.uint8)
        masks=[]
        

        for instance in coco_dict: 
            z=np.zeros((height, width), dtype=np.uint8)   
            points=[instance['segmentation'][0][i * 2:(i + 1) * 2] for i in range((len(instance['segmentation'][0]) + 2 - 1) // 2 )]
            mask=cv2.fillPoly(z, pts = [np.array(points)], color =(1))
            # masks=np.concatenate((masks, np.expand_dims(mask, axis=0)))
            # print(np.sum(mask))
            if np.sum(mask)>=100:
                masks.append(mask)
   
        

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd

        target = {}
        target["image_id"] = image_id
       

        if self.train:
            #data={'image': img, 'masks': masks}
            augmented = self.transforms(image=img, masks=masks)
            img=augmented["image"]
            #print(img.shape)
            masks=augmented["masks"]


        del_list=[]
        for i in range(len(masks)):
            if np.sum(masks[i])==0:
                del_list.append(i)
        many_deleted=0
        if len(del_list) != 0:
            for ind in del_list:
                masks.pop(ind-many_deleted)
                many_deleted=many_deleted+1
        # print("Time for line 168 to 176 in datatools")        
        # print(time.time()-tic)            
        
        
        img=np.moveaxis(img, (0,1,2), (1,2,0)) #pytorch wants a different format for the image ([C, H, W]) so run img=np.moveaxis(img, (0,1,2), (1,2,0)) on the augmented image before turning it into a float tensor
        img=torch.as_tensor(img, dtype=torch.float32)
        #print(img.shape)
        #print(type(masks))
        num_objs=len(masks)
        #print(img.shape)
        #print(target["masks"].shape)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        target["labels"] = labels
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target["iscrowd"] = iscrowd

        #print(target["masks"].shape)
        boxes = []
        # print('np.sum is')
        # print(np.sum(masks)) 
        
        # tic=time.time()
        amount_deleted=0
        for i in range(num_objs):
            pos = np.where(masks[i-amount_deleted])
            #print(np.sum(masks[i]))
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if (xmax - xmin > 0 ) and (ymax - ymin > 0):
                #print([xmin, ymin, xmax, ymax])
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                masks=np.delete(masks, i-amount_deleted, axis=0)
                amount_deleted+=1
                print('empty box')
                # print(img_path)
        # print("Time for line 201 to 214 in datatools")        
        # print(time.time()-tic)            


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        if boxes.shape == torch.Size([0]):
            target['masks'] = torch.zeros(0,img.size()[1],img.size()[2], dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        target["boxes"] = boxes

        return img, target

    def __len__(self):
        return len(self.ids)

class COCOObjectDetectionDataset(object):
    def __init__(self, root, json_dir, transforms, train):
        self.root = root
        self.json=json_dir
        self.transforms = transforms
        self.train=train
        # load all image files, sorting them to
        # ensure that they are aligned
  
        self.coco = COCO(json_dir)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        # load images ad masks
        id = self.ids[idx]

        path = self.coco.loadImgs(id)[0]["file_name"]
        img=Image.open(os.path.join(self.root, path)).convert("RGB")
        coco_dict=self.coco.loadAnns(self.coco.getAnnIds(id))
        width, height = img.size
        img=np.array(img, dtype=np.float32)/255
        # masks=np.zeros((0, height, width), dtype=np.uint8)
 
        boxes=[]
        for instance in coco_dict: 
            boxes.append([instance['bbox'][0],instance['bbox'][1],instance['bbox'][0]+instance['bbox'][2], instance['bbox'][1]+ instance['bbox'][3]])



        image_id = torch.tensor([idx])
        # suppose all instances are not crowd

        target = {}
        target["image_id"] = image_id
        #make sure your img and mask array are in this format before passing into albumentations transforms, img.shape=[H, W, C] and mask.shape= [N, H, W]
  
        num_objs=len(boxes)
     
        labels = torch.ones((num_objs,), dtype=torch.int64)

        if self.train:
            #data={'image': img, 'masks': masks}
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img=augmented["image"]
            #print(img.shape)
            boxes=augmented["bboxes"]
        img=np.moveaxis(img, (0,1,2), (1,2,0)) #pytorch wants a different format for the image ([C, H, W]) so run img=np.moveaxis(img, (0,1,2), (1,2,0)) on the augmented image before turning it into a float tensor
        img=torch.as_tensor(img, dtype=torch.float32)
        
        num_objs=len(boxes)
      
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # target["labels"] = labels
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target["iscrowd"] = iscrowd
        #print(target["masks"].shape)


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.shape == torch.Size([0]):
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        target["boxes"] = boxes
  
        

        return img, target

    def __len__(self):
        return len(self.ids)

