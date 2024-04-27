import numpy as np
import torch
import torch.nn as nn

from config import cfg
from model.upernet.vit import vit_base
from dataset.ade20k import TestDataset
from model.upernet.upernet import uper
from model.upernet.segmentation import SegmentationModule

from PIL import Image
import os

cfg.merge_from_file("config/ade20k-resnet50-upernet.yaml")

device = 'cuda'
# Model
vit = vit_base(
    image_size=256,
    patch_size=16,
    num_classes=150,
)

upernet = uper(use_softmax=True)

crit = nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(vit, upernet, crit)
segmentation_module.load_state_dict(torch.load('../data/upernet_pretrain_epoch400.pth'))
segmentation_module.to(device)
segmentation_module.eval()
torch.set_grad_enabled(False)

def test(test_image_name):
  dataset_test = TestDataset([{'fpath_img': test_image_name}], cfg.DATASET, max_sample=-1)

  batch_data = dataset_test[0]
  segSize = (256, 256)
  img_resized_list = batch_data['img_data']
  
  scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1]).to(device)

  for img in img_resized_list:
    feed_dict = batch_data.copy()
    feed_dict['img_data'] = img
    del feed_dict['img_ori']
    del feed_dict['info']

  for k in feed_dict.keys():
    feed_dict[k] = feed_dict[k].to(device)
    # forward pass
    pred_tmp = segmentation_module(feed_dict["img_data"], None, segSize=segSize)
    scores = scores + pred_tmp

    _, pred = torch.max(scores, dim=1)
    return pred.squeeze(0).cpu().numpy()
  
def segm_transform(segm):
    segm = torch.from_numpy(np.array(segm)).long() - 1
    return segm

def mIoU(pred, gt):
    
    pred = torch.from_numpy(pred).to(device)
    gt = gt.to(device)
    iou_per_class = []
    
    for c in range(150):
        
        match_pred = pred == c
        match_gt   = gt == c

        if match_gt.long().sum().item() == 0: iou_per_class.append(np.nan)
            
        else:
            intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
            union = torch.logical_or(match_pred, match_gt).sum().float().item()

            iou = (intersect + 1e-10) / (union + 1e-10)
            iou_per_class.append(iou)
            
    return np.nanmean(iou_per_class)


acc = []

for i, file in enumerate(os.listdir('../data/ADEChallengeData2016/images/validation')):

    image_file = f'..\\data\\ADEChallengeData2016\\images\\validation\\{file}'
    pred= test(image_file)
    segm = Image.open(image_file.replace("images","annotations").replace("jpg","png"))
    segm = segm.resize((256, 256), Image.NEAREST)
    segm = segm_transform(segm)

    accuracy = mIoU(pred,segm)
    print(f'{i}: {accuracy}')
    acc.append(accuracy)

print(sum(acc)/len(acc))