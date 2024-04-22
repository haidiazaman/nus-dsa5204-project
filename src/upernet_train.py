import argparse
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.ade20k import TrainDataset
from config import cfg
from model.upernet.vit import vit_base
from model.upernet.upernet import uper
from model.upernet.segmentation import SegmentationModule

def get_args():
    parser = argparse.ArgumentParser(description='MAE Pretraining on ImageNet')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')

    # Model parameters
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--image_patch_size', type=int, default=16, help='image patch size')

    # Dataset parameters
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory for data loading')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')

    return parser.parse_args()


def main(args):

    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


    # Dataset
    cfg.merge_from_file("config/ade20k-resnet50-upernet.yaml")
    dataset_train = TrainDataset("../data/","../data/training.odgt", cfg.DATASET)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory)

    # Model
    vit = vit_base(
        image_size=args.image_size,
        patch_size=args.image_patch_size,
        num_classes=150,
    )
    
    state_dict = torch.load('../data/pretrain_mae_tinyimagenet_epoch600.pth')
    new_state_dict = {key.replace('encoder.', ''): value for key, value in state_dict.items() if key.split('.')[0] == 'encoder'}
    for key in ['to_patch_embedding.1.weight', 'to_patch_embedding.1.bias', 'to_patch_embedding.2.weight', 'mlp_head.weight', 'mlp_head.bias']:
        new_state_dict.pop(key)
    vit.load_state_dict(new_state_dict, strict=False)
    
    upernet = uper()

    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(vit, upernet, crit)

    segmentation_module.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(segmentation_module.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    print(segmentation_module.encoder)
    print(segmentation_module.decoder)
    # Training
    start_time = time.time()
    for epoch in range(args.epochs):
        segmentation_module.train()
        for batch_idx, img_dict in enumerate(train_loader):
            print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')
            for k in img_dict.keys():
                img_dict[k] = img_dict[k].to(device)
            optimizer.zero_grad()
            loss, acc = segmentation_module(img_dict["img_data"], img_dict["seg_label"])
            loss = loss.mean()
            acc = acc.mean()
            print(f'Loss: {loss}, Acc: {acc}')
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % 50 == 0:
            torch.save(segmentation_module.state_dict(), f'{args.output_dir}/upernet_pretrain_epoch{epoch + 1}.pth')
            end_time = time.time()
            print(f'Training time: {end_time - start_time} seconds')


if __name__ == '__main__':
    arg = get_args()
    main(arg)
