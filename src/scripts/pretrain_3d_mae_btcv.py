import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


import argparse
import numpy as np
import torch
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)

from monai.data import DataLoader
from tqdm import tqdm
from src.dataset.btcv import get_btcv_dataset
from src.model.vit_3d import vit_3d_base
from src.model.mae import MAE


def get_args():
    parser = argparse.ArgumentParser(description='MAE 3D Pretraining on BTCV')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pretrain_mae_btcv_epoch10.pth')

    # Model parameters
    parser.add_argument('--image_size', type=int, default=96, help='image size')
    parser.add_argument('--image_patch_size', type=int, default=16, help='image patch size')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/btcv', help='path to ImageNet data')
    parser.add_argument('--data_json_name', type=str, default='dataset_0.json', help='filename of btcv data json')
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

    # Augmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )

    # Dataset
    dataset_train = get_btcv_dataset(args.data_path,
                                     args.data_json_name,
                                     subset="training",
                                     transform=train_transforms,
                                     )
    train_loader = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              )

    # Model
    vit_3d = vit_3d_base(
        image_size=args.image_size,
        image_patch_size=args.image_patch_size,
        frames=args.image_size,
        frame_patch_size=args.image_patch_size,
        num_classes=14,
        channels=1,
    )
    mae = MAE(encoder=vit_3d)
    mae.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(mae.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Load half trained model if needed
    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        mae.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        lr_scheduler.load_state_dict(checkpoint["lr"])
        curr_epoch = checkpoint['epoch']
        epoch_losses = checkpoint["loss"]
        print(f"Resuming from epoch: {curr_epoch}.")
    else:
        curr_epoch = 0
        epoch_losses = list()

    # Training
    epoch_iterator = tqdm(range(args.epochs))
    for epoch in epoch_iterator:
        epoch_iterator.set_description(
            f"Epoch: {epoch}, Loss: {epoch_losses[-1] if epoch_losses else '?'}")
        if epoch < curr_epoch:
            continue
        mae.train()
        epoch_loss = 0
        for batch in train_loader:
            img = batch["image"]
            img = img.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = mae(img)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        lr_scheduler.step()
        epoch_loss /= (len(train_loader) * args.batch_size)
        epoch_losses.append(epoch_loss)
        if (epoch > 0 and epoch % 50 == 0) or epoch == args.epochs - 1:
            state = {"model": mae.state_dict(),
                     "opt": optimizer.state_dict(),
                     "lr": lr_scheduler.state_dict(),
                     "loss": epoch_losses,
                     "epoch": epoch,
                     }
            torch.save(state, f'{args.output_dir}/pretrain_mae_btcv_epoch{epoch}.pth')


if __name__ == '__main__':
    arg = get_args()
    main(arg)
