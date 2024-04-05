import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


import argparse
import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
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

from monai.metrics import DiceMetric

from monai.data import DataLoader, decollate_batch
from tqdm import tqdm
from src.dataset.btcv import get_btcv_dataset
from src.model.vit_3d import vit_3d_base
from src.model.mae import MAE
from src.model.unetr import UNETR


def get_args():
    parser = argparse.ArgumentParser(description='MAE 3D Pretraining on BTCV')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--eval_num', type=int, default=50)
    parser.add_argument('--load_weights', type=bool, default=False)
    parser.add_argument('--pretrained_mae_path', type=str, default='checkpoints/pretrain_mae_btcv_epoch199.pth')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/unetr_btcv_best.pth')

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
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')

    return parser.parse_args()


def validation(model, val_loader, dice_metric, post_label, post_pred):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(model,
          optimizer,
          loss_function,
          epochs,
          train_loader,
          val_loader,
          eval_num,
          epoch_losses,
          dice_scores,
          output_dir,
          dice_metric,
          post_label,
          post_pred,
          load_weights,
          curr_epoch=0,
          ):
    model.train()
    epoch_loss = 0
    tot_data_count = len(train_loader)
    epoch_iterator = tqdm(range(epochs))
    for epoch in epoch_iterator:
        if epoch < curr_epoch:
            continue
        epoch_iterator.set_description(f"Epoch: {epoch}, Loss: {epoch_losses[-1] if epoch_losses else '?'}")
        for batch in train_loader:
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        epoch_losses.append(epoch_loss / tot_data_count)
        if (epoch > 0 and epoch % eval_num == 0) or epoch == epochs - 1:
            dice_val = validation(model, val_loader, dice_metric, post_label, post_pred)
            dice_scores.append(dice_val)
            best_dice_val = max(dice_scores)
            if dice_val >= best_dice_val:
                state = {"model": model.state_dict(),
                         "opt": optimizer.state_dict(),
                         "loss": epoch_losses,
                         "dice": dice_scores,
                         "epoch": epoch,
                         }
                if load_weights:
                    file_name = "pretrain_unetr_btcv_best.pth"
                else:
                    file_name = "baseline_unetr_btcv_best.pth"
                torch.save(state, os.path.join(output_dir, file_name))
                print(f"Model Was Saved ! Current Best Avg. Dice: {best_dice_val} Current Avg. Dice: {dice_val}")
            else:
                print(f"Model Was Not Saved ! Current Best Avg. Dice: {best_dice_val} Current Avg. Dice: {dice_val}")
        epoch_loss = 0
    return


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
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250,
                                 b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image",
                            allow_smaller=True),
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
    dataset_val = get_btcv_dataset(args.data_path,
                                   args.data_json_name,
                                   subset="validation",
                                   transform=val_transforms)
    val_loader = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            )

    # Model
    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(args.image_size, args.image_size, args.image_size),
        feature_size=args.image_patch_size,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    # Load weights if using pretrained ViT3D
    if args.load_weights:
        vit_3d = vit_3d_base(
            image_size=args.image_size,
            image_patch_size=args.image_patch_size,
            frames=args.image_size,
            frame_patch_size=args.image_patch_size,
            num_classes=14,
            channels=1,
        )
        mae = MAE(encoder=vit_3d)
        checkpoint = torch.load(args.pretrained_mae_path)
        mae.load_state_dict(checkpoint['model'])
        model.vit.load_state_dict(mae.encoder.state_dict())
        print("Pretrained weights successfully loaded.")

    # Optimizer
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  )

    # Load half trained model if needed
    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        curr_epoch = checkpoint['epoch']
        dice_scores = checkpoint['dice']
        epoch_losses = checkpoint["loss"]
        print(f"Resuming from epoch: {curr_epoch}.")
    else:
        curr_epoch = 0
        dice_scores = []
        epoch_losses = []

    # Training
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True,
                             reduction="mean",
                             get_not_nans=False)

    train(model,
          optimizer,
          loss_function,
          args.epochs,
          train_loader,
          val_loader,
          args.eval_num,
          epoch_losses,
          dice_scores,
          args.output_dir,
          dice_metric,
          post_label,
          post_pred,
          args.load_weights,
          curr_epoch=curr_epoch,
          )


if __name__ == '__main__':
    arg = get_args()
    main(arg)
