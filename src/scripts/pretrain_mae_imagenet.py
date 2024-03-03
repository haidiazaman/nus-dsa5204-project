import argparse
import numpy as np
import time
import torch

from torchvision import transforms

from ..dataset.imagenet import ImageNet
from ..model.vit import vit_large
from ..model.mae import MAE


def get_args():
    parser = argparse.ArgumentParser(description='MAE Pretraining on ImageNet')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=800, help='number of epochs to train')

    # Model parameters
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--image_patch_size', type=int, default=16, help='image patch size')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='masking ratio')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/imagenet1k', help='path to ImageNet data')
    parser.add_argument('--output_dir', type=str, default='src/checkpoints', help='path to save checkpoints')
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
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset_train = ImageNet(args.data_path, 'train', transform=transform_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )

    # Model
    vit = vit_large(
        image_size=args.image_size,
        patch_size=args.image_patch_size,
        num_classes=1000,
    )
    mae = MAE(encoder=vit)
    mae.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(mae.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training
    start_time = time.time()
    for epoch in range(args.epochs):
        mae.train()
        for img, _ in data_loader_train:
            img = img.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = mae(img)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % 50 == 0:
            torch.save(mae.state_dict(), f'{args.output_dir}/pretrain_mae_imagenet_epoch{epoch + 1}.pth')
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')


if __name__ == '__main__':
    arg = get_args()
    main(arg)
