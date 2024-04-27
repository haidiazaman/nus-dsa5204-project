import argparse
import numpy as np
import time
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from ..dataset.tinyimagenet import TinyImageNet
from ..model.vit import vit_base


def get_args():
    parser = argparse.ArgumentParser(description='MAE Fine-tuning on TinyImageNet')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')

    # Model parameters
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--image_patch_size', type=int, default=4, help='image patch size')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='masking ratio')
    parser.add_argument('--pretrained_model', type=str,
                        default='src/checkpoints/pretrain_mae_tinyimagenet_epoch600.pth',
                        help='path to pretrained model')

    # Dataset parameters
    parser.add_argument('--logs_dir', type=str, default='src/logs', help='path to save logs')
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
    #writer = SummaryWriter(log_dir=args.logs_dir)

    # Augmentation
    transform_train = transforms.Compose([
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset_train = TinyImageNet(split='train', transform=transform_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )

    dataset_val = TinyImageNet(split='valid', transform=transform_val)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    # Model
    vit = vit_base(
        image_size=args.image_size,
        patch_size=args.image_patch_size,
        num_classes=200,
    )
    mae_pretrained = torch.load(args.pretrained_model)
    vit_state_dict = vit.state_dict()
    for key, value in mae_pretrained.items():
        if key in vit_state_dict and vit_state_dict[key].shape == value.shape:
            vit_state_dict[key] = value
    vit.load_state_dict(vit_state_dict)
    vit.to(device)

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training
    start_time = time.time()
    for epoch in range(args.epochs):
        vit.train()
        for batch_idx, (img, target) in enumerate(data_loader_train):
            print(f'Epoch: {epoch + 1}, Training Batch: {batch_idx + 1}')
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            pred = vit(img)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        # Validation
        vit.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (img, target) in enumerate(data_loader_val):
                print(f'Epoch: {epoch + 1}, Validation Batch: {batch_idx + 1}')
                img, target = img.to(device), target.to(device)
                pred = vit(img)
                _, predicted = torch.max(pred.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            validation_accuracy = 100 * correct / total
            print(f'Validation Accuracy: {validation_accuracy:.2f}')
            #writer.add_scalar('Validation Accuracy', validation_accuracy, epoch)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    #writer.flush()
    #writer.close()


if __name__ == '__main__':
    arg = get_args()
    main(arg)
