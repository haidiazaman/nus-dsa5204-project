import argparse
import torch
from torchvision import transforms
from src.dataset import ImageNet

def get_args():
    parser = argparse.ArgumentParser(description='MAE Pretraining on ImageNet')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')

    # Model parameters
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='masking ratio')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='path', help='path to ImageNet data')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='epochs to warmup LR')
    return parser.parse_args()


def main(args):

    device = torch.device(args.device)

    # Augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageNet(args.path, 'train', transform=transform_train)
