import argparse
import torch
from torchvision import transforms
from ..dataset.imagenet import ImageNet


def get_args():
    parser = argparse.ArgumentParser(description='MAE Pretraining on ImageNet')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')

    # Model parameters
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='masking ratio')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/imagenet1k', help='path to ImageNet data')
    parser.add_argument('--output_dir', type=str, default='/weights', help='path to save checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory for data loading')

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
    dataset = ImageNet(args.data_path, 'train', transform=transform_train)


if __name__ == '__main__':
    arg = get_args()
    main(arg)
