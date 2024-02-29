# DSA5204 Project

This repository hosts the code for our DSA5204 Project, which aims to reproduce and extend the work done in [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).

## Note to group members (This section will be removed upon submission)

- Do not commit any data! Just update markdown on how to download data and structure the files/folders
- Do not commit any Jupyter notebooks, write a script
- Use PyTorch
- Refer to [here](https://github.com/joelparkerhenderson/git-commit-message) to write good commit messages

## Development Setup

### Setting up environment

#### pip

To set up using pip, run the following:

    cd <project_root>
    pip install -r requirements.txt

### Project Structure (To be updated)

    .
    ├── ...
    ├── data                    # Put data in this folder. **Do not commit!**
    ├── src                     # Source code
    │   ├── dataset             # Dataloaders
    │   ├── model               # Model architectures
    │   └── ...                 # etc.
    └── ...

## Datasets

We make use of the following datasets:
- ImageNet1K (IN1K): Download the following files from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php): (1) Development kit (Tasks 1 & 2), (2) Training images (Task 1 & 2), and (3) Validation images (all tasks), place them in the `data/imagenet1k` folder and run the following code from root folder:

        from torchvision.datasets import ImageNet
        train = ImageNet(root='./data/imagenet1k', split='train')
        val = ImageNet(root='./data/imagenet1k', split='val')


