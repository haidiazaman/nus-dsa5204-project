# DSA5204 Project

This repository hosts the code for our NUS DSA5204 Project, which aims to reproduce and extend the work done in [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
This diagram shows the model architecture for the paper. The encoder is a ViT and decoder is a set of transformer blocks.
![alt text](https://github.com/liawzhengkai/dsa5204-project/blob/main/imgs/mae_architecture.png)

## Development Setup

### Setting up environment

#### pip

To set up using pip, run the following:

    cd <project_root>
    pip install -r requirements.txt

### Project Structure
    .
    ├── src                         # Source code
    │   ├── dataset                 # Dataloaders
    │   ├── checkpoints             # To put reproduction weights in this folder
    │   ├── model                   # Model architectures
    │   ├── utilities               # Utility functions
    │   └── scripts                 # Training, Evaluation scripts  
    ├── inference_notebook_examples # Quick look on our model inference results after training
    ├── .gitignore
    ├── README.md
    └── requirements.txt

## Datasets

We make use of the following datasets:
- Reproduction, TinyImagenet: Data is available at https://huggingface.co/datasets/zh-plus/tiny-imagenet. Instead of manual download, it can be called using the huggingface library directly.
- Time Series, ETTh1: Data is available at: https://github.com/zhouhaoyi/ETDataset
- 2d Segmentation, ADE20k: Data is available at: http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip. Unzip and place directly in data folder as is
- 3d Segmentation, BTCV: https://www.synapse.org/#!Synapse:syn3193805

## Results
Reproduction of paper results using TinyImageNet dataset. Example results of image reconstruction of TinyImageNet data using MAE architecture. For each triplet, the leftmost image displays the base image, the middle image displays the input data and the rightmost image displays the reconstructed image.
![alt text](https://github.com/liawzhengkai/dsa5204-project/blob/main/imgs/tinyimgnet.png)

Extension 1: Time Series forecast. The smoothed models’ forecast predictions on the test dataset is shown. A smoothing window has been
applied using a moving average with a window size of 20 timesteps. This results in plots that only capture the general trend.
![alt text](https://github.com/liawzhengkai/dsa5204-project/blob/main/imgs/time_series_results.png)

Extension 2: 2D Segmentation. Example results of the models conducting semantic segmentation. MAE pretraining shows improvement over no MAE pretraining.
![alt text](https://github.com/liawzhengkai/dsa5204-project/blob/main/imgs/semseg_output.png)

Extension 3: 3D Segmentation. Example 3D Segmentation result. MAE pretraining shows improvement over no MAE pretraining.
![alt text](https://github.com/liawzhengkai/dsa5204-project/blob/main/imgs/3d_seg_output.png)

Extension 4: Data Imputation. Training loss over epochs comparison. MAE shows unstable training and eventually poorer results compared to no MAE.
![alt text](https://github.com/liawzhengkai/dsa5204-project/blob/main/imgs/imputation_res.png)

## Model weights
Trained model weights can be accessed via this Google drive link: https://drive.google.com/drive/u/0/folders/1oI8RIMEDl6vOW0-mutafXjPSTByXshNO

## References 

- Upernet: https://github.com/CSAILVision/semantic-segmentation-pytorch
- Time Series: https://github.com/asmodaay/ti-mae
