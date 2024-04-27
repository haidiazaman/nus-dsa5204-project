import numpy as np

from einops.layers.torch import Rearrange


def reconstruct_image(patches,
                      model_input,
                      masked_indices=None,
                      pred_pixel_values=None,
                      patch_size=4,
                      mean=[0.485, 0.456, 0.406],  # mean for TinyImageNet
                      std=[0.229, 0.224, 0.225]  # std for TinyImageNet
                      ):
    """
    Reconstructs the image given patches. Can also reconstruct the masked image as well as the predicted image.
    To reconstruct the raw image from the patches, set masked_indices=None and pred_pixel_values=None. To reconstruct
    the masked image, set masked_indices= the masked_indices tensor created in the forward call. To reconstruct the
    predicted image, set masked_indices and pred_pixel_values = to their respective tensors created in the forward call.

    ARGS:
        patches (torch.Tensor): The raw patches (pre-patch embedding) generated for the given model input. Shape is
            (batch_size x num_patches x patch_size^2 * channels)
        model_input (torch.Tensor): The input images to the given model (batch_size x channels x height x width)
        masked_indices (torch.Tensor): The patch indices that are masked (batch_size x masking_ratio * num_patches)
        pred_pixel_values (torch.Tensor): The predicted pixel values for the patches that are masked
            (batch_size x masking_ratio * num_patches x patch_size^2 * channels)
        patch_size (int): The size of the patches
        mean (list): The mean values for the dataset
        std (list): The standard deviation values for the dataset

    RETURN:
        reconstructed_image (torch.Tensor): Tensor containing the reconstructed image
            (batch_size x channels x height x width)
    """
    patches = patches.cpu()

    masked_indices_in = masked_indices is not None
    predicted_pixels_in = pred_pixel_values is not None

    if masked_indices_in:
        masked_indices = masked_indices.cpu()

    if predicted_pixels_in:
        pred_pixel_values = pred_pixel_values.cpu()

    patch_width = patch_height = patch_size
    reconstructed_image = patches.clone()

    if masked_indices_in or predicted_pixels_in:
        for i in range(reconstructed_image.shape[0]):
            if masked_indices_in and predicted_pixels_in:
                reconstructed_image[i, masked_indices[i].cpu()] = pred_pixel_values[i, :].cpu().float()
            elif masked_indices_in:
                reconstructed_image[i, masked_indices[i].cpu()] = 0

    invert_patch = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                             w=int(model_input.shape[3] / patch_width),
                             h=int(model_input.shape[2] / patch_height), c=model_input.shape[1],
                             p1=patch_height, p2=patch_width)

    reconstructed_image = invert_patch(reconstructed_image)

    reconstructed_image = reconstructed_image.detach().numpy().transpose(0, 2, 3, 1)
    #
    # reconstructed_image *= np.array(std)
    # reconstructed_image += np.array(mean)

    return reconstructed_image
