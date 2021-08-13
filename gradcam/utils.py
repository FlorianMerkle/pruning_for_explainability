import numpy as np
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def create_occlusion(act, thres):
    act = act.reshape(-1)
    cutoff = len(act) * thres
    indx = len(act) - int(cutoff)

    # Get the sorting indices and divide to values with zero and ones
    sort_ind = np.argsort(act)
    sort_zero = sort_ind[:indx]
    sort_ones = sort_ind[indx + 1:]

    # set to zero
    act[sort_zero] = 0.0
    # set to one
    act[sort_ones] = 1.0

    # Reshape back to original size
    act = act.reshape(224, 224)
    return act


def overlay_heatmap(img, activation_map, colormap='jet', alpha=0.7):
    """
    Function that displays the heatmap over the respective image.
    Inspired by https://github.com/frgfm/torch-cam
    :param img: image send to network
    :param activation_map: activation map created
    :param colormap: which cmap to apply
    :param alpha: the opacity of the heatmap
    :return: image with heatmap to be displayed
    """
    if not isinstance(colormap, str):
        raise ValueError('colormap must be string')
    if not isinstance(alpha, float):
        raise ValueError('alpha argument must be float')

    img = img.squeeze(0)
    activation_map = to_pil_image(activation_map, mode='F')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    activation_map = activation_map.resize(img.size, resample=Image.BICUBIC)
    activation_map = (255 * cmap(np.asarray(activation_map) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * activation_map).astype(np.uint8))

    return overlayed_img


def plot_gradcam(overlay):
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
