from config import *
import matplotlib.pyplot as plt

def visualize_images(images, labels, no_images=25):
    """
    method to visualize no_images images

    :param images: np.ndarray, a numpy array of all the images
    :param labels: list, a list of all the labels
    :param no_images: int, number of images to be visualized
    """
    fig, ax = plt.subplots(int(no_images/5), 5, sharex=True, sharey=True, figsize=(14, 6))
    for i in range(no_images):
        ax[i//5, i%5].imshow(images[i])
        ax[i//5, i%5].set_xticks([])
        ax[i//5, i%5].set_yticks([])
        ax[i//5, i%5].set_title(labels[i])
    fig.tight_layout()
    plt.show()

