import os
import skimage.io
import matplotlib.pyplot as plt
import numpy as np

# Import Mask RCNN
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize


%matplotlib inline

# Root directory of the project
ROOT_DIR = os.path.abspath("C:\\Users\\rmdu\\MaskTrack_RCNN")
os.chdir(ROOT_DIR)
print('Project Directory: {}'.format(ROOT_DIR))

# Root directory of the dataset
DATA_DIR = os.path.join(ROOT_DIR, "dataset\\wad")
print('Data Directory: {}'.format(DATA_DIR))

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
print('Logs and Model Directory: {}'.format(LOGS_DIR))

from samples.balloon import balloon
config = balloon.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "dataset\\balloon")

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = balloon.BalloonDataset()
dataset.load_balloon(BALLOON_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))

print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


def crop_image_by_mask(image, mask, same_dim=False):
    """Returns an array with the actual image pixel values for the mask
    image: the image to get pixel values
    mask: the mask to select the pixel values from the image (bool np array)
    same_dim: if true then the returned array has the same dimensions as the image
              otherwise the returned array has the smallest dimension possible
    Returns:
    an array with the actual image pixel values in place of the mask
    """

    assert len(
        image.shape) == 3, 'just images, no batch here'  # ,'are you a >3d being whose images are >rank 3 tensors?!'
    assert len(mask.shape) == 2, 'mask should be [height, width]'
    assert image.shape[:2] == mask.shape

    if same_dim:
        return image * mask[:, :, np.newaxis]

    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0

    return image[y1:y2, x1:x2] * mask[y1:y2, x1:x2, np.newaxis]

# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    masks, class_ids = dataset.load_mask(image_id)
    plt.imshow(image)
    plt.show()
    for i in range(0, masks.shape[-1]):
        cropped_mask = crop_image_by_mask(image , masks[:, :, i])
        plt.imshow(cropped_mask)
        plt.show()

