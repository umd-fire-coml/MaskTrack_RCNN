import os
import re
from mrcnn import config, utils
import skimage.io
import numpy as np
from os.path import join, isfile


# TRAIN IMAGES ISSUE
# No. Files in video lists = 42,369
# No. Files in train_color = 39,222
# No. Files in train_label = 37,689


###############################################################################
#                               CLASS DICTIONARY                              #
###############################################################################

classes = {
    33: 'car',
    34: 'motorbicycle',
    35: 'bicycle',
    36: 'person',
    37: 'rider',
    38: 'truck',
    39: 'bus',
    40: 'tricycle',
    0: 'others',
    1: 'rover',
    17: 'sky',
    161: 'car_groups',
    162: 'motorbicycle_group',
    163: 'bicycle_group',
    164: 'person_group',
    165: 'rider_group',
    166: 'truck_group',
    167: 'bus_group',
    168: 'tricycle_group',
    49: 'road',
    50: 'siderwalk',
    65: 'traffic_cone',
    66: 'road_pile',
    67: 'fence',
    81: 'traffic_light',
    82: 'pole',
    83: 'traffic_sign',
    84: 'wall',
    85: 'dustbin',
    86: 'billboard',
    97: 'building',
    98: 'bridge',
    99: 'tunnel',
    100: 'overpass',
    113: 'vegatation',
    255: 'background'
}

###############################################################################
#                                CONFIGURATION                                #
###############################################################################


class WADConfig(config.Config):
    NAME = 'WAD'

    NUM_CLASSES = len(classes)

###############################################################################
#                                   DATASET                                   #
###############################################################################


class WADDataset(utils.Dataset):
    image_height = 2710
    image_width = 3384

    def load_video(self, video_list_filename, img_dir, train, mask_dir=None):
        """Loads all the images from a particular video list into the dataset.
        video_list_filename: path of the file containing the list of images
        img_dir: directory of the images (full, color)
        train: if this is training data or test data (dtype: boolean)
        mask_dir (Optional): directory of the mask data
        """

        # Get list of images for this video
        video_file = open(video_list_filename, 'r')
        image_filenames = video_file.readlines()
        video_file.close()

        for img_mask_paths in image_filenames:
            # Set paths and img_id
            if train:
                matches = re.search('^.*\\\\(.*\\.jpg).*\\\\(.*\\.png)', img_mask_paths)
                img_path, mask_path = matches.group(1, 2)
                img_id = img_path[:-4]
            else:
                matches = re.search('^([0-9a-zA-z]+)', img_mask_paths)
                img_id = matches.group(1)
                img_path = img_id + '.jpg'

            mask_path = join(mask_dir, mask_path) if train else None

            # Add the image to the dataset
            self.add_image("WAD", image_id=img_id, path=join(img_dir, img_path), mask_path=mask_path)

    def load_WAD(self, root_dir, subset):
        """Load a subset of the WAD image segmentation dataset.
        root_dir: Root directory of the data
        subset: Which subset to load: train or test
        """

        # Add classes (36)
        for class_id, class_name in classes.items():
            self.add_class(class_name, class_id, class_name)

        # Set up directories
        assert subset in ['train', 'test']
        train = subset == 'train'

        # Set up directories and paths
        video_list_dir = os.path.join(root_dir, 'train_video_list' if train else 'list_test_mapping')
        video_files_list = [f for f in os.listdir(video_list_dir) if isfile(join(video_list_dir, f))]

        img_dir = os.path.join(root_dir, 'train_color' if train else 'test')
        mask_dir = os.path.join(root_dir, 'train_label') if train else None

        # Load images by video (according to their mappings)
        for video_file in video_files_list:
            self.load_video(join(video_list_dir, video_file), img_dir, train, mask_dir=mask_dir)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        image_id: integer id of the image
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]

        # If not a WAD dataset image, delegate to parent class
        if info["source"] != "WAD":
            return super(self.__class__, self).load_mask(image_id)

        # Read the original mask image
        raw_mask = skimage.io.imread(info["mask_path"])

        # unique is a sorted array of unique instances (including background)
        # which is basically class_ids
        # unique_inverse is an array of indices that correspond to unique
        unique, unique_inverse = np.unique(raw_mask, return_inverse=True)

        # we can reconstruct anyway
        raw_mask = None

        ##########################################
        # section that removes/involves background
        index = np.searchsorted(unique, 255)
        unique = np.delete(unique, index, axis=0)

        # prepare to broadcast
        instance_count = unique.shape[0]
        # make array of all possible indices
        # [0, 1, ..., index - 1, index + 1, index + 2, ... , instance_count]
        caster = np.array(list(range(0, index))
                          + list(range(index + 1, instance_count + 1)))
        ##########################################

        # get the actually class id
        # int(PixelValue / 1000) is the label (class of object)
        class_ids = np.floor_divide(unique, 1000)

        unique_inverse = unique_inverse.reshape(WADDataset.image_height,
                                                WADDataset.image_width, 1)

        # broadcast!!!!
        # k = instance_count
        # (h, w, 1) x (1, 1, k) => (h, w, k) : bool array
        masks = unique_inverse == caster

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        return masks, class_ids

    def image_reference(self, image_id):
        """Return a link to the image."""

        return self.image_info[image_id]["path"]

###############################################################################
#                              TESTING & SCRIPTS                              #
###############################################################################


def test_loading():
    root_dir = 'G:\\Team Drives\\COML-Summer-2018\\Data\\CVPR-WAD-2018'

    wad = WADDataset()
    wad.load_WAD(root_dir, 'train')

    print(len(wad.image_info))

    img = skimage.io.imread(wad.image_info[0]['path'])
    skimage.io.imshow(img)
    skimage.io.show()

    print(wad.image_info[3000])

    masks, labels = wad.load_mask(3000)

    print(masks.shape)

    skimage.io.imshow(np.uint16(masks[:, :, 0]))
    skimage.io.show()
