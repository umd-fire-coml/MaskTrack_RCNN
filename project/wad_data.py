import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import skimage.io

from mrcnn import config, utils
from os.path import join, isfile
from time import time


###############################################################################
#                              CLASS DICTIONARIES                             #
###############################################################################

classes = {
    33: 'car',
    34: 'motorcycle',
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
    162: 'motorcycle_group',
    163: 'bicycle_group',
    164: 'person_group',
    165: 'rider_group',
    166: 'truck_group',
    167: 'bus_group',
    168: 'tricycle_group',
    49: 'road',
    50: 'sidewalk',
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
    113: 'vegetation'
}

classes_to_index = dict([(e, i + 1) for i, e in enumerate(classes.keys())])
index_to_classes = {v: k for k, v in classes_to_index.items()}

###############################################################################
#                                CONFIGURATION                                #
###############################################################################


class WADConfig(config.Config):
    NAME = 'WAD'

    NUM_CLASSES = len(classes) + 1

###############################################################################
#                                   DATASET                                   #
###############################################################################


class WADDataset(utils.Dataset):
    image_height = 2710
    image_width = 3384

    _train_all_image_info_filename = 'train-all_image_info.pkl'

    def _load_video(self, video_list_filename, img_dir, train, mask_dir=None):
        """Loads all the images from a particular video list into the dataset.
        video_list_filename: path of the file containing the list of images
        img_dir: directory of the images (full, color)
        train: if this is training data or test data (datatype: boolean)
        mask_dir (Optional): directory of the mask data

        If train is False, mask_dir is ignored.
        """

        # Get list of images for this video
        video_file = open(video_list_filename, 'r')
        image_filenames = video_file.readlines()
        video_file.close()

        for img_mask_paths in image_filenames:
            # Set paths and img_id
            if train:
                matches = re.search('^.*\\\\(.*\\.jpg).*\\\\(.*\\.png)', img_mask_paths)
                img_file, mask_file = matches.group(1, 2)
                img_id = img_file[:-4]
            else:
                matches = re.search('^([0-9a-zA-z]+)', img_mask_paths)
                img_id = matches.group(1)
                img_file = img_id + '.jpg'

            # Paths
            img_path = join(img_dir, img_file)
            mask_path = join(mask_dir, mask_file) if train else None

            # Check if files exist
            if not isfile(img_path):
                continue
            elif not isfile(mask_path):
                mask_path = None

            # Add the image to the dataset
            self.add_image("WAD", image_id=img_id, path=img_path, mask_path=mask_path)

    def _load_all_images(self, img_dir, mask_dir=None, pickle_dir=None):
        """Load all images from the img_dir directory, with corresponding masks
        if doing training.
        img_dir: directory of the images
        mask_dir: directory of the corresponding masks
        """

        # Retrieve list of all images in directory
        for _, _, images in os.walk(img_dir):
            break

        # Iterate through images and add to dataset
        for img_filename in images:
            img_id = img_filename[:-4]
            img_path = join(img_dir, img_filename)

            # If using masks, only add images to dataset that also have a mask
            if mask_dir is not None:
                mask_path = join(mask_dir, img_id + '_instanceIds.png')

                # Ignores the image (doesn't add) if no mask exists
                if not isfile(mask_path):
                    continue
            else:
                mask_path = None

            # Adds the image to the dataset
            self.add_image('WAD', img_id, img_path, mask_path=mask_path)

        pickle_path = join(pickle_dir if pickle_dir is not None else '', self._train_all_image_info_filename)

        with open(pickle_path, 'wb') as f:
            pickle.dump(self.image_info, f, pickle.HIGHEST_PROTOCOL)

    def load_WAD(self, root_dir, subset, pickle_dir=None):
        """Load a subset of the WAD image segmentation dataset.
        root_dir: Root directory of the data
        subset: Which subset to load: train-video, train-all, test-video, test-all
        pickle_dir: For train-all, directory of the pickle file containing image_info,
        or the directory to save the pickle file
        """

        # Add classes (35)
        for class_id, class_name in classes.items():
            self.add_class(class_name, classes_to_index[class_id], class_name)

        # Set up directories
        assert subset in ['train-video', 'train-all', 'test-video', 'test-all']
        train = subset in ['train-video', 'train-all']

        img_dir = os.path.join(root_dir, 'train_color' if train else 'test')
        mask_dir = os.path.join(root_dir, 'train_label') if train else None

        # Process images by video
        if subset.endswith('video'):
            # Set up directories and paths
            video_list_dir = os.path.join(root_dir, 'train_video_list' if train else 'list_test_mapping')
            for _, _, video_files_list in os.walk(video_list_dir):
                break

            # Load images by video (according to their mappings)
            for video_file in video_files_list:
                self._load_video(join(video_list_dir, video_file), img_dir, train, mask_dir=mask_dir)
        else:
            # Use previously generated pickle file if it exists
            pickle_path = join(pickle_dir if pickle_dir is not None else '', self._train_all_image_info_filename)

            if isfile(pickle_path):
                print('Using pickle file for train-all: {}'.format(self._train_all_image_info_filename))
                with open(self._train_all_image_info_filename, 'rb') as f:
                    self.image_info = pickle.load(f)
            else:
                # Process all available images
                self._load_all_images(img_dir, mask_dir, pickle_dir=pickle_dir)

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
        unique = np.unique(raw_mask)

        # section that removes/involves background
        index = np.searchsorted(unique, 255)
        unique = np.delete(unique, index, axis=0)

        # tensors!
        raw_mask = raw_mask.reshape(2710, 3384, 1)

        # broadcast!!!!
        # k = instance_count
        # (h, w, 1) x (k,) => (h, w, k) : bool array
        masks = raw_mask == unique

        # get the actually class id
        # int(PixelValue / 1000) is the label (class of object)
        unique = np.floor_divide(unique, 1000)
        class_ids = np.array([classes_to_index[e] for e in unique])

        # Return mask, and array of class IDs of each instance.
        return masks, class_ids

    def image_reference(self, image_id):
        """Return the path to the image."""

        return self.image_info[image_id]["path"]

###############################################################################
#                               TESTING SCRIPTS                               #
###############################################################################


def test_loading():
    # SET THESE AS APPROPRIATE
    root_dir = 'G:\\Team Drives\\COML-Summer-2018\\Data\\CVPR-WAD-2018'
    subset = 'train-all'

    # Load and prepare dataset
    start_time = time()

    wad = WADDataset()
    wad.load_WAD(root_dir, subset)
    wad.prepare()

    print('Time to Load and Prepare Dataset = {} seconds'.format(time() - start_time))

    # Check number of classes and images
    image_count = len(wad.image_info)
    print('No. Images:\t{}'.format(image_count))
    print('No. Classes:\t{}'.format(len(wad.class_info)))

    # Choose a random image to display
    which_image = np.random.randint(0, image_count)
    print('\nShowing Image No. {}\n'.format(which_image))

    # Display original image
    plt.figure(0)
    plt.title('Image No. {}'.format(which_image))
    img_path = skimage.io.imread(wad.image_info[which_image]['path'])
    plt.imshow(img_path)

    # Display masks if available
    if wad.image_info[which_image]['mask_path'] is not None:
        # Set up grid of plots for the masks
        masks, labels = wad.load_mask(which_image)
        num_masks = masks.shape[2]
        rows, cols = math.ceil(math.sqrt(num_masks)), math.ceil(math.sqrt(num_masks))
        plt.figure(1)

        # Plot each mask
        for i in range(num_masks):
            instance_class = classes[index_to_classes[labels[i]]]

            frame = plt.subplot(rows, cols, i+1)
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.title('Mask No. {0} of class {1}'.format(i, instance_class))
            plt.imshow(np.uint8(masks[:, :, i]))

    plt.show()

