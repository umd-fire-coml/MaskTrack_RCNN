from keras.utils import Sequence
import re
from os.path import join, isfile
import csv

class TrajectoryDataGenerator(Sequence):
    """Documentation to be written
    Note that this generator will be only for training.
    We will use another one for test data (which does not include ground truth mask).
    """

    def __init__(self, re_id_module):

        self.re_id_module = re_id_module
        self.m_len = 0
        self.image_info = []
        self.video_indices = []
        # indicies to pull from when generating batch
        self.indices = []
        self.epoch_order = None
        
    def add_image(self, image_id, path, mask_path, prev_index):
        image_info = {
            "id": image_id,
            "path": path,
            "mask_path": mask_path,
            "prev_index": prev_index
        }
        self.image_info.append(image_info)
    
    def load_video(self, video_list_filename):
        """Loads all the images from a particular video list into the dataset.
        video_list_filename: path of the file containing the list of images
        """

        # nothing to do with image
        # instance id
        prev_ids = None

        with open(video_list_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # nothing to do with image
                # instance id
                curr_ids = {}

                row_iter = iter(row)

                img_id = row_iter.next()

                

            # Add the image to the dataset
            self.add_image(image_id=img_id, path=img_file, mask_path=mask_file)

    def load_mp_data(self):

        pass

    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        raise NotImplementedError

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return self.m_len

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass
