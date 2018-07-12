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
        # 3-tuple list containing 1st, 2nd and end indices
        self.video_indices = []
        self.epoch_order = None
        
    def add_data(self, image_id, local_id, global_id, prev_index):
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

        with open(video_list_filename, 'rb') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            first_row = next(reader)

            # nothing to do with image
            # instance id
            prev_ids = {}
            start_index = 0
            image_info_index = 0
            
            col_iter = iter(row)

                img_id = next(row_iter)

            while True:
                try:
                    # get the next item
                    local_id = next(col_iter)
                except StopIteration:
                    # if StopIteration is raised, break from loop
                    break
                # expect and assume
                global_id = next(col_iter)

                prev_ids[global_id] = image_info_index

                # Add to the dataset
                self.add_data(image_id=img_id, local_id, global_id, None)
                image_info_index += 1
            
            for row in reader:
                # nothing to do with image
                # instance id
                curr_ids = []

                col_iter = iter(row)

                img_id = next(row_iter)

                while True:
                    try:
                        # get the next item
                        local_id = next(col_iter)
                    except StopIteration:
                        # if StopIteration is raised, break from loop
                        break
                    # expect and assume
                    global_id = next(col_iter)
                    curr_ids.append((local_id, global_id))
                    
                    # Add to the dataset
                    self.add_data(image_id=img_id, local_id, global_id, prev_ids[global_id])
                    
                 if prev_ids:
                    pass
                 prev_ids = curr_ids

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
