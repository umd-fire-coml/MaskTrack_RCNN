from keras.utils import Sequence
import re
from os.path import join, isfile
import os
import csv
import numpy as np

class TrajectoryDataGenerator(Sequence):
    """Documentation to be written
    Note that this generator will be only for training.
    We will use another one for test data (which does not include ground truth mask).
    """

    def __init__(self, step_size, mask_directory):
        """
        :param step_size: the number of frames per batch (does not account for instances)
        """
        
        self.step_size = step_size
        self.mask_directory = mask_directory
        self.m_len = 0
        self.image_info = []
        # 2-tuple list containing start and end indices of video in image_infox
#         self.video_indices = []
        self.epoch_order = None
        self.on_epoch_end()

    def load_video(self, video_list_filename):
        """Loads all the images from a particular video list into the dataset.
        video_list_filename: path of the file containing the list of images
        """

        with open(video_list_filename, 'rb') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')
            
#             start = m_len
            for row in reader:
                
                self.image_info.append(row)
                m_len += 1
                
#             video_indices.append((start, m_len))

    def load_all_videos(self, directory):

        _, _, files = next(os.walk(vid_list_dir))
        for name in files:
            load_video(join(directory, name))

    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        
        i = index * self.step_size
        end = i + self.step_size
        while i < end:
            
            mapped_i = self.epoch_order[i]
            
            image_info

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return self.m_len // self.step_size

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.epoch_order = np.arange(self.m_len)
        np.random.shuffle(self.epoch_order)
