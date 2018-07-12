from keras.utils import Sequence
import re
from os.path import join, isfile
import os
import csv
import numpy as np
import skimage.io

class TrajectoryDataGenerator(Sequence):
    """Documentation to be written
    Note that this generator will be only for training.
    We will use another one for test data (which does not include ground truth mask).
    """
    
    #CHANGE THIS
    dimensions = (256, 256)

    def __init__(self, batch_size, img_directory, mask_directory):
        """
        :param batch_size: the number of frames per batch (does account for instances)
        """
        
        self.batch = batch_size
        self.img_directory = img_directory
        self.mask_directory = mask_directory
        self.m_len = 0
        self.image_info = []
        self.video_map = []
        # 2-tuple list containing start and end indices of video in image_info
#         self.video_indices = []
        self.epoch_order = None
        self.on_epoch_end()
        self.input = {'flow_field': np.empty((batch_size, *dimensions, 2)),
                      'prev_mask': np.empty((batch_size, *dimensions, 1))}
        self.output = {'P0_conv': np.empty((batch_size, *dimensions, 1))}

    def load_video(self, video_list_filename):
        """Loads all the images from a particular video list into the dataset.
        video_list_filename: path of the file containing the list of images
        """

        with open(video_list_filename, 'rb') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')
            
#             start = m_len
            for row in reader:
                img_id_pair = (row[0], row[1])
                for c in row[:2]:
                    self.image_info.append(c)
                    self.video.append(img_id_pair)
                m_len += len(row) - 2
                
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
        
        map_index = index * self.batch_size
        n = 0

        unique_img_ids = {}
        while n < self.batch_size:
            
            mapped_i = self.epoch_order[n + map_index]
            data = self.image_info[mapped_i]
            img_id_pair = self.video_map[mapped_i]
            # change it so that the map is the pair, and values ARE the instances themselves
            unique_img_ids[img_id_pair[0]] = img_id_pair[1]
#             mask = skimage.io.imread(join(self.mask_directory, ))
#             self.input['flow_field'][n, :, :, :] = 
#             self.input['prev_mask'] = 
#             self.input['P0_conv'] = 
             n += 1
        
        unique_data = {}
        for k, v in unique_img_ids:
            
            # work in progress
            prev_img = skimage.io.imread(join(self.img_directory, k))
            curr_img = skimage.io.imread(join(self.img_directory, v))
            flow_field = None # (prev_img, curr_img)
            prev_img = None
            curr_img = None
            prev_mask = skimage.io.imread(join(self.mask_directory, k))
            curr_mask = 
            
        return self.input, self.output

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return self.m_len // self.batch_size

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.epoch_order = np.arange(self.m_len)
        np.random.shuffle(self.epoch_order)
