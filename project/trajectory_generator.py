####
# THIS CODE IS BEING TEST, EXERCISE EXTREME CAUTION
####

from keras.utils import Sequence
import re
from os.path import join, isfile
import os
import numpy as np
import skimage.io
import csv

class TrajectoryData(object):
    
    def __init__(self):
        
        self.m_len = 0
        self.image_info = []
    
    def add_data(self, prev_img_id, curr_img_id, prev_ins_id, curr_ins_id):
        
        image_info = {'prev_img_id': prev_img_id,
                      'curr_img_id': curr_img_id,
                      'prev_ins_id': prev_ins_id,
                      'curr_ins_id': curr_ins_id}
        self.image_info.append(image_info)

    def load_video(self, video_list_path, header=False):
        """Loads all the images from a particular video list into the dataset.
        video_list_path: path of the file containing the list of images
        """

        with open(video_list_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # skip header
            if header:
                next(reader)
            for row in reader:
                prev_img_id = row[1]
                curr_img_id = row[2]
                prev_ins_id = int(row[4])
                curr_ins_id = int(row[5])
                self.add_data(prev_img_id, curr_img_id, prev_ins_id, curr_ins_id)
                m_len += len(row) - 2

    def load_all_videos(self, directory, header=False):

        _, _, files = next(os.walk(vid_list_dir))
        for name in files:
            load_video(join(directory, name), header)


class TrajectoryDataGenerator(Sequence):
    """Documentation to be written
    Note that this generator will be only for training.
    We will use another one for test data (which does not include ground truth mask).
    """
    
    #CHANGE THIS
    dimensions = (256, 256)

    def __init__(self, batch_size, img_directory, mask_directory, optical_flow, image_info):
        """
        :param batch_size: the number of frames per batch (does account for instances)
        """
        
        self.batch = batch_size
        self.img_directory = img_directory
        self.mask_directory = mask_directory
        self.optical_flow = optical_flow

        self.m_len = len(image_info)
        self.image_info = image_info
#         self.video_map = []
        # 2-tuple list containing start and end indices of video in image_info
#         self.video_indices = []
        self.epoch_order = None
        self.on_epoch_end()
        self.input = {'flow_field': np.empty((batch_size, *dimensions, 2)),
                      'prev_mask': np.empty((batch_size, *dimensions, 1))}
        self.output = {'P0_conv': np.empty((batch_size, *dimensions, 1))}

    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        
        map_index = index * self.batch_size
        n = 0

        while n < self.batch_size:
            
            mapped_i = self.epoch_order[n + map_index]
            data = self.image_info[mapped_i]
            
            # work in progress
            prev_img = skimage.io.imread(join(self.img_directory, data['prev_img_id']))
            curr_img = skimage.io.imread(join(self.img_directory, data['curr_img_id']))
            self.input['flow_field'][n] = self.optical_flow.get_flow(prev_img, curr_img)
            # collect garbage
            prev_img = None
            curr_img = None

            self.input['prev_mask'][n] = skimage.io.imread(join(self.mask_directory, data['prev_img_id'])) == data['prev_ins_id']
            self.output['P0_conv'][n] = skimage.io.imread(join(self.mask_directory, data['curr_img_id'])) == data['curr_ins_id']

            n += 1
            
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
