from keras.utils import Sequence

class TensorflowDataGenerator(Sequence):

	@abstractmethod
    def slice_tensor(self, tensor):
        """Slices the tensor for input into the mask propagation module.
        # Returns
            A dictionary of the correct slices of the tensor.
            ```python
            result = {'prev_image':      slice,
                      'self.curr_image': slice,
                       'prev_mask':      slice,
                       'gt_mask':        slice}
            ```
        """
        raise NotImplementedError

class MPDataGenerator(TensorflowDataGenerator):
    """Documentation to be written
    """

    def __init__(self, re_id_module):

    	self.re_id_module = re_id_module
	self.m_len = 0
    
    def load_video(self, video_list_filename, labeled=True, assume_match=False):
        """Loads all the images from a particular video list into the dataset.
        video_list_filename: path of the file containing the list of images
        img_dir: directory of the images
        mask_dir: directory of the mask images, if available
        assume_match: Whether to assume all images have ground-truth masks
        """\
	
#         # Get list of images for this video
#         video_file = open(video_list_filename, 'r')
#         image_filenames = video_file.readlines()
#         video_file.close()

#         if image_filenames is None:
#             print('No video list found at {}.'.format(video_list_filename))
#             return

#         # Generate images and masks
#         for img_mask_paths in image_filenames:
#             # Set paths and img_id
#             if labeled:
#                 matches = re.search('^.*\\\\(.*\\.jpg).*\\\\(.*\\.png)', img_mask_paths)
#                 img_file, mask_file = matches.group(1, 2)
#                 img_id = img_file[:-4]
#             else:
#                 matches = re.search('^([0-9a-zA-z]+)', img_mask_paths)
#                 img_id = matches.group(1)
#                 img_file = img_id + '.jpg'
#                 mask_file = None

#             # Check if files exist
#             if not isfile(join(self.root_dir + '_color', img_file)):
#                 continue
#             if not assume_match and not isfile(join(self.root_dir + '_label', mask_file)):
#                 mask_file = None

#             # Add the image to the dataset
#             self.add_image("WAD", image_id=img_id, path=img_file, mask_path=mask_file)


    def load_mp_data(self):

    	pass

    def slice_tensor(self, tensor):
 
        raise NotImplementedError

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
        raise self.m_len

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass
