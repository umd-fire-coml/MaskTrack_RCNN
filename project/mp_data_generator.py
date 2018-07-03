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
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass