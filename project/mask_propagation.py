import keras.layers as KL
import keras.models as KM

from mrcnn.model import BatchNorm, conv_block, identity_block

class MaskPropagation(object):

    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

        assert mode in ['training', 'inference']

    def build(self):
        # set up image and mask inputs
        curr_image = KL.Input(shape=(None, None, 3))
        prev_image = KL.Input(shape=(None, None, 3))
        prev_masks = KL.Input(shape=(None, None, 1))

        # feed images through PWC-Net for optical flow
        opt_flow = KL.Input(shape=(None, None, 1))

        # feed masks and flow field into CNN
        x = KL.concatenate([prev_masks, opt_flow])
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        # return model
        mp_model = KM.Model(inputs=[], outputs=[])

        return mp_model

