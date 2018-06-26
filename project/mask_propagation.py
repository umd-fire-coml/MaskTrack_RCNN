import keras.layers as KL
import keras.models as KM
import numpy as np
import tensorflow as tf

from mrcnn.model import conv_block, identity_block
from pwc_net.model import PWCNet


class MaskPropagation(object):

    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

        assert mode in ['training', 'inference']

        self.tf_model = self._build()

    def _build(self):
        # set up image and mask inputs
        prev_image = KL.Input(shape=(None, None, 3))
        curr_image = KL.Input(shape=(None, None, 3))

        scaled_down = KL.Lambda(lambda x: tf.divide(x, 255))

        prev_scaled = scaled_down(prev_image)
        curr_scaled = scaled_down(curr_image)

        # feed images into PWC-Net to get optical flow field
        flow_field = PWCNet()(prev_scaled.output, curr_scaled.output)

        # feed masks and flow field into CNN (conv5)
        prev_masks = KL.Input(shape=(None, None, 1))

        x = KL.concatenate([prev_masks, flow_field])
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        mask_prop_conv = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        # return model
        mp_model = KM.Model(inputs=[prev_image, curr_image, prev_masks],
                            outputs=[mask_prop_conv])

        return mp_model


# test script
mp = MaskPropagation('training', None)
