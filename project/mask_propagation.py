import keras.layers as KL
import keras.models as KM

from mrcnn.model import BatchNorm, conv_block, identity_block

class MaskPropagation(object):

    def build(self, mode):
        assert mode in ['training', 'inference']

        # set up image and mask inputs
        # optical flow net inputs
        curr_image = KL.Input(shape=(None, None, 3))
        prev_image = KL.Input(shape=(None, None, 3))
        # CNN inputs
        prev_masks = KL.Input(shape=(None, None, 1))
        opt_flow = KL.Input(shape=(None, None, 1))

        # feed images through PWC-Net for optical flow

        # feed masks and flow field into CNN
        ###################################################################
        # Conv 5
        x = KL.concatenate([prev_masks, opt_flow])
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        ###################################################################

        # return model
        mp_model = KM.Model(inputs=[], outputs=[C5])

        return mp_model

