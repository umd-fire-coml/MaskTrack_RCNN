import keras.layers as KL
import keras.models as KM


class MaskPropagation(object):

    def build(self, mode):
        assert mode in ['training', 'inference']

        # set up image and mask inputs
        curr_image = KL.Input(shape=(None, None, 3))
        prev_image = KL.Input(shape=(None, None, 3))
        prev_masks = KL.Input(shape=(None, None, 1))

        # feed images through PWC-Net for optical flow

        # feed masks and flow field into CNN

        # return model
        mp_model = KM.Model(inputs=[], outputs=[])

        return mp_model

