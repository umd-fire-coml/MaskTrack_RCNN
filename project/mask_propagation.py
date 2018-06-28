import imageio
import keras.backend as K
import numpy as np
import skimage
import tensorflow as tf

from pwc_net.model import PWCNet
from pwc_net.flow_utils import vis_flow


class MaskPropagation(object):

    def __init__(self, mode, config):
        self.name = 'maskprop'
        self.mode = mode
        self.config = config

        assert mode in ['training', 'inference']

        self._build()

    def _build(self):
        # set up image and mask inputs

        self.prev_image = tf.placeholder(tf.float32, shape=(None, None, 3), name='prev_image')
        self.curr_image = tf.placeholder(tf.float32, shape=(None, None, 3), name='curr_image')

        def scale_and_expand(v):
            return tf.expand_dims(tf.expand_dims(v[0], axis=0), axis=0) / 255

        prev = scale_and_expand(self.prev_image)
        curr = scale_and_expand(self.curr_image)

        # feed images into PWC-Net to get optical flow field
        _, flows, _ = PWCNet()(prev, curr)
        self.flow_field = flows[-1]

        # feed masks and flow field into CNN (conv5)
        self.prev_masks = tf.placeholder(tf.float32, shape=(None, None, 1), name='prev_masks')

        # TODO implement in tensorflow

        # OLD STUFF IN KERAS
        # x = KL.concatenate([prev_masks, flow_field], axis=3, name='merge_block_inputs')
        # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        # mask_prop_conv = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        self.propagated_mask = None  # put final tensor here

    def propagate_mask(self, prev_image, curr_image, prev_masks):
        sess = K.get_session()
        inputs = {self.prev_image: prev_image,
                  self.curr_image: curr_image,
                  self.prev_masks: prev_masks}

        mask = sess.run(self.flow_field, feed_dict=inputs)

        return mask


# test script
mp = MaskPropagation('training', None)

img1 = imageio.imread('../pwc_net/test_images/frame1.jpg')
img2 = imageio.imread('../pwc_net/test_images/frame2.jpg')

oflow = mp.propagate_mask(img1, img2, np.reshape(np.empty(img1.shape)[:, :, 0], (1080, 1349, 1)))
print(oflow.shape)
flow_view = vis_flow(oflow)
skimage.io.imshow(flow_view)
