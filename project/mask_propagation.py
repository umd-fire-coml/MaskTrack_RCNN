import imageio
import keras.backend as K
import numpy as np
import skimage
import tensorflow as tf
from tensorflow.contrib.slim.nets.resnet_v2 import resnet_v2, resnet_v2_block

from pwc_net.model import PWCNet
from pwc_net.flow_utils import vis_flow


def _conv5(inputs,
           num_classes=None,
           is_training=True,
           global_pool=True,
           output_stride=None,
           reuse=tf.AUTO_REUSE,
           scope='resnet_v2_101'):
    blocks = [resnet_v2_block('block4', base_depth=512, num_units=3, stride=1), ]

    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=False,
                     reuse=reuse, scope=scope)


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
        # TODO freeze optical flow layers
        # feed masks and flow field into CNN (conv5)
        self.prev_masks = tf.placeholder(tf.float32, shape=(None, None, 1), name='prev_masks')

        x = tf.concat([self.prev_masks, self.flow_field], axis=0)

        x, not_sure_what_this_is = _conv5(x)
        # TODO add training losses
        self.propagated_mask = x  # put final tensor here

    def propagate_mask(self, prev_image, curr_image, prev_masks):
        sess = K.get_session()
        inputs = {self.prev_image: prev_image,
                  self.curr_image: curr_image,
                  self.prev_masks: prev_masks}

        mask = sess.run(self.flow_field, feed_dict=inputs)

        return mask

    # TODO add training section

# test script
mp = MaskPropagation('training', None)

img1 = imageio.imread('../pwc_net/test_images/frame1.jpg')
img2 = imageio.imread('../pwc_net/test_images/frame2.jpg')

oflow = mp.propagate_mask(img1, img2, np.reshape(np.empty(img1.shape)[:, :, 0], (1080, 1349, 1)))
print(oflow.shape)
flow_view = vis_flow(oflow)
skimage.io.imshow(flow_view)
