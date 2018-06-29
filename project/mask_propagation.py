import os

import imageio
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import layers as TL

from pwc_net.model import PWCNet


class MaskPropagation(object):

    def __init__(self, mode, config, weights_path, model_dir='./logs', debugging=False, isolated=False):
        """
        Creates and builds the mask propagation network.
        :param mode: either 'training' or 'inference'
        :param config: not used atm
        :param weights_path: path to the weights for pwc-net
        :param model_dir: directory to save/load logs and model checkpoints
        :param debugging: whether to include extra print operations
        :param isolated: whether this is the only network running
        """
        self.name = 'maskprop'
        self.mode = mode
        self.config = config
        self.weights_path = weights_path
        self.model_dir = model_dir
        self.debugging = debugging

        if isolated:
            self.sess = tf.Session(tf.ConfigProto())
        else:
            self.sess = K.get_session()

        assert mode in ['training', 'inference']

        self._build()

    def _build(self):
        """
        Builds the computation graph for the mask propagation network.
        """
        # set up image and mask inputs
        self.prev_image = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='prev_image')
        self.curr_image = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='curr_image')

        if self.mode == 'training':
            self.gt_mask = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='gt_masks')

        prev = tf.divide(self.prev_image, 255)
        curr = tf.divide(self.curr_image, 255)

        # feed images into PWC-Net to get optical flow field
        x, _, _ = PWCNet()(prev, curr)

        self.flow_field = tf.image.resize_bilinear(x, tf.shape(prev)[1:3])

        # feed masks and flow field into CNN
        self.prev_mask = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='prev_masks')
        x = tf.concat([self.prev_mask, self.flow_field], axis=3)

        x = self._build_unet(x)
        self.propagated_masks = x

        if self.mode == 'training':
            self.loss = tf.losses.mean_squared_error(self.gt_mask, self.propagated_masks)
            trainables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='unet')
            self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.loss, var_list=trainables)

        # load weights for optical flow model from disk
        tf.global_variables_initializer().run(session=self.sess)

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pwcnet'))

        saver.restore(self.sess, self.weights_path)

    def _build_unet(self, x, conv_act=tf.nn.relu6, deconv_act=None):
        """
        Builds the mask propagation network proper (based on the u-Net architecture).
        :param x: input tensor of the previous mask and flow field concatenated [batch, w, h, 1+2]
        :param conv_act: activation function for the convolution layers
        :param deconv_act: activation function for the transposed convolution layers
        :return: output tensor of the U-Net [batch, w, h, 1]

        As a side effect, two instance variables unet_left_wing and unet_right_wing are set with the final output tensors
        for each layer of the two halves of the U.
        """
        with tf.variable_scope('unet'):
            input = x

            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='L1_conv1')
            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='L2_conv2')
            L1 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L2_pool')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='L2_conv1')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='L2_conv2')
            L2 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L3_pool')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='L3_conv1')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='L3_conv2')
            L3 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L4_pool')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='L4_conv1')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='L4_conv2')
            L4 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L5_pool')
            x = TL.conv2d(x, 1024, (3, 3), activation=conv_act, name='L5_conv1')
            x = TL.conv2d(x, 1024, (3, 3), activation=conv_act, name='L5_conv2')
            L5 = x

            x = TL.conv2d_transpose(x, 1024, (2, 2), strides=(2, 2), activation=deconv_act, name='P4_upconv')
            x = tf.concat([L4, tf.image.resize_images(x, tf.shape(L4)[1:3], name='P4_resize')],
                          axis=3, name='P4_concat')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='P4_conv1')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='P4_conv2')
            P4 = x

            x = TL.conv2d_transpose(x, 512, (2, 2), strides=(2, 2), activation=deconv_act, name='P3_upconv')
            x = tf.concat([L3, tf.image.resize_images(x, tf.shape(L3)[1:3], name='P3_resize')],
                          axis=3, name='P3_concat')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='P3_conv1')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='P3_conv2')
            P3 = x

            x = TL.conv2d_transpose(x, 256, (2, 2), strides=(2, 2), activation=deconv_act, name='P2_upconv')
            x = tf.concat([L2, tf.image.resize_images(x, tf.shape(L2)[1:3], name='P2_resize')],
                          axis=3, name='P2_concat')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='P2_conv1')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='P2_conv2')
            P2 = x

            x = TL.conv2d_transpose(x, 128, (2, 2), strides=(2, 2), activation=deconv_act, name='P1_upconv')
            x = tf.concat([L1, tf.image.resize_images(x, tf.shape(L1)[1:3], name='P1_resize')],
                          axis=3, name='P1_concat')
            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='P1_conv1')
            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='P1_conv2')
            P1 = x

            x = tf.image.resize_images(x, tf.shape(input)[1:3], name='P0_resize')
            x = TL.conv2d(x, 1, (1, 1), activation=tf.sigmoid, name='P0_conv')

            self.unet_left_wing = [L1, L2, L3, L4, L5]
            self.unet_right_wing = [P4, P3, P2, P1]

            return x

    # TODO add multi-epoch, multi-step train() method that uses a generator pattern to load images to train_single()

    def train_batch(self, prev_images, curr_images, prev_masks, gt_masks):
        """
        Trains the mask propagation network on a single batch of inputs.
        :param prev_images: previous images at time t-1 [batch, w, h, 3]
        :param curr_images: current images at time t [batch, w, h, 3]
        :param prev_masks: previous (correct) masks at time t-1 [batch, w, h, 1]
        :param gt_masks: ground truth next masks at time t [batch, w, h, 1]
        :return: batch loss of the predicted masks against the provided ground truths
        """
        assert self.mode == 'training'

        inputs = {self.prev_image: prev_images,
                  self.curr_image: curr_images,
                  self.prev_mask: prev_masks,
                  self.gt_mask: gt_masks}

        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=inputs)

        return loss

    def get_flow_field(self, prev_image, curr_image):
        """
        Evaluates the model to get the flow field between the two images.
        :param prev_image: starting image for flow [w, h, 3]
        :param curr_image: ending image for flow [w, h, 3]
        :return: flow field for the images [batch, w, h, 2]
        """
        inputs = {self.prev_image: np.expand_dims(prev_image, 0),
                  self.curr_image: np.expand_dims(curr_image, 0)}

        mask = self.sess.run(self.flow_field, feed_dict=inputs)

        return mask

    def propagate_mask(self, prev_image, curr_image, prev_mask):
        """
        Propagates the masks through the model to get the predicted mask.
        :param prev_image: starting image for flow at time t-1 [w, h, 3]
        :param curr_image: ending image for flow at time t [w, h, 3]
        :param prev_mask: previous (correct) mask at time t-1 [w, h, 1]
        :return: predicted/propagated mask at time t [1, w, h, 1]
        """
        inputs = {self.prev_image: np.expand_dims(prev_image, 0),
                  self.curr_image: np.expand_dims(curr_image, 0),
                  self.prev_mask: np.expand_dims(prev_mask, 0)}

        mask = self.sess.run(self.propagated_masks, feed_dict=inputs)

        return mask

    def save_weights(self, filename):
        weights_pathname = os.path.join(self.model_dir, filename)

        # TODO implement saving all weights
        pass

    def load_weights(self, filename):
        weights_pathname = os.path.join(self.model_dir, filename)

        # TODO implement loading all weights
        pass


# test script
def test():
    MODEL_PATH = 'C:/Users/tmthy/Documents/prog/python3/coml/MaskTrack_RCNN/pwc_net/model_3000epoch/model_3007.ckpt'
    mp = MaskPropagation('training', None, MODEL_PATH, debugging=True, isolated=False)

    img1 = imageio.imread('../pwc_net/test_images/frame1.jpg')
    img2 = imageio.imread('../pwc_net/test_images/frame2.jpg')

    oflow = mp.get_flow_field(img1, img2)
    plt.figure(1)
    plt.imshow(oflow[0, :, :, 0])
    plt.figure(2)
    plt.imshow(oflow[0, :, :, 1])
    plt.show()

    mp.propagate_mask(img1, img2, np.reshape(np.empty(img1.shape)[:, :, 0], (1080, 1349, 1)))

    mp.train_batch(img1, img2, np.empty((1080, 1349, 1)), np.empty((1080, 1349, 1)))

