# add PWCNet library to path
import sys

sys.path.append("../")

from pwc_net.model import PWCNet
# import tensorflow as tf
from keras.backend import tf
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.transform import rescale


##################################################################
#
# This class can work for a tensor representing images or masks
#
# Usage:
#
# model_path = '../PWC-Net/model_3000epoch/model_3007.ckpt'
# model = OpticalFlow(model_path, session)
#
# image_prev, image_curr = model.read_images_from_path(path_prev, path_curr)
# final_flow = model.get_flow(image_prev, image_curr)
#
# model.plot_flow(image_prev, image_curr, final_flow)
#
##################################################################

class OpticalFlowTFOld(object):

    def __init__(self, model_path, session):

        # Run before any call to the model
        # Defines graph and restores trained weights

        # tf.reset_default_graph()
        # self.sess = tf.Session()
        self.sess = session

        self.img_prev = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="img_prev")
        self.img_curr = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="img_curr")

        x, self.flows, self.pyramid_0 = PWCNet()(self.img_prev, self.img_curr)

        self.final_flow = x  # tf.image.resize_bilinear(x, tf.shape(self.img_prev)[1:3], name='flow_field')

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pwcnet'))
        saver.restore(self.sess, model_path)

    def infer_flow(self, image_prev, image_curr, resize_ratio=4.0):
        """
        Generate optical flow field between two frames
        :param image_prev: previous frame as a numpy array [batch, height, width, 3]
        :param image_curr: current frame as a numpy array [batch, height, width, 3]
        :return: flow field between input images [batch, height, width, 2]
        """
        def scaler(img):
            scale_factors = (1.0, 1.0 / resize_ratio, 1.0 / resize_ratio)
            return rescale(img, scale_factors, mode='constant', multichannel=True)

        image_prev = scaler(image_prev)
        image_curr = scaler(image_curr)

        final_flow = self.sess.run(self.final_flow,
            feed_dict={
                self.img_prev: image_prev,
                self.img_curr: image_curr
            }
        )

        return final_flow

    def _apply_flow(self, image_prev, final_flow):
        """DO NOT USE.
        Apply optical flow to previous frame to get predicted next frame
        """

        x_range, y_range, depth = image_prev.shape
        # image_updated = np.zeros(shape=image_prev.shape) # should not be zeros
        image_updated = np.copy(image_prev)  # start with image_prev, to keep nonmoving pixels

        for x in range(x_range):
            for y in range(y_range):
                dx, dy = final_flow[x, y, :]

                # ensure that new coordinates are within original image dimensions
                xp = int(min(max(0, x + round(dx)), x_range - 1))
                yp = int(min(max(0, y + round(dy)), y_range - 1))

                # there might be collisions: two pixels might be moved to the same location
                # in which case the pixel with the highest x value, or highest y if a tie
                # will be the one placed in the spot in the image_updated
                image_updated[xp, yp, :] = image_prev[x, y, :]

        return image_updated

    # Utility functions
    @staticmethod
    def plot_flow(image_prev, image_curr, image_pred):
        fig, axarr = plt.subplots(1, 3)
        fig.set_size_inches(24, 8)
        axarr[0].imshow(image_prev)
        axarr[1].imshow(image_curr)
        axarr[2].imshow(image_pred)

    @staticmethod
    def read_images_from_path(image_prev_path, image_curr_path):
        image_prev = io.imread(image_prev_path)
        image_curr = io.imread(image_curr_path)
        return image_prev, image_curr
