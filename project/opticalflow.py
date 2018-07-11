# add PWCNet library to path
import sys
sys.path.append("../pwc_net")

from pwc_net import PWCNet
import tensorflow as tf
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.transform import rescale

##################################################################
##
## This class can work for a tensor representing images or masks
##
## Usage:
## 
## model_path = '../model_3000epoch/model_3007.ckpt'
## model = OpticalFlow(model_path)
##
## image_prev, image_curr = model.read_images_from_path(path_prev, path_curr) 
## image_prev, image_curr, finalflow, flows, pyramid_0 = model.get_flow(image_prev, image_curr)
## 
## image_pred = model.apply_flow(image_prev, finalflow)
##
## model.plot_flow(image_prev, image_curr, image_pred)
##
##################################################################

class OpticalFlow(object):
  
  def __init__(self, model_path):
    
    # Run before any call to the model
    # Defines graph and restores trained weights
  
    tf.reset_default_graph()
    self.sess = tf.Session()
  
    self.img_prev = tf.placeholder(tf.float32, shape=(None, None, 3), name="img_prev") 
    self.img_curr = tf.placeholder(tf.float32, shape=(None, None, 3), name="img_curr")
  
    img_prev_p = tf.expand_dims(self.img_prev, axis = 0)
    img_curr_p = tf.expand_dims(self.img_curr, axis = 0)
  
    self.finalflow, self.flows, self.pyramid_0 = PWCNet()(img_prev_p, img_curr_p)
  
    saver = tf.train.Saver()
    saver.restore(self.sess, model_path)
    
  def get_flow(self, image_prev, image_curr, resize_ratio=4.0):
    
    """
    Get optical flow between two consecutive frames
    :param image_prev: previous frame as a numpy array
    :param image_curr: current frame as a numpy array
    """
    
    # Resize image
    image_prev = rescale(image_prev, 1.0 / resize_ratio)
    image_curr = rescale(image_curr, 1.0 / resize_ratio)
    
    r_finalflow, r_flows, r_pyramid_0 = self.sess.run(
        [self.finalflow, self.flows, self.pyramid_0],
        # note: for feed dict, input name must match the variable name exactly
        feed_dict={
            self.img_prev: image_prev,
            self.img_curr: image_curr
        }
    )
  
    return image_prev, image_curr, r_finalflow[0], r_flows[0], r_pyramid_0[0]
  
  def apply_flow(self, image_prev, finalflow):
    
    """
    Apply optical flow to previous frame to get predicted next frame
    """
    
    print(image_prev.shape)
    print(finalflow.shape)
    
    x_range, y_range, depth = image_prev.shape
    image_updated = np.zeros(shape=image_prev.shape)

    for x in range(x_range):
      for y in range(y_range):
        dx, dy = finalflow[x, y, :]
    
        # ensure that new coordinates are within original image dimensions
        xp = int(min(max(0, x + round(dx)), x_range - 1))
        yp = int(min(max(0, y + round(dy)), y_range - 1))
    
        # there might be collisions: two pixels might be moved to the same location
        # in which case the pixel with the highest x value, or highest y if a tie
        # will be the one placed in the spot in the image_updated
        image_updated[xp, yp, :] = image_prev[x, y, :]
        
    return image_updated
  
  # Utility functions
  
  def plot_flow(self, image_prev, image_curr, image_pred):
    fig, axarr = plt.subplots(1,3)
    fig.set_size_inches(24, 8)
    axarr[0].imshow(image_prev)
    axarr[1].imshow(image_curr)
    axarr[2].imshow(image_pred)
    
  def read_images_from_path(self, image_prev_path, image_curr_path):
    image_prev = io.imread(image_prev_path)
    image_curr = io.imread(image_curr_path)
    return image_prev, image_curr