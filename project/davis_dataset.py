import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import glob
from sklearn.model_selection import train_test_split


##################################################################
#
# Usage:
# dataset = DavisDataset("DAVIS", "480p") 
#
##################################################################


class DavisDataset(object):
  
  # TODO: in init include way to download dataset
  # include download link and expected directory structure
  
  def __init__(self, directory, quality):
    
    # generate mask pairs
    
    self.frame_pairs = [] # tuples of image and masks at t-1 and t
    
    image_dir = "%s/JPEGImages/%s/" % (directory, quality)
    mask_dir = "%s/Annotations/%s/" % (directory, quality)
    
    videos = [x[len(image_dir):] for x in glob.glob(image_dir + "*")]
    
    for video in videos:
      
      frames = [x[len(image_dir) + len(video) + 1:-4] for x in glob.glob(image_dir + video + "/*")]
      frames.sort()
      
      for prev, curr in zip(frames[:-1], frames[1:]):
        
        image_prev = image_dir + video + "/" + prev + ".jpg"
        image_curr = image_dir + video + "/" + curr + ".jpg"
        mask_prev = mask_dir + video + "/" + prev + ".png"
        mask_curr = mask_dir + video + "/" + curr + ".png"
      
        self.frame_pairs.append( (image_prev, image_curr, mask_prev, mask_curr) )
  
  def plot_random_frame_and_mask(self, frame_pair):
    fig, axarr = plt.subplots(3,2)
    fig.set_size_inches(16, 16)
    mask_prev = io.imread(frame_pair[2])
    mask_curr = io.imread(frame_pair[3])
    axarr[0][0].set_title("Image Prev")
    axarr[0][0].imshow(io.imread(frame_pair[0]))
    axarr[0][1].set_title("Image Curr")
    axarr[0][1].imshow(io.imread(frame_pair[1]))
    axarr[1][0].set_title("Mask Prev")
    axarr[1][0].imshow(mask_prev)
    axarr[1][1].set_title("Mask Curr")
    axarr[1][1].imshow(mask_curr)
    axarr[2][0].set_title("In Prev Mask, Not in Curr")
    axarr[2][0].imshow( (mask_prev == 255) & (mask_curr != 255) )
    axarr[2][1].set_title("In Curr Mask, Not in Prev")
    axarr[2][1].imshow( (mask_prev != 255) & (mask_curr == 255) )
    
  def get_training_set(self, val_size = 0.0, random_state = 42):
    
    frame_pairs = self.frame_pairs.copy()
    np.random.shuffle(frame_pairs)
    
    train, val = train_test_split(frame_pairs, test_size = val_size, random_state = random_state)
    return train, val
    
