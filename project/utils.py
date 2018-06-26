#######################################
### Put your utility functions here ###
#######################################

def crop_image_by_mask(image, mask, same_dim=False):
    """Returns an array with the actual image pixel values for the mask
    image: the image to get pixel values
    mask: the mask to select the pixel values from the image (bool np array)
    same_dim: if true then the returned array has the same dimensions as the image
              otherwise the returned array has the smallest dimension possible
    Returns:
    an array with the actual image pixel values in place of the mask
    """

    assert len(image.shape) == 3 , 'just images, no batch here'#,'are you a >3d being whose images are >rank 3 tensors?!'
    assert len(mask.shape) == 2, 'mask should be [height, width]'
    assert image.shape[:2] == mask.shape
    
    if same_dim:
      
      return image * mask[:, :, np.newaxis]

    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0

    return image[y1:y2, x1:x2] * mask[y1:y2, x1:x2, np.newaxis]
