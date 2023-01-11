import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.ndimage import rotate

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    
    depth_major = record.reshape((3, 32, 32))

    
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE
    
    if training:
        # Randomly crop a [32, 32] section of the image. 
        layers = [np.pad(image[:,:,ch], ((4,4),(4,4)),'constant', constant_values=255) for ch in range(3)]
        image = np.stack(layers, axis=2)
        
        upper_left = np.random.randint(0,9, (2,1))
        image = image[int(upper_left[0]):int(upper_left[0])+32, int(upper_left[1]):int(upper_left[1])+32,:]
        
        # Randomly flip the image horizontally.
        image = image if np.random.rand()>0.4 else np.fliplr(image)

        # Randomly rotate
        image=   image if np.random.rand()>0.6 else rotate_img(image, np.random.randint(-45,45))
        
   
    
    
    ### YOUR CODE HERE
    image=(image - np.mean(image))/np.std(image)

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    #view preprocess
    # ppimage=image.reshape(3,32,32).transpose(1,2,0)
    # ppimage=preprocess_image(ppimage, True)
    # print('final',ppimage.shape)
    # # ppimage = ppimage.transpose(1,2,0)
    # plt.imshow(ppimage)
    # plt.savefig("pp"+save_name)


    image = image.reshape(3,32,32).transpose(1,2,0)


    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE
def random_crop(img, crop_size=(10, 10)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    print(img.shape)
    return img

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


### END CODE HERE
