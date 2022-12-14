import tensorflow as tf
import numpy as np
from PIL import Image
from six import BytesIO

def load_image_into_numpy_array(path):
    """ Load an image from file to numpy array

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels = 3 for RGB.

    Args:
    path: a file path

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
