import matplotlib
import matplotlib.pyplot as plt

import os
import random
import zipfile
import io
import scipy.misc
import numpy as np

import glob
import imageio
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
# from IPython.display import display, Javascript
# from IPython.display import Image as IPyImage

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import sys
sys.path.insert(0, '/mnt/data/TD_API/models/') 
sys.path.insert(0, '/mnt/data/TD_API/models/research')
sys.path.insert(0, '/mnt/data/TD_API/models/research/object_detection')
from utils import label_map_util
from utils import config_util
from utils import visualization_utils as viz_utils
from builders import model_builder
from helper import load_image_into_numpy_array, plot_detections


@dataclass
class InferenceConfig:
    num_classes: int
    category_index: dict
    pipeline_config: str
    checkpoint_path: str

@tf.function
def detect(detection_model, input_tensor):
    """Run detection on a input image.

    Args:
        detection_model: Model
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs to the model within the
            function.

    Returns:
    A dict containing 3 Tensors ('detection_boxes', 'detection_classes' and
        'detection_scores').
    """

    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)

    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def load_inference_config() -> InferenceConfig:
    """Helper to create dataclass of Inference Configuration

    Returns:
        Instance of InferenceConfig dataclass
    """
    # Assign the tree class ID
    tree_class_id = 1

    # Define a dictionary describing the tree class
    category_index = {tree_class_id: {'id': tree_class_id,
                                        'name': 'tree'}}

    # Number of classes
    num_classes = len(category_index.keys())

    # Pipeline config file
    pipeline_config = '/mnt/data/TD_API/model/v2/pipeline.config'

    # checkpoint path
    checkpoint_path = '/mnt/data/TD_API/model/v2/ckpt/ckpt-1'

    # Generate instance of Inference Config
    inference_config = InferenceConfig(num_classes = num_classes,
                                        category_index = category_index,
                                        pipeline_config = pipeline_config,
                                        checkpoint_path = checkpoint_path)

    return inference_config

def main():
    
    # 1. Load Inference Configuration
    inference_config = load_inference_config()

    # 2.a Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path = inference_config.pipeline_config, config_override = None)

    print(configs)
    
    # 2.b Get model config
    model_config = configs['model']

    # 3. Build model
    detection_model = model_builder.build(model_config = model_config,
                                        is_training = False)

    # 4. Load checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
    ckpt.restore(inference_config.checkpoint_path).expect_partial()

    # 5. Define testing directory
    test_image_dir = 'data'
    test_images_np = []

    # 6. Testing images
    test_image_paths = glob.glob(os.path.join(test_image_dir, '*'))
    
    for image_path in test_image_paths:
        print(image_path)
        test_images_np.append(np.expand_dims(load_image_into_numpy_array(image_path), axis = 0))


    # 7. Prediction
    label_id_offset = 1
    results = {'boxes': [], 'scores': []}

    for i in range(len(test_images_np)):
        input_tensor = tf.convert_to_tensor(test_images_np[i], dtype = tf.float32)
        detections = detect(detection_model, input_tensor)
        plot_detections(
            test_images_np[i][0],
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset,
            detections['detection_scores'][0].numpy(),
            inference_config.category_index,
            figsize = (15, 20),
            image_name = f'./results/{i}.jpg'
            )
        
        results['boxes'].append(detections['detection_boxes'][0][0].numpy())
        results['scores'].append(detections['detection_scores'][0][0].numpy())

    print(results)

if __name__ == "__main__":
    main()

