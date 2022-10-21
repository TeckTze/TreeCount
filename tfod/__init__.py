import sys
import os
import glob
import numpy as np
from six import BytesIO
from dataclasses import dataclass
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

sys.path.append('/mnt/data/TD_API/models/')
sys.path.append('/mnt/data/TD_API/models/research')
sys.path.append('/mnt/data/TD_API/models/research/object_detection')

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
    min_score_thresh: float

def load_inference_config(version) -> InferenceConfig:
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
    pipeline_config_dict = {'v2': '/mnt/data/TD_API/model/v2/pipeline.config',
                            'v3': '/mnt/data/TD_API/model/v3_mobilenet/pipeline.config'}
    pipeline_config = pipeline_config_dict[version] # '/mnt/data/TD_API/model/v3_mobilenet/pipeline.config' # '/mnt/data/TD_API/model/v2/pipeline.config'

    # checkpoint path
    checkpoint_path_dict = {'v2': '/mnt/data/TD_API/model/v2/ckpt/ckpt-1',
                            'v3': '/mnt/data/TD_API/model/v3_mobilenet/ckpt/ckpt-1'}
    checkpoint_path = checkpoint_path_dict[version] # '/mnt/data/TD_API/model/v3_mobilenet/ckpt/ckpt-1' # '/mnt/data/TD_API/model/v2/ckpt/ckpt-1'
    
    # Min Score threshold
    min_score_thresh = 0.8

    # Generate instance of Inference Config
    inference_config = InferenceConfig(num_classes = num_classes,
                                        category_index = category_index,
                                        pipeline_config = pipeline_config,
                                        checkpoint_path = checkpoint_path,
                                        min_score_thresh = min_score_thresh)

    return inference_config

def load_model():

    # 1. Load Inference Configuration
    inference_config_dict = {'v2': load_inference_config(version = 'v2'),
                            'v3': load_inference_config(version = 'v3')}

    detection_model_dict = {}
    
    for version, inference_config in inference_config_dict.items():
        # 2.a Load the configuration file into a dictionary
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path = inference_config.pipeline_config, config_override = None)

        # 2.b Get model config
        model_config = configs['model']

        # 3. Build model
        detection_model = model_builder.build(model_config = model_config,
                                        is_training = False)

        # 4. Load checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
        ckpt.restore(inference_config.checkpoint_path).expect_partial()
        
        # 5. Add to model dictionary
        detection_model_dict[version] = detection_model

    return inference_config_dict, detection_model_dict
    # return detection_model

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
