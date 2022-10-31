# Load packages
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from cv import predict_tree_count
import numpy as np
import cv2
import os
import datetime
import random
import string
import sys
import json
sys.path.insert(0, '/mnt/data/TD_API/models/')
sys.path.insert(0, '/mnt/data/TD_API/models/research')
sys.path.insert(0, '/mnt/data/TD_API/models/research/object_detection')
from helper import load_image_into_numpy_array, plot_detections
from tfod import load_inference_config, load_model, detect
import tensorflow as tf

# Preload Tensorflow Object Detection Model
# inference_config_dict = load_inference_config()
# detection_model = load_model()
inference_config_dict, detection_model_dict = load_model()

# Initialize instance of FastAPI
app = FastAPI(title = "Tree Counting FastAPI")

# CORS Middleware
app.add_middleware(
        CORSMiddleware,
        allow_origins = ["*"],
        allow_methods = ["GET", "POST"],
        allow_headers = ["*"],
)

def check_extension(file: UploadFile):
    extension = file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")

    if not extension: 
        raise HTTPException(status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail = "Image must be in jpg or png")

def generate_input_output_filename():
    
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d_%H%M%S')
    uuid = ''.join(random.choices(string.ascii_uppercase, k = 8))
    in_file = os.path.join('input', f'input_{now_str}_{uuid}.jpg')
    out_file = os.path.join('output', f'output_{now_str}_{uuid}.jpg')

    return in_file, out_file

# Get output + header
def get_tree_count_response(contents):
    """
    Args: 
        contents: Reads file in ["jpg", "jpeg", "png"]

    Return:
        headers: response header
        out_file: output image file
    """
    arr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Predict tree count
    tree_count, bbox_list, imgOut = predict_tree_count(img)
    
    # Create output folder if not exists
    os.makedirs('output', exist_ok = True)

    # Save input and output to disk
    in_file, out_file = generate_input_output_filename()
    # now = datetime.datetime.now()
    # now_str = now.strftime('%Y%m%d_%H%M%S')
    # uuid = ''.join(random.choices(string.ascii_uppercase, k = 8))
    # in_file = os.path.join('input', f'input_{now_str}_{uuid}.jpg') # Added 20221018 - Save input image
    cv2.imwrite(in_file, img)
    # out_file = os.path.join('output', f'output_{now_str}_{uuid}.jpg') # Modified 20220914 - PNG to JPG
    cv2.imwrite(out_file, imgOut)

    # headers
    headers = {'tree_count': str(tree_count),
              'bounding_boxes': bbox_list} # Added 20221024 - Add bounding boxes

    # return header and out_file
    return headers, out_file

def get_tree_count_response_tfod(contents, version = 'v2'):
    arr = np.fromstring(contents, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    higher_dim = max(height, width)
    if higher_dim > 640: # Modified 20221024 - Modify to 640
        resize_factor = float(640/ higher_dim)
        img = cv2.resize(img, (0,0), fx = resize_factor, fy = resize_factor)

    test_image_np = np.expand_dims(img, axis = 0)

    input_tensor = tf.convert_to_tensor(test_image_np, dtype = tf.float32)
    detections = detect(detection_model_dict.get(version), input_tensor)

    in_file, out_file = generate_input_output_filename()
    cv2.imwrite(in_file, img_bgr)

    plot_detections(test_image_np[0],
                    detections['detection_boxes'][0].numpy(),
                    detections['detection_classes'][0].numpy().astype(np.uint32) + 1,
                    detections['detection_scores'][0].numpy(),
                    inference_config_dict.get(version).category_index,
                    figsize = (15,20),
                    image_name = out_file)
    
    if len(detections['detection_boxes'][0].numpy()) > 0:
        detections_boxes_np = detections['detection_boxes'][0].numpy() # Added 20221024
        multiply_arr = np.array([height, width, height, width])
        detections_boxes_np = detections_boxes_np * multiply_arr
        detections_boxes_np = detections_boxes_np.astype(np.int32)
        
        # Get bounding boxes for those exceeding min score thresh
        flag_arr = detections['detection_scores'][0].numpy() > inference_config_dict.get(version).min_score_thresh
        detections_boxes_np = detections_boxes_np[flag_arr]

        detections_boxes_list = detections_boxes_np.tolist()
    else:
        detections_boxes_list = []
    
    tree_count = sum(detections['detection_scores'][0].numpy() > inference_config_dict.get(version).min_score_thresh)
    headers = {'tree_count': str(tree_count),
              'bounding_boxes': json.dumps(detections_boxes_list)}

    return headers, out_file

# Index
@app.get("/")
def root():
    return {"message": "Tree Counting"}

@app.post("/predict/tree_count")
async def get_tree_count(file: UploadFile = File(...)):

    check_extension(file)

    contents = await file.read()
    headers, out_file = get_tree_count_response(contents)
    
    return headers

@app.post("/predict/tree_count/v2") # Added 20221018 - TensorFlow Object Detection
async def get_tree_count(file: UploadFile = File(...)):
    
    check_extension(file)

    contents = await file.read()
    headers, out_file = get_tree_count_response_tfod(contents, version = 'v4') #  'v2')

    return headers

# @app.post("/predict/tree_count/v3")
# async def get_tree_count(file: UploadFile = File(...)):
#     check_extension(file)

#     contents = await file.read()
#     headers, out_file = get_tree_count_response_tfod(contents, version = 'v3')

    return headers

@app.post("/predict/tree_bbox")
async def get_tree_bbox(file: UploadFile = File(...)):
    
    check_extension(file)
    
    contents = await file.read()
    headers, out_file = get_tree_count_response(contents)

    return FileResponse(out_file, headers = headers)

@app.post("/predict/tree_bbox/v2")
async def get_tree_bbox(file: UploadFile = File(...)):

    check_extension(file)
    contents = await file.read()
    headers, out_file = get_tree_count_response_tfod(contents, version = 'v4') # 'v2')

    return FileResponse(out_file, headers = headers)

# @app.post("/predict/tree_bbox/v3")
# async def get_tree_bbox(file: UploadFile = File(...)):
#     check_extension(file)
#     contents = await file.read()
#     headers, out_file = get_tree_count_response_tfod(contents, version = 'v3')

#     return FileResponse(out_file, headers = headers)

if __name__ == "__main__":
    uvicorn.run(app)
