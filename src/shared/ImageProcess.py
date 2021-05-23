import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage
import tensorflow as tf

MASK_RCNN_DIR = os.getenv('MASK_RCNN_DIR')
assert os.path.exists(MASK_RCNN_DIR)

# Import mrcnn libraries
sys.path.append(MASK_RCNN_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

MODEL_DIR = os.path.join(MASK_RCNN_DIR, "logs")

class VehicleConfig(Config):
    NAME = "vehicle"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 7  # background + 6 (classes)
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 5
    BACKBONE = 'resnet101'

class InferenceConfig(VehicleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.60

global graph
graph = tf.compat.v1.get_default_graph()

class ImageProcess():
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        inference_config = InferenceConfig()
        print("Loading model from ", MODEL_DIR)
        
        self.model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
        self.model_path = self.model.find_last()

        print("Loading weights from ", self.model_path)
        self.model.load_weights(self.model_path, by_name=True)

    def predict(self, image):
        # self.model_path = self.model.find_last()

        # print("Loading weights from ", self.model_path)
        # self.model.load_weights(self.model_path, by_name=True)

        image_path = 'C:/Experiment/aktwelve_vehicle/datasets/vehicle/test/NVR_ch1_main_20201021000000_20201021010000_0051.jpg'
        img = skimage.io.imread(image_path)
        print(img)
        img_arr = np.array(img)

        with graph.as_default():
            results = self.model.detect([img_arr], verbose=1)
        print('DONE')
        return 'results'