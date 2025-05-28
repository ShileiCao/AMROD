from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import cv2
import os



corruption_types = get_corruption_names()
severity = 5
root_dir =  "./datasets"
raw_cityscapes_dir = os.path.join(root_dir, "cityscapes/leftImg8bit/val")
b = ["gaussian_noise", "shot_noise", "impulse_noise"]
for corruption in corruption_types:
    corrupt_root_dir = os.path.join(root_dir, corruption)
    if corruption in b:
        continue
    else:
        corrupt_cityscapes_dir = os.path.join(root_dir, corruption+"/leftImg8bit/val")
        if not os.path.exists(corrupt_cityscapes_dir):
            os.makedirs(corrupt_cityscapes_dir)
        print(corruption)
        for root, _, files in os.walk(raw_cityscapes_dir):
            city = root.split("/")[-1]
            if city == "val":
                continue
            corrupt_cityscapes_city_dir = os.path.join(corrupt_cityscapes_dir, city)
            if not os.path.exists(corrupt_cityscapes_city_dir):
                os.makedirs(corrupt_cityscapes_city_dir)
            for filename in files:
                image = cv2.imread(os.path.join(root, filename))
                corrupted = corrupt(image, corruption_name=corruption, severity=severity)
                cv2.imwrite(os.path.join(corrupt_cityscapes_city_dir, filename), corrupted)
        