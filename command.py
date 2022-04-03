import torch
from moviepy.editor import * # pip install --trusted-host pypi.python.org moviepy
from roboflow import Roboflow # pip install roboflow
import os
import cv2

## What I added: a file named main.py
#import track.py
#import sys
#sys.path.insert(0, 'Yolov5_DeepSort_Pytorch/yolov5')
sys.path.insert(0, '/home/zhan3447/Merck_CV/Yolov5_DeepSort_Pytorch')
import track
sys.path.insert(0, '/home/zhan3447/Merck_CV/Yolov5_DeepSort_Pytorch/yolov5')
import train
import detect


def clip_video(data_filename, time_length):
    # loading video gfg
    clip = VideoFileClip(data_filename)
    # getting only first time_length seconds
    clip = clip.subclip(0, time_length)
    # save as mp4
    clip.write_videofile(f"{data_filename}-clip-{time_length}s.mp4")

def load_data(rf):
    os.environ["DATASET_DIRECTORY"] = "Yolov5_DeepSort_Pytorch/yolov5/datasets"
    ## temp data ##
    #project = rf.workspace("zhan3447-purdue-edu").project("single-bag-detection-57fdi")
    #dataset_train = project.version(2).download("yolov5")

    ## Single Bags ##
    # training data from single bag detection 1 and 2
    #project = rf.workspace().project("single_bag_detection_from_film")
    #dataset_train = project.version(2).download("yolov5")

    ## Multiple Bags ##
    # training data from multiple bags detection 1 and 2
    project = rf.workspace().project("multiple_bages_detection_from_film")
    dataset_train = project.version(1).download("yolov5")
    return dataset_train

def train_task(dataset_train):
    # python train.py --img 416 --batch 16 --epochs 100 --data {dataset_train.location}/data.yaml --weights yolov5s.pt --cache
    train.run(img=416, batch=16, epochs=100, data = f"{dataset_train.location}/data.yaml", weights = "yolov5s.pt")

def detect_task(source, our_yolo_model):
    # python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.2 --source {dataset_train.location}/valid/images
    detect.run(imgsz=(416, 416), conf_thres = 0.2, source = source, weights = our_yolo_model, save_txt = True)
    
def track_task(source, our_yolo_model, deepsort_yamlfile):
    track.run(imgsz=(416, 416), conf_thres = 0.2, source = source, yolo_model = our_yolo_model, save_txt = True, config_deepsort = deepsort_yamlfile) 

def main():
    print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

    # load training dataset - from Roboflow
    rf = Roboflow(api_key="9yUjZmEj7tKIDljA4Fk1")   
    dataset_train = load_data(rf)

    # clip video for detection
    #data_filename = "/home/zhan3447/Merck_CV/Film_Bag_data/Multiple_Bags_film1.mp4"
    #time_length = 2
    #clip_video(data_filename, time_length)

    bag_model = "/home/zhan3447/Merck_CV/Film_Bag_data/bag_model_new.pt"
    test_data = "/home/zhan3447/Merck_CV/Film_Bag_data/Multiple_Bags_film1.mp4-clip-2s.mp4"
    deepsort_yamlfile = "/home/zhan3447/Merck_CV/Yolov5_DeepSort_Pytorch/deep_sort/configs/deep_sort.yaml"

    
    command = input("What task do you want to perform? ")
    while True:
        if command == "train":
            train_task(dataset_train)
            break
        elif command == "detect":
            detect_task(source = test_data, our_yolo_model = bag_model)
            break
        elif command == "track":
            track_task(source = test_data, our_yolo_model = bag_model, deepsort_yamlfile =deepsort_yamlfile)
            break
        else:
            command = input("Please type it in again ")

    
 
main()