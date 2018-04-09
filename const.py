import os

MAIN_FOLDER = "C:/Users/titou/Desktop/Centrale/Option OSY/Deep Learning/data/train2017"
IMAGES_FOLDER = os.path.join(MAIN_FOLDER, "train2017")
IMAGES_FOLDER_TEST = os.path.join(MAIN_FOLDER, "val2017")
ANNOTATION_FILE = os.path.join(MAIN_FOLDER, "annotations/person_keypoints_train2017.json")
ANNOTATION_FILE_TEST = os.path.join(MAIN_FOLDER, "annotations/person_keypoints_val2017.json")
CHECKPOINTS_FOLDER = "./cktp/"