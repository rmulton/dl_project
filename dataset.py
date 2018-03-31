from pycocotools.coco import COCO
import numpy as np
import os
import torch.utils.data as data
import torch
from heatmap import heatmaps_from_keypoints
from imageio import imread
from skimage.transform import resize

class MSCOCO(data.Dataset):
    """ Represents a MSCOCO Keypoints dataset """
    
    def __init__(self, images_folder, annotations_json, train=False, evalu=False):
        """ Instantiate a MSCOCO dataset """
        super().__init__()
        
        self.images_folder = images_folder
        
        # Load the annotations
        self.annotations = COCO(annotations_json)
        imgs_id = self.annotations.getImgIds()
        if train:
            self.img_ids = imgs_id[:int(len(imgs_id)*2/3)]
        
        elif evalu:
            self.img_ids = imgs_id[int(len(imgs_id)*2/3)+1:]
        
        else:
            self.img_ids = imgs_id        
    
    def __len__(self):
        return len(self.img_ids)
            
    def __getitem__(self, index):
        """ Returns the index-th image with keypoints annotations, both as tensors """
        
        # Get the image informations
        img_id = self.img_ids[index]
        img = self.annotations.loadImgs(img_id)[0]
        
        # Load the image from the file
        img_path = os.path.join(self.images_folder, img['file_name'])
        img_array = load_image(img_path)
        # Black and white images
        if len(img_array.shape)==2:
            # Add a channel axis
            img_array = np.expand_dims(img_array, axis=2)
            # Fill all the axes with the black&white image
            img_array = np.concatenate((img_array, img_array, img_array), axis=2)
        img_array = np.transpose(img_array, (2,1,0))
        
        # Get the keypoints
        annIds = self.annotations.getAnnIds(imgIds=img['id'])
        anns = self.annotations.loadAnns(annIds)
        # Some images do not contain any coco object, so anns = []
        if len(anns)>0:
            keypoints = anns[0]['keypoints'] # anns is a list with only one element
        else:
            # keypoints are not visible so 
            keypoints = [0 for i in range(3*17)]
            
        # Check to avoid errors
        if len(keypoints)!=3*17:
            print('Warning: Keypoints list for image {} has length {} instead of 17'.format(img_id, len(keypoints)))
    
        # Generate the heatmaps
        heatmaps_array = heatmaps_from_keypoints(keypoints)
        
        # Transform arrays into tensors
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.float() # Pytorch needs a float tensor
        keypoints_tensor = torch.from_numpy(heatmaps_array).float() # Pytorch needs a float tensor
        
        return img_tensor, keypoints_tensor

# Homemade image loader
def load_image(image_path):
    image = imread(image_path)
    image = resize(image, (256, 256))
    return image