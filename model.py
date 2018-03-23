import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from torch.autograd.variable import Variable
import os
from skimage.transform import resize
import matplotlib.pyplot as plt

from dataset import load_image
from heatmap import keypoints_from_heatmaps, get_xs_ys_vs


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, training=True, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            stride=stride)

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.training = training

    def forward(self, x):
        x = self.relu(self.conv(x))
        if self.training:
            x = self.batch_norm(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.feature_extraction = nn.Sequential(
                ConvRelu(3, 64, 3),
                ConvRelu(64, 64, 3),
                self.pool,
                ConvRelu(64, 128, 3),
                ConvRelu(128, 128, 3),
                self.pool,
                ConvRelu(128, 128, 3),
                ConvRelu(128, 128, 3),
                self.pool,
                ConvRelu(128, 512, 3),
                ConvRelu(512, 512, 3),
                )
        
        self.features_to_heatmaps = nn.Conv2d(512, 17, 1) # 17 kind of joints, 17 heatmaps

    def forward(self, x):
        x = self.feature_extraction(x)
        heatmaps = self.features_to_heatmaps(x)
        return heatmaps

def plotKeypointsOverOutputModel(index,dataset,model,img_folder):
    """Forward a img to the model and display the output keypoints over the image.
       It enables us to see the loss evolution over the model visually over the image
       index is the index of the img in the dataset argument"""
    # Get an image
    imgId = dataset.img_ids[index]
    img, keypoints = dataset[index]

    # Transform into a pytorch model input and Forward pass 
    y = model(Variable(img.unsqueeze(0)))

    #Get the coordinates of the keypoints
    keypoints = keypoints_from_heatmaps(y[0].data.numpy())

    # Plot the image
    img_anno = dataset.annotations.loadImgs(imgId)[0]
    img_path = os.path.join(img_folder, img_anno['file_name'])
    img_array = load_image(img_path)
    img_array_resized = resize(img_array, (512, 640))
    plt.figure()
    plt.title('Original image')
    plt.imshow(img_array_resized)
    xs,ys,vs = get_xs_ys_vs(keypoints)
    plt.plot(xs,ys,'ro',color='c')
    plt.show()