import denseFCN
from video_dataloader import H5VideoDataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch

def show_image(image):
    """ show an image"""
    plt.imshow(image.astype(np.int32))


vid_dataset = H5VideoDataset("../data/filtered_smaller_period1_227.h5")
vid_loader = DataLoader(vid_dataset, batch_size=5)
model = denseFCN.fcdensenet56(0.2, n_classes=3).cuda()
for data in vid_loader:
    inputs = Variable(data.transpose(3,1).cuda())
    outputs = model(inputs)
    #outputs shape [5, 3, 227, 227]
    #meaning 5 images of size 227by227 and 3 channels
    for imgs in outputs.data:
        out_img = imgs
        #criteria is the difference between this concatnated image and the original image
        out_img.transpose()

