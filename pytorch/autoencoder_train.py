from alexnet_conv4 import AlexNetConv4
from video_dataloader import H5VideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
import matplotlib.pyplot as plt

dataset = H5VideoDataset('../data/filtered_smaller_period1_227.h5',
                         transforms=transforms.Normalize(mean=[], std=[]))
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=5)

#segment image into 12 pretrained objects
# model = FCN32(num_classes=16).cuda()
model = AlexNetConv4()
#get the images

for imgs in dataloader:
    inputs = Variable(imgs.cuda().transpose(1, 3))
    #output shape 5, 16, 227, 227
    #i.e. num_images, num_channels, height, width
    outputs = model(inputs)
    #ISSUE: What to compare this to????
    #first attempt
    #concatnate the 16 channels