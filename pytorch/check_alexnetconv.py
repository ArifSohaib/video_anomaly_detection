from alexnet_conv4 import AlexNetConv4
from video_dataloader import H5VideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

dataset = H5VideoDataset('../data/filtered_smaller_period1_227.h5',
                         transforms=transforms.Normalize(mean=[], std=[]))
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=5)

#segment image into 12 pretrained objects
# model = FCN32(num_classes=16).cuda()
model = AlexNetConv4().cuda()
#get the images

for imgs in dataloader:
    inputs = Variable(imgs.cuda().transpose(1, 3))
    #output shape [5, 256, 13, 13]
    #i.e. num_images, num_output_channels, filter_h, filter_w
    outputs = model(inputs)
    print(outputs.data.shape)
    print(outputs.data)
    plt.imshow(outputs.cpu().data.numpy()[1,1,:,:])
    plt.show()
    break
