"""
checks if fcn32s model is working
"""
from FCN32 import fcn32s #fcn32s has pretrained data from imagenet
from video_dataloader import H5VideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

class AlexNetConv4(nn.Module):
    """
    outputs the 4th convolution layer of pretrained AlexNet
    """
    def __init__(self):
        super(AlexNetConv4, self).__init__()
        self.original_model  = models.alexnet(
                    pretrained=True)
        self.features = nn.Sequential(
            # stop at conv4
            *list(self.original_model.features.children())[:-3]
        )

    def forward(self, x):
        x = self.features(x)
        return x


model = AlexNetConv4().cuda()

dataset = H5VideoDataset('../data/filtered_smaller_period1_227.h5')#,transforms = transforms.Normalize(mean=[], std=[]))
batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10 )

#segment image into 12 pretrained objects
# model = FCN32(num_classes=16).cuda()
# model = models.densenet121(pretrained=True).cuda()

#get the images
print(model)
for imgs in dataloader:
    break

inputs = Variable(imgs.cuda().transpose(1,3))
outputs = model(inputs)
#shape of output is batch_size x 256 x 13 x 13
print(outputs.data.shape)

#get a target of size batch_size and fill it with all 0s
targets = torch.zeros(batch_size).cuda()

#build a one class classifier that outputs everything as zero


# output_t = outputs.transpose(1,3).cpu().data.numpy()
# print("output_t_shape: {}".format(output_t.shape))
# print("output_t:{}".format(output_t))
# for i in range(output_t.shape[-1]):
#     plt.imshow(output_t[9,:,:,i])
#     plt.show()
