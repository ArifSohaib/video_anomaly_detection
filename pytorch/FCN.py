#-------- FCN-32 implementation part --------
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import video_dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

def upsample_filt(size):
    """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)






def conv3x3(in_planes, out_planes, stride=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(padding, padding))


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0)


class FCN_32(nn.Module):
    def __init__(self):
        super(FCN_32, self).__init__()

        # vgg part
        self.conv1_1 = conv3x3(3, 64, stride=1, padding=100)
        self.conv1_2 = conv3x3(64, 64)

        self.conv2_1 = conv3x3(64, 128)
        self.conv2_2 = conv3x3(128, 128)

        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(256, 256)
        self.conv3_3 = conv3x3(256, 256)

        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(512, 512)
        self.conv4_3 = conv3x3(512, 512)

        self.conv5_1 = conv3x3(512, 512)
        self.conv5_2 = conv3x3(512, 512)
        self.conv5_3 = conv3x3(512, 512)

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)

        self.score_fr_sem = nn.Conv2d(
            4096, 34, kernel_size=1, stride=1, padding=0, bias=False)

        self.upscore_sem = nn.ConvTranspose2d(
            34, 34, kernel_size=64, stride=32, padding=0, output_padding=0, bias=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax()

        self._initialize_weights()

    def forward(self, x):
        # vgg part
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.pool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.pool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.pool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.pool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))
        pool5 = self.pool(conv5_3)

        fc6 = self.dropout(self.relu(self.fc6(pool5)))
        fc7 = self.dropout(self.relu(self.fc7(fc6)))

        score_fr_sem = self.score_fr_sem(fc7)

        upscore_sem = self.upscore_sem(score_fr_sem)

        crop = upscore_sem[:, :, 19:19 + 256,
                           19:19 + 256]  # batch, 34, 256, 256

        crop = crop.transpose(1, 3)
        crop = crop.transpose(1, 2)  # batch, 256, 256, 34

        output = crop.contiguous().view(-1, crop.size(3))

        output = self.softmax(output)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def main():

    #-------- main part --------
    model = FCN_32()

    pth_file = '../model/vgg16.pth'  # download from model zoo
    pre_trained_weights = torch.load(pth_file)

    layer_names = [layer_name for layer_name in pre_trained_weights]

    counter = 0
    for p in model.parameters():
        if counter < 26:  # conv1_1 to pool5
            p.data = pre_trained_weights[layer_names[counter]]
        elif counter == 26:  # fc6 weight
            p.data = pre_trained_weights[layer_names[counter]].view(4096, 512, 7, 7)
        elif counter == 27:  # fc6 bias
            p.data = pre_trained_weights[layer_names[counter]]
        elif counter == 28:  # fc7 weight
            p.data = pre_trained_weights[layer_names[counter]].view(4096, 4096, 1, 1)
        elif counter == 31:   # upscore layer
            m, k, h, w = 34, 34, 64, 64
            filter = upsample_filt(h)
            filter = torch.from_numpy(filter.astype('float32'))
            p.data = filter.repeat(m, k, 1, 1)
        counter += 1
    print(model)    
    dataset = video_dataloader.H5VideoDataset(
        '../data/filtered_smaller_period1_227.h5')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=5)
    for val in dataloader:
        break
    val = Variable(val)
    output = model(val.transpose(1,3))
    # print(output)
    plt.imshow(output.numpy())
    plt.show()

if __name__ == '__main__':
    main()
