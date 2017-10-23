import torch
from torch import nn

__all__ = ['FCDenseNet', 'fcdensenet_tiny', 'fcdensenet56_nodrop',
           'fcdensenet56', 'fcdensenet67', 'fcdensenet103']

class DenseBlock(nn.Module):
    def __init__(self, nIn, growth_rate, depth, drop_rate=0, only_new=False, bottle_neck=False):
        """
        Args:
            nIn: size of input layer 
            growth_rate: size K to be appended to each layer channel
            depth: number of layers
            drop_rate: dropout rate
            only_new:
            bottle_neck:
        """
        super(DenseBlock, self).__init__()
        self.only_new = only_new
        self.depth = depth
        self.growth_rate = growth_rate
        #ModuleList contains an iterable list of modules
        #for the depth 
        self.layers = nn.ModuleList([self.get_transform(nIn + i * growth_rate,
                                    growth_rate, bottle_neck, drop_rate)
                                    for i in range(depth)])
    
    def forward(self,x):
        if self.only_new:
            #if there is no previous layer
            outputs = []
            for i in range(self.depth):
                #get output of layer i
                tx = self.layers[i](x)
                #concat the output with the input in dimension 1
                x  = torch.cat((x, tx), 1)
                #append to outputs
                outputs.append(tx)
            #return the concatnated tensor
            return torch.cat(outputs, 1)
        else:
            for i in range(self.depth):
                #concat the input with the output of previous layer
                x = torch.cat((x, self.layers[i](x)), 1)
            #returned the concatnated tensor
            return x

    def get_transform(self, nIn, nOut, bottle_neck=None, drop_rate=0):
        if not bottle_neck or nIn <= nOut * bottle_neck:
            #build a dense block 
            #first do a batch_norm, then a ReLU, then Conv2d and finally Dropout
            return nn.Sequential(
                                 nn.BatchNorm2d(nIn),
                                 nn.ReLU(True),
                                 nn.Conv2d(nIn, nOut, 3, stride=1, padding=1, bias=True),
                                 nn.Dropout(drop_rate)
        )
        else:
            nBottle = nOut * bottle_neck
            return nn.Sequential(
                                 nn.BatchNorm2d(nIn),
                                 nn.ReLU(True),
                                 nn.Conv2d(nIn, nBottle, 1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(nBottle),
                                 nn.ReLU(True),
                                 nn.Conv2d(nBottle, nOut, 3, stride=1, padding=1, bias=True),
                                 nn.Dropout(drop_rate)
        )

class FCDenseNet(nn.Module):
    
    def __init__(self, depths, growth_rates, n_scales=5, n_channel_start=48,
                 n_classes=12, drop_rate=0, bottle_neck=False):
        super(FCDenseNet, self).__init__()
        self.n_scales = n_scales
        self.n_classes = n_classes
        self.n_channel_start = n_channel_start
        self.depths = [depths] * \
            (2 * n_scales + 1) if type(depths) == int else depths
        self.growth_rates = [growth_rates] * (2 * n_scales + 1) if \
            type(growth_rates) == int else growth_rates
        self.drop_rate = drop_rate
        assert len(self.depths) == len(self.growth_rates) == 2 * n_scales + 1
        self.conv_first = nn.Conv2d(
            3, n_channel_start, 3, stride=1, padding=1, bias=True)
        #list of dense_modules    
        self.dense_blocks = nn.ModuleList([])
        #list of modules that reduce size(Conv+Pool)
        self.transition_downs = nn.ModuleList([])
        #list of modules that incrase size(DeConv)
        self.transition_ups = nn.ModuleList([])

        nskip = []
        nIn = self.n_channel_start
        
        for i in range(n_scales):
            #add n_scales DenseBlocks to the dense_blocks list
            self.dense_blocks.append(
                DenseBlock(nIn, self.growth_rates[i], self.depths[i],
                           drop_rate=drop_rate, bottle_neck=bottle_neck))
            #increment size of nIn
            nIn += self.growth_rates[i] * self.depths[i]
            #append current nIn to nskip
            nskip.append(nIn)
            #add to DeConv list
            self.transition_downs.append(self.get_TD(nIn, drop_rate))

        #add another DenseBlock
        self.dense_blocks.append(
            DenseBlock(nIn, self.growth_rates[n_scales], self.depths[n_scales],
                       only_new=True, drop_rate=drop_rate,
                       bottle_neck=bottle_neck))
        #increment nIn
        nIn = self.growth_rates[n_scales] * self.depths[n_scales]

        #
        for i in range(n_scales-1):
            #
            self.transition_ups.append(nn.ConvTranspose2d(
                nIn, nIn, 3, stride=2, padding=1, bias=True))
            #increment nIn by the last value n nskip
            nIn += nskip.pop()
            self.dense_blocks.append(
                DenseBlock(nIn, self.growth_rates[n_scales + 1 + i],
                           self.depths[n_scales + 1 + i],
                           only_new=True, drop_rate=drop_rate,
                           bottle_neck=bottle_neck))
            nIn = self.growth_rates[n_scales + 1 + i] * \
                self.depths[n_scales + 1 + i]
        # last dense block
        self.transition_ups.append(nn.ConvTranspose2d(
            nIn, nIn, 3, stride=2, padding=1, bias=True))
        nIn += nskip.pop()
        self.dense_blocks.append(
            DenseBlock(nIn, self.growth_rates[2 * n_scales],
                       self.depths[2 * n_scales], drop_rate=drop_rate,
                       bottle_neck=bottle_neck))
        nIn += self.growth_rates[2 * n_scales] * \
            self.depths[2 * n_scales]
        self.conv_last = nn.Conv2d(nIn, n_classes, 1, bias=True)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv_first(x)
        skip_connects = []
        # down sample
        for i in range(self.n_scales):
            x = self.dense_blocks[i](x)
            skip_connects.append(x)
            x = self.transition_downs[i](x)
        # bottle neck
        x = self.dense_blocks[self.n_scales](x)
        # up sample
        for i in range(self.n_scales):
            skip = skip_connects.pop()
            TU = self.transition_ups[i]
            # adjust padding
            TU.padding = (((x.size(2) - 1) * TU.stride[0] - skip.size(2)
                           + TU.kernel_size[0] + 1) // 2,
                          ((x.size(3) - 1) * TU.stride[1] - skip.size(3)
                              + TU.kernel_size[1] + 1) // 2)
            x = TU(x, output_size=skip.size())
            x = torch.cat((skip, x), 1)
            x = self.dense_blocks[self.n_scales + 1 + i](x)
        x = self.conv_last(x)
        return self.logsoftmax(x)

    def get_TD(self, nIn, drop_rate):
        layers = [nn.BatchNorm2d(nIn), nn.ReLU(
            True), nn.Conv2d(nIn, nIn, 1, bias=True)]
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

def fcdensenet_tiny(drop_rate=0):
    return FCDenseNet(2, 6, drop_rate=drop_rate)


def fcdensenet56_nodrop():
    return FCDenseNet(4, 12, drop_rate=0)


def fcdensenet56(drop_rate=0.2, n_classes=12):
    return FCDenseNet(4, 12, n_classes=n_classes, drop_rate=drop_rate)


def fcdensenet67(drop_rate=0.2):
    return FCDenseNet(5, 16, drop_rate=drop_rate)


def fcdensenet103(drop_rate=0.2, n_classes=12):
    return FCDenseNet([4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], 16, n_classes = n_classes,
                      drop_rate=drop_rate)


def fcdensenet103_nodrop(drop_rate=0, n_classes=12):
    return FCDenseNet([4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], 16, n_classes=nclasses,
                      drop_rate=drop_rate)
