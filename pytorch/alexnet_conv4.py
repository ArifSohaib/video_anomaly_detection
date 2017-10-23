from torchvision import transforms, models
import torch.nn as nn

class AlexNetConv4(nn.Module):
            def __init__(self):
                super(AlexNetConv4, self).__init__()
                original_model = models.alexnet(pretrained=True)
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(original_model.features.children())[:-3]
                )

            def forward(self, x):
                x = self.features(x)
                return x
