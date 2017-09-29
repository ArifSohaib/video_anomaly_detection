"""
checks if the FCNModel is working
"""

import denseFCN
from video_dataloader import H5VideoDataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

def show_image(image):
    """ show an image"""
    plt.imshow(image)

vid_dataset = H5VideoDataset("../data/filtered_smaller_period1.h5")
vid_loader = DataLoader(vid_dataset, batch_size=5)
model = denseFCN.fcdensenet56(0.2, n_classes=3).cuda()
for data in vid_loader:
    inputs = data
    break

outputs = model(Variable(inputs.transpose(1,3).cuda()))
print(outputs.data)

plt.imshow(outputs.data[0][0].cpu().numpy())
plt.show()
img = []
for i in range(len(outputs.data[1])):
    sample = outputs.data[1][i].cpu().numpy()
    img.append(sample)
    # print(i, sample['image'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title("Sample #{}".format(i))
    ax.axis('off')
    show_image(sample)
    if i==3:
        plt.show()
        break
plt.show()
image = cv2.merge(img)
plt.imshow(image)
plt.show()
