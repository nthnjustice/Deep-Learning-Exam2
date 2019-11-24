import os
os.system("sudo pip install PIL")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def predict(x):
    images = []
    for img_path in x:
        image = Image.open(img_path)
        #image = np.array(image.resize((100, 75)))
        images.append(image)
    x = torch.FloatTensor(np.array(images))
    print(x.size())
    x = x.reshape([len(x), 3, 75, 100])

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 35, (3, 3), stride=1, padding=1)
            self.convnorm1 = nn.BatchNorm2d(35)
            self.pool1 = nn.MaxPool2d((2, 2), stride=2)
            self.conv2 = nn.Conv2d(35, 70, (3, 3), stride=1, padding=1)
            self.convnorm2 = nn.BatchNorm2d(70)
            self.pool2 = nn.AvgPool2d((2, 2), stride=2)
            self.linear1 = nn.Linear(31500, 200)
            self.linear1_bn = nn.BatchNorm1d(200)
            self.drop = nn.Dropout(0.2)
            self.linear2 = nn.Linear(200, 7)
            self.act = torch.relu

        def forward(self, x):
            x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
            x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
            x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
            return self.linear2(x)

    model = CNN()
    model.load_state_dict(torch.load('model_njustice.pt'))
    logits = model(x)
    sigmoids = torch.sigmoid(logits)
    y_pred = (sigmoids > 0.5).float()
    return y_pred


# # %% -------------------------------------------------------------------------------------------------------------------
# path = 'Data/sample_test/'
# x_test = [path + file for file in os.listdir(path) if file.endswith('.png')]
# y_test_pred = predict(x_test)
#
# # %% -------------------------------------------------------------------------------------------------------------------
# assert isinstance(y_test_pred, type(torch.Tensor([1])))  # Checks if your returned y_test_pred is a Torch Tensor
# assert y_test_pred.dtype == torch.float  # Checks if your tensor is of type float
# assert y_test_pred.device.type == "cpu"  # Checks if your tensor is on CPU
# assert y_test_pred.requires_grad is False  # Checks if your tensor is detached from the graph
# assert y_test_pred.shape == (len(x_test), 7)  # Checks if its shape is the right one
# # Checks whether the your predicted labels are one-hot-encoded
# assert set(list(np.unique(y_test_pred))) in [{0}, {1}, {0, 1}]
# print("All tests passed!")
