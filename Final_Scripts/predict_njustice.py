import numpy as np
import torch
import torch.nn as nn
import os
os.system('sudo pip install PIL')
from PIL import Image


def predict(x):
    images = []
    for img_path in x:
        image = Image.open(img_path)
        image = np.array(image.resize((150, 150)))
        images.append(image)

    x = torch.FloatTensor(np.array(images))
    x = x.reshape([len(x), 3, 150, 150])

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.2),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.fc = nn.Sequential(
                nn.Linear(41472, 500),
                nn.ReLU(),
                nn.BatchNorm1d(500),
                nn.Dropout(0.5),
                nn.Linear(500, 7)
            )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.fc(x.view(len(x), -1))

            return x

    model = CNN()
    model.load_state_dict(torch.load('model_njustice.pt'))

    logits = model(x)
    sigmoids = torch.sigmoid(logits)
    y = (sigmoids > 0.5).float()

    return y


# path = 'Data/test/'
# test = [path + file for file in os.listdir(path) if file.endswith('.png')]
# pred = predict(test)
#
# assert isinstance(pred, type(torch.Tensor([1])))
# assert pred.dtype == torch.float
# assert pred.device.type == 'cpu'
# assert pred.requires_grad is False
# assert pred.shape == (len(test), 7)
# assert set(list(np.unique(pred))) in [{0}, {1}, {0, 1}]
# print('All tests passed!')
