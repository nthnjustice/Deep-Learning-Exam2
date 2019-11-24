import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, path, width, height, transform=None):
        self.transform = transform
        self.width = width
        self.height = height
        self.images = self.get_images(path)
        self.labels = self.get_labels(path)

    def __getitem__(self, index):
        label = torch.from_numpy(self.labels[index])

        image = Image.open(self.images[index])
        image = image.resize((self.width, self.height))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_images(path):
        return np.asarray([path + file for file in os.listdir(path) if file.endswith('.png')])

    @staticmethod
    def get_labels(path):
        labels = []
        for i in [file for file in os.listdir(path) if file.endswith('.txt')]:
            targets = []
            with open(path + i) as f:
                for _, line in enumerate(f):
                    targets.append(line.strip())
            labels.append(targets)

        encoder = {
            'red blood cell': 0,
            'difficult': 1,
            'gametocyte': 2,
            'trophozoite': 3,
            'ring': 4,
            'schizont': 5,
            'leukocyte': 6
        }

        labels_ohe = []
        for targets in labels:
            encoded = [0] * len(encoder)
            for target in targets:
                encoded[encoder[target]] = 1
            labels_ohe.append(encoded)

        return np.asarray(labels_ohe).astype('float32')


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


def learn_something(path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transformations = transforms.Compose([transforms.ToTensor()])

    train = Dataset(path, 100, 75, transformations)
    train_loader = data.DataLoader(dataset=train, batch_size=20, shuffle=True)

    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(25):
        model.train()
        for i, (batch, batch_labels) in enumerate(train_loader):
            batch, batch_labels = batch.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                size = i * len(batch)
                total = train.__len__()
                percentage = 100 * (size / total)
                loss = loss.item()
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch + 1, size, total, percentage, loss))
        torch.save(model.state_dict(), 'model_njustice.pt')


learn_something('Data/train/')
