import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, path, width, height, transformations=None):
        self.width = width
        self.height = height
        self.transformations = transformations
        self.images = self.get_images(path)
        self.labels = self.get_labels(path)

    def __getitem__(self, index):
        label = torch.from_numpy(self.labels[index])

        image = Image.open(self.images[index])
        image = image.resize((self.width, self.height))

        if self.transformations is not None:
            image = self.transformations(image)

        return image, label

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_images(path):
        return np.asarray([path + file for file in os.listdir(path) if file.endswith('.png')])

    @staticmethod
    def get_labels(path):
        labels = []
        for name in [file for file in os.listdir(path) if file.endswith('.txt')]:
            targets = []
            with open(path + name) as f:
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
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(43808, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.2),
            nn.Linear(200, 7)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x.view(len(x), -1))
        return x


def learn_something():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    train = Dataset('Data/train/', 150, 150, transformations)
    train_loader = data.DataLoader(dataset=train, batch_size=10, shuffle=True)

    test = Dataset('Data/test/', 150, 150, transforms.Compose([transforms.ToTensor()]))
    test_loader = data.DataLoader(dataset=test, batch_size=len(test), shuffle=False)

    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(5):
        model.train()
        for i, (batch, batch_labels) in enumerate(train_loader):
            batch, batch_labels = batch.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch_labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                curr_epoch = epoch + 1
                size = i * len(batch)
                total = len(train)
                percentage = size / total * 100
                loss = loss.item()
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(curr_epoch, size, total, percentage, loss))

    model.eval()
    with torch.no_grad():
        for i, (batch, batch_labels) in enumerate(test_loader):
            batch, batch_labels = batch.to(device), batch_labels.to(device)

            pred = (torch.sigmoid(model(batch)) > 0.5).float()
            percentage = torch.sum(torch.eq(pred, batch_labels)).item() / pred.nelement()
            loss = loss_fn(pred, batch_labels).item()
            print('Accuracy: ({:.0f}%)\tLoss: {:.6f}'.format(percentage, loss))

    torch.save(model.state_dict(), 'model_njustice.pt')


learn_something()
# Epoch: 5 [900/929 (97%)]	Loss: 0.494119
# Accuracy: (1%)	Loss: 0.643302