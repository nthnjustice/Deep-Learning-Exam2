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


def train_network(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine((-45, 45)),
        transforms.ToTensor()
    ])
    train = Dataset('Data/train/', 150, 150, transformations)
    train_loader = data.DataLoader(dataset=train, batch_size=20, shuffle=True)

    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(25):
        model.train()

        for i, (batch, batch_labels) in enumerate(train_loader):
            batch, batch_labels = batch.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch_labels)
            loss.backward()
            optimizer.step()

            if i % 1 == 0:
                size = i * len(batch)
                total = len(train)
                pct = size / total * 100
                loss = loss.item()
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch + 1, size, total, pct, loss))

    torch.save(model.state_dict(), model_path)


def test_network(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test = Dataset('Data/test/', 150, 150, transforms.Compose([transforms.ToTensor()]))
    test_loader = data.DataLoader(dataset=test, batch_size=len(test), shuffle=False)

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i, (batch, batch_labels) in enumerate(test_loader):
        batch, batch_labels = batch.to(device), batch_labels.to(device)

        logits = model(batch)
        sigmoids = torch.sigmoid(logits)
        y = (sigmoids > 0.5).float()

        pct = torch.sum(torch.eq(y, batch_labels)).item() / y.nelement() * 100
        loss = nn.BCELoss()
        loss = loss(y, batch_labels)
        print('Accuracy: ({:.3f}%)\tLoss: {:.6f}'.format(pct, loss.item()))


train_network('model_njustice.pt')
test_network('model_njustice.pt')
