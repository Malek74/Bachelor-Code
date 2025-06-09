from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb

sweep_config = {
    'method': 'grid',
    'metric': { 
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [64, 128]
        },
        'learning_rate': {
            'values': [0.001, 0.1]
        },
        'optimizer': {
            'values': ['sgd']
        },
        'epochs': {
            'value': 5
        }
    }
}
# wandb.config.update(sweep_config)
sweep_id = wandb.sweep(sweep_config, project='new-test-run-3')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

with wandb.init(config=sweep_config):
    config = wandb.config


    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    from torch.utils.data import DataLoader

    loader = {
        'train': DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True, num_workers=1),
        'test': DataLoader(test_data, batch_size=wandb.config.batch_size, shuffle=True, num_workers=1)
    }



    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = CNN().to(device)
    if wandb.config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    elif wandb.config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9)
    elif wandb.config.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=wandb.config.learning_rate, alpha=0.9)
    loss_fn = nn.CrossEntropyLoss()

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(loader['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if(batch_idx%2000 == 0):
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader['train'].dataset)} ({100. * batch_idx / len(loader['train'])}%)]\t{loss.item():.6f}')

    def test():
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in loader['test']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader['test'])
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader['test'].dataset),
            100. * correct / len(loader['test'].dataset)))
        test_loss += loss_fn(output, target).item()
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    # # test_loss /= len(loader['test'])
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(loader['test'].dataset),
    #     100. * correct / len(loader['test'].dataset)))

    for epoch in range(wandb.config.epochs):
    #   wandb.log({"loss": test_loss, "epoch": epoch})
        train(epoch)
    test()