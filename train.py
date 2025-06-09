from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pprint
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
sweep_id = wandb.sweep(sweep_config, project='new-test-run-2')


pprint.pprint(wandb.config)

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


# loader = {
#     'train': DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True, num_workers=1),
#     'test': DataLoader(test_data, batch_size=wandb.config.batch_size, shuffle=True, num_workers=1)
# }


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cov1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
      x = F.relu(F.max_pool2d(self.cov1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      x = x.view(-1, 320)
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)

      return F.softmax(x, dim=1)
    
def build_dataset(batch_size):
    # Training data
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Test data
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return {'train': train_loader, 'test': test_loader}



def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(epoch, loader, optimizer):
  cmu = 0
  model.train()
  for batch_idx, (data, target) in enumerate(loader['train']):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    cmu += loss.item()

    loss.backward()
    optimizer.step()
    
    if(batch_idx%2000 == 0):
      print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader["train"].dataset)} ({100. * batch_idx / len(loader["train"])}%)]\t{loss.item():.6f}')
    
    wandb.log({"batch loss": loss.item()})

  return cmu/len(loader['train']) 
      

def test(loader):
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
  wandb.log({"loss": test_loss, "accuracy": 100. * correct / len(loader['test'].dataset)})
  

def train(config = None):
  with wandb.init(config=config):
    config = wandb.config
    pprint.pprint(config)

    loader = build_dataset(config.batch_size)
        # network = build_network(config.fc_layer_size, config.dropout)
    optimizer = build_optimizer( model, config.optimizer, config.learning_rate)
    for epoch in range(wandb.config.epochs):
      avg_loss = train_epoch(epoch, loader, optimizer)
      wandb.log({'loss': avg_loss, "epoch": epoch})

    test(loader)


wandb.agent(sweep_id, function=train)  # Adjust count as needed