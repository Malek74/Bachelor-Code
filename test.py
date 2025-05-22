import torch
import os
from torch.utils.data import Dataset
from torchvision import datasets
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb

wandb.init(project="mnist-classification", entity="malek-mohamed12345678-german-university-in-cairo")
#this file is to write config file for biases and weights
hypeparameters={
    "learning_rate":{
        "values":[0.01,0.001,0.0001],
        },
    "epochs":{
        "values":[1,5,10],
    },"batch_size":{
        "value":8,
    },
}


config={
    "method":"grid",
    "metric":{
        "name":"loss",
        "goal":"minimize",
    }
}

config['parameters'] = hypeparameters
training_data = datasets.MNIST(
    root="data", #location of data
    train=True, #specify is it train or test data
    download=True, #download data if not found in root
    transform=ToTensor() #transform data to tensor
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

classes={
    0:"Zero",
    1:"One",
    2:"Two",
    3:"Three",
    4:"Four",
    5:"Five",
    6:"Six",
    7:"Seven",
    8:"Eight",
    9:"Nine",
}

batch_size=8

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)


figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    #load from training data
    if(i>4):
      sample_idx = torch.randint(len(training_data), size=(1,)).item()
      img, label = training_data[sample_idx]
    #load from test data
    else:
      sample_idx = torch.randint(len(test_data), size=(1,)).item()
      img, label = test_data[sample_idx]

    figure.add_subplot(rows, cols, i)
    plt.title(classes[label])
    plt.axis("off")
    plt.imshow(img.squeeze())
plt.show()


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")

class MNISTNeuralNetwork(nn.Module):
  def __init__(self):
      super().__init__()
      self.flatten = nn.Flatten()
      self.linear_relu_stack = nn.Sequential(
          nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

  def forward(self,x):
    x=self.flatten(x)
    logits=self.linear_relu_stack(x)
    return logits

net=MNISTNeuralNetwork()
net = net.to(device)
print(net)

def create_model():
    model = MNISTNeuralNetwork()  # replace with your model class
    return model.to(device)

def train():
    with wandb.init() as run:
        config = wandb.config
        net = create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

        net.train()
        for epoch in range(config.epochs):
            running_loss = 0.0
            for i, (data, label) in enumerate(train_loader):
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                outputs = net(data)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Log every 100 batches
                if i % 100 == 99:
                    wandb.log({
                        'epoch': epoch + 1,
                        'step': epoch * len(train_loader) + i,
                        'loss': running_loss / 100
                    })
                    running_loss = 0.0

#initialize wandb
sweep_id = wandb.sweep(config, project="MNIST-Sweep")

#start the sweep using the sweep agent
wandb.agent(sweep_id, function=train)

correct=0
total=0
confidence=[]

with torch.no_grad():
  for data,labels in test_loader:
    data=data.to(device)
    labels=labels.to(device)

    output=net(data)
    conf=F.softmax(output,dim=1)
    conf=torch.max(conf,1)[0]
    confidence.extend(conf.cpu().numpy())
    _,predicted_class=torch.max(output,1)
    total+=labels.size(0)
    correct+=(predicted_class==labels).sum().item()


plt.hist(confidence, bins=50)
plt.xlabel('Max Softmax Confidence')
plt.ylabel('Number of Samples')
plt.title('Confidence Distribution')
plt.show()

print(f"Accuracy on test data: {100*correct/total}%")