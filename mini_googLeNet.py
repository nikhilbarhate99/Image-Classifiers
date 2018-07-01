import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#=========================================>
#   < Preprocessing the Images >
#=========================================>

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#=========================================>
#   < Building the Neural Network >
#=========================================>

class Net(nn.Module):
    
    def __init__(self): 
        super(Net, self).__init__()
        
        #inception module 1:
        self.conv11 = nn.Conv2d(3, 4, 1)
        self.conv13 = nn.Conv2d(3, 4, 3, padding = 1)
        self.conv15 = nn.Conv2d(3, 4, 5, padding = 2)
        
        #inception module 2:
        self.conv21 = nn.Conv2d(12, 16, 1)
        self.conv23 = nn.Conv2d(12, 16, 3, padding = 1)
        self.conv25 = nn.Conv2d(12, 16, 5, padding = 2)
        
        #inception module 3:
        self.conv31 = nn.Conv2d(48, 64, 1)
        self.conv33 = nn.Conv2d(48, 64, 3, padding = 1)
        self.conv35 = nn.Conv2d(48, 64, 5, padding = 2)
        
        #fully connected layer :
        self.fc1 = nn.Linear(192 * 32 * 32, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        

    def forward(self, x):       
        
        #inception 1:
        out1 = F.relu(self.conv11(x))
        out2 = F.relu(self.conv13(x))
        out3 = F.relu(self.conv15(x))
        
        out = torch.cat([out1, out2, out3], 1)
        
        #inception 2:
        out1 = F.relu(self.conv21(out))
        out2 = F.relu(self.conv23(out))
        out3 = F.relu(self.conv25(out))
        
        out = torch.cat([out1, out2, out3], 1)
        
        #inception 3:
        out1 = F.relu(self.conv31(out))
        out2 = F.relu(self.conv33(out))
        out3 = F.relu(self.conv35(out))
        
        out = torch.cat([out1, out2, out3], 1)
        
        #flatten and FC :
        out = out.view(4, 192 * 32 * 32)
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
    

#=========================================>
#   < Training the Network >
#=========================================>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
net = Net()
net = net.to(device)

print(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

for epoch in range(10):  

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
       
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = outputs.to(device)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('\n < Finished Training > \n')


#=========================================>
#   < Testing the Netwok >
#=========================================>

dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

outputs = net(images)

_, predicted = torch.max(outputs, 1)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy : %d %%' % (
    100 * correct / total))


