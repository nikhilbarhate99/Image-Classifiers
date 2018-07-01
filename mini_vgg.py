import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#=========================================>
#   < Preprocessing the Images >
#=========================================>

transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=2)


#=========================================>
#   < Building the Network >
#=========================================>

class VGG_mini(nn.Module):
    
    def __init__(self): 
        super(VGG_mini, self).__init__()
        
        # Maxpool 2x2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv layers with batch norm
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.norm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(64)
               
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.norm4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.norm5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.norm6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3, padding = 1)
        self.norm7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 512, 3, padding = 1)
        self.norm8 = nn.BatchNorm2d(512)
        
        # fully connected layer with batch norm

        self.fc1 = nn.Linear(512 * 4 * 4, 128)
        self.norm9 = nn.BatchNorm1d(128)
       
        self.fc2 = nn.Linear(128, 64)
        self.norm10 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 10)
        

    def forward(self, x):       
        
        out = F.elu(self.norm1(self.conv1(x)))
        out = F.elu(self.norm2(self.conv2(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm3(self.conv3(out)))
        out = F.elu(self.norm4(self.conv4(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm5(self.conv5(out)))
        out = F.elu(self.norm6(self.conv6(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm7(self.conv7(out)))
        out = F.elu(self.norm8(self.conv8(out)))
        
        out = out.view(-1, 512 * 4 * 4)
        
        out = F.elu(self.norm9(self.fc1(out)))
        out = F.elu(self.norm10(self.fc2(out)))
        out = self.fc3(out)

        return out
    
    
#=========================================>
#   < Training the Network >
#=========================================>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
net = VGG_mini().to(device)
print(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

for epoch in range(15):  

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
       
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = net(inputs)        

        loss = loss_fn(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        
        if epoch + 1 == 10:
            torch.save(VGG_mini, './mini_vgg_trained_epochs_10')
            
    print('epoch : %d , loss = %.3f' % (epoch+1, running_loss / 6250))

torch.save(VGG_mini, './mini_vgg_trained')

print('\n < Finished Training > \n')


#=========================================>
#   < Testing the Netwok >
#=========================================>

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

