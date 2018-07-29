import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
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
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
                
                nn.Conv2d(n_features, n_features, 3, 1, 1),
                nn.BatchNorm2d(n_features),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(n_features, n_features, 3, 1, 1),
                nn.BatchNorm2d(n_features),
                nn.ReLU(inplace=True) )
    
    def forward(self, x):
        return self.conv_block(x) + x


class Resnet(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, n_blocks=4):
        super(Resnet, self).__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks
        
        # input size: in_ch x 32 x 32
        
        ## Features ##
        n_features = 64
        
        # 1st conv:
        features = [ nn.Conv2d(in_ch, n_features, 4, 2, 1),
                  nn.BatchNorm2d(n_features),
                  nn.ReLU(inplace=True) ]
        
        for i in range(self.n_blocks):
            features += [ResidualBlock(n_features)]
        
        features += [ nn.Conv2d(n_features, out_ch, 4, 2, 1),
                   nn.ReLU(inplace=True) ]
        
        self.features = nn.Sequential(*features)
        # state size: out_ch x 8 x 8
        
        ## Classifier ##
        classifier = [ nn.Linear(self.out_ch * 8 * 8, 128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(inplace=True),
                      
                      nn.Linear(128, 64),
                      nn.BatchNorm1d(64),
                      nn.ReLU(inplace=True),
                      
                      nn.Linear(64, 10) ]
        
        self.classifier = nn.Sequential(*classifier)
        
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.out_ch * 8 * 8 )
        x = self.classifier(x)
        
        return x
   
    
#=========================================>
#   < Training the Network >
#=========================================>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
     
net = Resnet().to(device)
net.apply(weights_init)

net.load_state_dict(torch.load('./mini_resnet_trained.pth'))
net.eval()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

for epoch in range(5):  

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
                  
    print('epoch : %d , loss = %.3f' % (epoch+1, running_loss / 6250))

torch.save(net.state_dict(), './mini_resnet_trained.pth')

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

