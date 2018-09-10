import pdb
'''

'''

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

#To convert data from PIL to tensor
transform = transforms.Compose(
    [transforms.ToTensor()]
    )

#Load train and test set:
train = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainset = torch.utils.data.DataLoader(train,batch_size=4,shuffle=True)

test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testset = torch.utils.data.DataLoader(test,batch_size=4,shuffle=False)
'''
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Display a random batch
iterable_batch = iter(trainset)
random_batch,labels = iterable_batch.next()
grid = torchvision.utils.make_grid(random_batch)
grid_matrix = grid.numpy()
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
print(labels)
plt.imshow(grid_matrix.transpose(1,2,0))
plt.show()
'''
#Build our Convolutional neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.convol1 = torch.nn.Conv2d(3,10,5)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.convol2 = torch.nn.Conv2d(10,20,5)
        self.fc1 = torch.nn.Linear(20*5*5,200)
        self.fc2 = torch.nn.Linear(200,75)
        self.fc3 = torch.nn.Linear(75,10)
    
    def forward(self,x):
        x = self.pool(F.relu(self.convol1(x)))
        x = self.pool(F.relu(self.convol2(x)))
        x = x.view(-1,20*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
net = Net()
#train the Network
costFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(2):
    closs = 0
    for i,batch in enumerate(trainset,0):
        data , output = batch
        prediction = net(data)
        loss = costFunction(prediction,output)
        closs += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print every 1000th time
        if i%1000 == 0:
            print('[%d  %d] loss: %.4f'% (epoch+1,i+1,closs/1000))
            closs = 0

#Calculate the overall performance of the network

correctHits=0
total=0
for batches in testset:
    data,output = batches
    prediction = net(data)
    _,prediction = torch.max(prediction.data,1)
    total += output.size(0)
    correctHits += (prediction==output).sum().item()
pdb.set_trace()
print('Accuracy = '+str((correctHits/total)*100))



'''
Official Documentation:
torchvision - https://pytorch.org/docs/stable/torchvision/transforms.html
toch.util.data - https://pytorch.org/docs/stable/data.html
np.transpose - https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
'''