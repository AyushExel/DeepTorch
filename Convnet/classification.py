import pdb
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

#Build our Convolutional neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.convol1 = torch.nn.Conv2d(3,20,5)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.convol2 = torch.nn.Conv2d(20,30,5)
        self.fc1 = torch.nn.Linear(30*5*5,300)
        self.fc2 = torch.nn.Linear(300,75)
        self.fc3 = torch.nn.Linear(75,10)
    
    def forward(self,x):
        x = self.pool(F.relu(self.convol1(x)))
        x = self.pool(F.relu(self.convol2(x)))
        x = x.view(-1,30*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


#To convert data from PIL to tensor
transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
    )

#Load train and test set:
train = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainset = torch.utils.data.DataLoader(train,batch_size=4,shuffle=True)

test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testset = torch.utils.data.DataLoader(test,batch_size=4,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
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

net = Net()
#Print network parameter details for deeper understanding:
for parameter in net.parameters():
    print(str(parameter.data.numpy().shape)+'\n')
#train the Network
costFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

acc = 0  #Store best accuracy
epoch = 0 #store epochs
#chek if there is a trained parameters present:
'''
if os.path.isdir('save'):
    saved_model = torch.load('./save/model.pt')
    net.load_state_dict(saved_model['params'])
    acc = saved_model['acc']
    epoch = saved_model['epoch']
    print('Saved model has accuracy of ' + str(acc))
''' 
def train_model(epochs):
    net.train() #set the model to training mode
    for epoch in range(epochs):
        losses = []
        num_times=0
        closs = 0
        for i,batch in enumerate(trainset,0):
            data , output = batch
            prediction = net(data)
            loss = costFunction(prediction,output)
            closs += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Track every 100th loss
            if i%100 == 0:
                losses.append(loss.item())
                num_times = num_times + 1

            #print every 1000th time
            if i%1000 == 0:
                print('[%d  %d] loss: %.4f'% (epoch+1,i+1,closs/1000))
                closs = 0
        #Calculate the accuracy and save the model state
        accuracy()
        #Plot the graph of loss with iteration
        plt.plot([i for i in range(num_times)],losses,label='epoch'+str(epoch))
        plt.legend(loc=1,mode='expanded',shadow=True,ncol=2)
    plt.show()

def accuracy():
    net.eval() #set the model to evaluation mode
    #Calculate the overall performance of the network
    correctHits=0
    total=0
    accuracy=0
    for batches in testset:
        data,output = batches
        prediction = net(data)
        _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
        total += output.size(0)
        correctHits += (prediction==output).sum().item()
        accuracy = (correctHits/total)*100
    print('Accuracy = '+str(accuracy))
    '''
    global acc
    if accuracy > acc: #Save the model if accuracy increases
        acc=accuracy
        if not os.path.isdir('save'):
            os.mkdir('save')
        torch.save({'params':net.state_dict(),
                    'epoch':epoch,
                    'acc':accuracy},'./save/model.pt')
    '''

if __name__ == '__main__':
    train_model(2)

'''
Official Documentation:
torchvision - https://pytorch.org/docs/stable/torchvision/transforms.html
torchvision.tansforms - https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-pil-image
toch.util.data - https://pytorch.org/docs/stable/data.html
np.transpose - https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
'''
