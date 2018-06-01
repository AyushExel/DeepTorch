import sklearn.datasets as datasets
import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt 

'''
Make fake classification data for us to train upon
#Samples = 200
#features = 10
#classes = 2
'''
data = datasets.make_classification(n_samples=200,n_features=10,n_classes=2)
X = torch.from_numpy(data[0]).float()
Y = torch.from_numpy(data[1]).long()
epoch = 1500
print(X.shape,'\n',Y.shape)
'''
Let's build two neural nets.First, by implementing the entire network architecture manually and
then by using the inbuilt Sequential() function 
'''
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_classes):
        '''
        Intialize 2 layer neural net
        '''        
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.output = torch.nn.Linear(n_hidden,n_classes)
    
    def forward(self,input):
        '''
        input=> The input to compute forward prop on
        returns => output after forward prop
        '''   
        hidden = functional.relu(self.hidden(input))
        output = functional.sigmoid(self.output(hidden))

net = Net(10,20,2)
print(net)

#Antoher method using Sequential function. Requires much less work
seq_net = torch.nn.Sequential(torch.nn.Linear(10,20),
                              torch.nn.ReLU(),
                              torch.nn.Linear(20,2),
                              torch.nn.Sigmoid()
                             )
print(seq_net)

lossFunc = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(seq_net.parameters(),lr=0.0711,momentum=0.9)

plt.ion()

for i in range(epoch):
    '''
    To cange the network used(net or seq_net), you just need to replace one network name
     with another
    '''
    output = seq_net(X)
    loss = lossFunc(output,Y)
    #Manual Updation of parameters
    '''
    seq_net.zero_grad()
    for f in seq_net.parameters():
        f.data.sub_(f.grad.data*learning_rate)
    '''

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (i+1)%4==0 :
        print(str((i+1))+'th loss',loss.squeeze())
    
    if i % 2 == 0:
        # plot the data to verify the learning process
        plt.cla()
        prediction = torch.max(output, 1)[1]
        predicted_y = prediction.data.numpy().squeeze()
        target_y = Y.data.numpy()
        plt.scatter(X.data.numpy()[:, 0], X.data.numpy()[:, 1], c=predicted_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(predicted_y == target_y)/200.
        plt.text(2, -8, 'Accuracy=%.2f' % accuracy, fontdict={'size': 33, 'color':  'black'})
        plt.pause(0.05)

plt.ioff()
plt.show()





  

