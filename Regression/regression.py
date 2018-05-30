'''
You can find more PyTorch based tutoials here=> https://github.com/AyushExel/DeepTorch

This file contains easy to understand and well documented code for implemeting regression using
Pytorch. There are two methods mentioned, one involves building layer architecture manually 
while the other uses Sequential() function to ease the task
'''

import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt


x = torch.randn(200,1) #Generate random data
y = x.pow(4) + 0.05*torch.rand(x.size()) #add some noise to the output
epoch = 1500
learning_rate = 0.05

#Manual Method. Involves building the layer architecture:

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        '''
        Intialize 2 layer neural net
        '''
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_input,n_hidden)
        self.output = torch.nn.Linear(n_hidden,n_output)
    
    def forward(self,input):
        '''
        input=> The input to compute forward prop on
        returns => output after forward prop
        '''
        hidden = functional.relu(self.hidden(input))
        output = self.output(hidden)
        return output

net = Net(1,15,1)
print(net)

#Antoher method using Sequential function. Much less work
seq_net = torch.nn.Sequential(torch.nn.Linear(1,15),
                              torch.nn.ReLU(),
                              torch.nn.Linear(15,1)
                             )

print(seq_net) #This should be same as net

lossFunc = torch.nn.MSELoss()  #Mean Squared Error Function
optim = torch.optim.SGD(seq_net.parameters(),lr=0.0611,momentum=0.9) #SGD optimizer with momentum

for i in range(epoch):
    '''
    To cange the network used(net or seq_net), you just need to replace one network with another
    '''
    output = seq_net(x) #forward Prop
    loss = lossFunc(output,y)
    optim.zero_grad()
    loss.backward()
    #Manual Updation of parameters
    '''
    for f in seq_net.parameters():
        f.data.sub_(f.grad.data*learning_rate)
    '''
    optim.step()
    if (i+1)%4==0 :
        print(str((i+1))+'th loss',loss.squeeze())
    if i % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
