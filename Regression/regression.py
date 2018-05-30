import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt


x = torch.randn(200,1) #Generate random data
y = x.pow(4) + 0.05*torch.rand(x.size()) #add some noise to the output
epoch = 150  
learning_rate = 0.05

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

lossFunc = torch.nn.MSELoss()  #Mean Squared Error Function
optim = torch.optim.SGD(net.parameters(),lr=0.05,momentum=0.9) #SGD optimizer with momentum

for i in range(epoch):
    output = net(x) #forward Prop
    loss = lossFunc(output,y)
    optim.zero_grad()
    loss.backward()
    #Manual Updation of parameters
    '''
    for f in net.parameters():
        f.data.sub_(f.grad.data*learning_rate)
    '''
    optim.step()
    if (i+1)%4==0 :
        print(loss.squeeze() ,'  ')
    if i % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()