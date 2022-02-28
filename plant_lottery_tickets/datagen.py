import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

import math

import torch
from torch.utils.data import Dataset, DataLoader

def relu(x):
    return np.clip(x, a_min=0, a_max=None)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    y = np.exp(x)
    return y/np.sum(y)

def draw_data_relu(nsamples, d_in, problemType):
    #assume 1-dim output
    dat = np.zeros((nsamples,d_in+1))
    if problemType == 'reg':
        for i in range(nsamples):
            dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
            dat[i,-1] = relu(dat[i,0])
    else:
        for i in range(nsamples):
            dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
            y = dat[i,0]
            if y>0:
                dat[i,-1] = 1
            else:
                dat[i,-1] = 0
    return dat

def draw_data_relu_with_noise(nsamples, d_in, problemType, sigma):
    #assume 1-dim output, Gaussian noise with standard deviation sigma
    dat = np.zeros((nsamples,d_in+1))
    if problemType == 'reg':
        for i in range(nsamples):
            noise = np.random.normal(loc=0.0, scale=sigma, size=1)
            dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
            dat[i,-1] = relu(dat[i,0]) + noise
    else:
        for i in range(nsamples):
            noise = np.random.normal(loc=0.0, scale=sigma, size=1)
            dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
            y = dat[i,0]+noise
            if y>0:
                dat[i,-1] = 1
            else:
                dat[i,-1] = 0
    return dat


#plotting

def plot_relu(dat, problemType):
    ind = np.argsort(dat[:,0])
    if problemType == 'reg':
        plt.plot(dat[ind,0], dat[ind,-1], 'x', markersize=4, markeredgewidth=1)
        plt.show()
    else:
        for i in range(dat.shape[0]):
            cc = "red"
            mark = 's'
            if dat[i,-1] == 1:
                cc = "black"
                mark = 'o'
            plt.plot(dat[i,0], dat[i,1], mark, color=cc, markersize=4, markeredgewidth=1)#col[np.asarray(dat[:,2],dtype=int)])
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.1, 1.1))
        plt.show()

def plot_relu_with_ticket(dat, ticket, problemType):
    ind = np.argsort(dat[:,0])
    if problemType == 'reg':
        xx = np.linspace(-1,1,1000)
        plt.plot(xx, ticket(xx),color="black")
        plt.plot(dat[ind,0], dat[ind,-1], 'x', markersize=4, markeredgewidth=1, color="blue")
        plt.show()
    else:
        xx = np.linspace(-1.1,1.1,10000)
        yy = ticket(np.vstack([xx,xx]).T)
        xdec = np.where(yy>0)[0][0]
        if xdec == 0:
            xdec = -1
        plt.plot(xx[xdec]*np.ones(10000), xx, color="black")
        for i in range(dat.shape[0]):
            cc = "red"
            mark = 's'
            if dat[i,-1] == 1:
                cc = "blue"
                mark = 'o'
            plt.plot(dat[i,0], dat[i,1], mark, color=cc, markersize=4, markeredgewidth=1)#col[np.asarray(dat[:,2],dtype=int)])
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.1, 1.1))
        plt.show()

#Sphere
def h(x, ind):
    #inner functions
    return x**2

def f(x):
    #outer function
    #decision boundary: 0.2, 0.5, 0.7
    scale = 50
    y = np.exp(scale*np.array([0.1*x, (0.2*x-0.02), (0.4*x+((0.2-0.4)*0.5-0.02)), (0.8*x+((0.4-0.8)*0.7+(0.2-0.4)*0.5-0.02))]))
    return y/np.sum(y)

def freg(x):
    #outer function
    return x

def target(x, f, h, d_in):
    y = 0
    for i in range(d_in):
        y = y + h(x[i],i)
    return f(y)

# def draw_data_sphere(nsamples, f, h, d_in):
#     dat = np.zeros((nsamples,d_in+1), dtype=np.float32)
#     for i in range(nsamples):
#         dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
#         y = target(dat[i,:d_in], f, h, d_in)
#         dat[i,-1] = np.where(y==max(y))[0][0]
#     return dat

#def draw_data_sphere(nsamples, f, h, d_in, p_noise):
def draw_data_sphere(nsamples, p_noise):
    d_in = 2
    dat = np.zeros((nsamples,d_in+1), dtype=np.float32)
    for i in range(nsamples):
        dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
        y = target(dat[i,:d_in], f, h, d_in)
        dat[i,-1] = np.where(y==max(y))[0][0]
        pp = np.random.uniform(low=0.0, high=1.0, size=1)
        if pp <= p_noise:
            dat[i,-1] = np.where(y==np.sort(y)[-2])[0][0]
    return dat

def draw_data_sphere_general(nsamples, p_noise, type):
    d_in = 2
    dat = np.zeros((nsamples,d_in+1), dtype=np.float32)
    if type == "reg":
        for i in range(nsamples):
            dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
            y = target(dat[i,:d_in], freg, h, d_in)
            dat[i,-1] = y + np.random.normal(0, p_noise, size=1)
    else:
        for i in range(nsamples):
            dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
            y = target(dat[i,:d_in], f, h, d_in)
            dat[i,-1] = np.where(y==max(y))[0][0]
            pp = np.random.uniform(low=0.0, high=1.0, size=1)
            if pp <= p_noise:
                dat[i,-1] = np.where(y==np.sort(y)[-2])[0][0]
    return dat

def plot_data_sphere(dat):
    cmap = cm.get_cmap(name='rainbow')
    for i in range(dat.shape[0]):
        plt.plot(dat[i,0], dat[i,1], 'x', color=cmap(dat[i,2]/3), markersize=4, markeredgewidth=1)#col[np.asarray(dat[:,2],dtype=int)])
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.show()

#Helix
def fun_ex1(x):
    y = 5*np.pi + 3*np.pi*x
    return y*np.cos(y)/(8*np.pi)

def fun_ex2(x):
    y = 5*np.pi + 3*np.pi*x
    return y*np.sin(y)/(8*np.pi)

def fun_ex3(x):
    return (5*np.pi + 3*np.pi*x)/(8*np.pi)

def draw_data_helix(nsamples, d_in, sigma):
    dat = np.zeros((nsamples,d_in+3))
    for i in range(nsamples):
        noise = np.random.normal(loc=0.0, scale=sigma, size=3)
        dat[i,:d_in] = np.random.uniform(low=-1, high=1.0, size=d_in)
        dat[i,d_in] = fun_ex1(dat[i,0]) + noise[0]
        dat[i,d_in+1] = fun_ex2(dat[i,0]) + noise[1]
        dat[i,d_in+2] = fun_ex3(dat[i,0]) + noise[2]
    return dat

def plot_data_helix(dat, ticket):
    ax = plt.axes(projection='3d')
    xx = np.linspace(-1,1,1000)
    out = ticket(xx)
    xline = fun_ex1(xx)
    yline = fun_ex2(xx)
    zline = fun_ex3(xx)
    ind0 = np.array([np.sum(np.abs(xline-out[:,0])), np.sum(np.abs(xline-out[:,1])), np.sum(np.abs(xline-out[:,2]))]).argmin()
    ind1 = np.array([np.sum(np.abs(yline-out[:,0])), np.sum(np.abs(yline-out[:,1])), np.sum(np.abs(yline-out[:,2]))]).argmin()
    ind2 = np.array([np.sum(np.abs(zline-out[:,0])), np.sum(np.abs(zline-out[:,1])), np.sum(np.abs(zline-out[:,2]))]).argmin()
    ax.plot3D(out[:,ind0], out[:,ind1], out[:,ind2], 'black')#'gray')
    ax.scatter3D(dat[:,-3], dat[:,-2], dat[:,-1], color="blue")#c=dat[:,-1], cmap='viridis');#cmap='Greens');
    plt.show()







class TicketDataset(Dataset):

    def __init__(self, data, d_in, train_proportion, is_training):

        np.savetxt('test.txt', data, fmt='%1.7f')
        if is_training:
            ran = np.arange(0,math.ceil(train_proportion*data.shape[0]))
        else:
            ran = np.arange(math.ceil(train_proportion*data.shape[0]),data.shape[0])
        m = data.shape[1] - d_in
        self.data_feats = torch.Tensor(data[ran[:,np.newaxis],np.arange(d_in)[np.newaxis,:]])
        self.data_resp = torch.Tensor(data[ran[:,np.newaxis],np.arange(d_in,d_in + m)[np.newaxis,:]])
        self.d_in = d_in


    def __len__(self):
        return self.data_feats.shape[0]


    def __getitem__(self, index):
        return self.data_feats[index,:], self.data_resp[index]

def gen_data_circle_syn(n = 10000, is_train = True, split = .9, noise = 0.01):

    ## draw circle data
    data = draw_data_sphere_general(n, noise, 'class')

    ## shuffle with fixed seed for same effect across datasets of synflow
    perm = np.random.RandomState(seed=42).permutation(n)
    data = data[perm,]

    d_in = 2
    dataset = TicketDataset(data, d_in, split, is_train)
    return dataset


def gen_data_relu_syn(n = 10000, is_train = True, split = .9, noise = 0.01):

    ## draw relu data
    data = draw_data_relu_with_noise(n, 1, 'reg', noise)

    ## shuffle with fixed seed for same effect across datasets of synflow
    perm = np.random.RandomState(seed=42).permutation(n)
    data = data[perm,]

    d_in = 1
    dataset = TicketDataset(data, d_in, split, is_train)
    return dataset


def gen_data_helix_syn(n = 10000, is_train = True, split = .9, noise = 0.01):

    ## draw helix data
    data = draw_data_helix(n, 1, noise)

    ## shuffle with fixed seed for same effect across datasets of synflow
    perm = np.random.RandomState(seed=42).permutation(n)
    data = data[perm,]

    d_in = 1
    dataset = TicketDataset(data, d_in, split, is_train)
    return dataset
