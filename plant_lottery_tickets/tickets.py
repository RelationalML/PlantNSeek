import numpy as np

import torch

def relu(x):
    return np.clip(x, a_min=0, a_max=None)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    y = np.exp(x)
    return y/np.sum(y)

def net_eval(x, weight, bias, scale):
    L = len(bias)
    for i in range(L-1):
        x = relu(x@weight[i] + bias[i])
    i = L-1
    x = x@weight[i] + bias[i]
    #x = softmax(x@weight[i] + bias[i])
    return x/scale

def net_eval_classification(x, weight, bias, scale):
    L = len(bias)
    for i in range(L-1):
        x = relu(x@weight[i] + bias[i])
        #print(x)
    i = L-1
    x = softmax((x@weight[i] + bias[i])/scale)
    #x = x@weight[i] + bias[i]
    return x

def param_outer(xs, pm, fun):
    #xs = np.sort(xs.flatten())
    xs = xs.flatten()
    ind = np.argsort(xs)
    indRev = np.arange(len(ind))
    indRev = indRev[ind]
    pm = pm.flatten()
    xs = xs[ind]
    pm = pm[ind]
    #make sure that first knot has positive sign
    pm[0] = 1
    # Layer 1
    w1 = pm[indRev]
    w1 = w1.reshape(1, -1)
    b1 = -xs*pm
    b1 = b1[indRev]
    # Layer 2
    delta = xs[1]-xs[0]
    xt = np.hstack([xs, xs[-1]+delta])
    yt = np.asarray([fun(x) for x in xt]).flatten()
    ms = (yt[1:]-yt[:-1])/(xt[1:]-xt[:-1])
    w2 = ms[1:]-ms[:-1]
    w2 = np.concatenate((np.array([ms[0]+np.sum(w2[pm[1:]==-1])]), w2))
    b2 = np.array([fun(xs[0])-np.sum(relu(xs[0]*w1+b1)*w2)])
    w2 = w2[indRev]
    w2 = w2.reshape(-1, 1)
    return (w1, w2), (b1, b2)

def rescale_output(Y, F):
    #Y data output matrix of size nxd (where n is the number of samples and d the output dimension)
    #F output matrix of deep neural network of size nxd (where n is the number of samples and d the output dimension)
    #output: rescaling factors lambda that minimizes  \sum_ij (yij - fij scale_j)**2
    #thus output F[:,j] is rescales to scale_j F[:,j] to match the output data Y[:,j]
    n, d = F.shape
    scale = np.zeros(d)
    for j in range(d):
        scale[j] = np.sum(Y[:,j]*F[:,j])/np.sum(F[:,j]**2)
    return scale

def rescale_output_torch(Y, F):
    #Y data output matrix of size nxd (where n is the number of samples and d the output dimension)
    #F output matrix of deep neural network of size nxd (where n is the number of samples and d the output dimension)
    #output: rescaling factors lambda that minimizes  \sum_ij (yij - fij scale_j)**2
    #thus output F[:,j] is rescales to scale_j F[:,j] to match the output data Y[:,j]
    scale = torch.zeros(F.size()[1])
    for j in np.arange(F.size()[1]):
        denom = torch.sum(torch.pow(F[:,j],2))
        if denom == 0:
            scale[j] = 1
        else:
            scale[j] = torch.sum(Y[:,j]*F[:,j])/denom
    return scale

def lottery_univariate_wide(fun, depth, sparsity):
    #sparsity: number of weights in lottery ticket, i.e., non-zero weights in lottery ticket
    #depth = number of layers (including input layer)
    #assume continuous function fun: [-1,1] -> [-1,1]
    #ReLU activation functions, except last layer, last layer linear
    if depth == 3:
        nknots = int(np.floor(sparsity/2))
        s = np.linspace(-1,1,nknots)
        pm = np.ones(nknots)
        pm[np.arange(1,nknots,2)] = -1
        weightTarget, biasTarget = param_outer(s, pm, fun)
        #lottery ticket of form relu(1+x) then wide univariate function (2 layers) + bias 1 followed by relu()
    if depth > 3:
        nknots = int(np.floor((sparsity-1)/2))
        s = np.linspace(0,2,nknots)
        pm = np.ones(nknots)
        pm[np.arange(1,nknots,2)] = -1
        ff = lambda x: fun(x-1)
        (w1, w2), (b1, b2) = param_outer(s, pm, ff)
        #lottery ticket of form relu(1+x) then wide univariate function (2 layers) + bias 1 followed by relu()
        weightTarget = list()
        biasTarget = list()
        w0 = np.array([1]).reshape((1,1))
        b0 = np.array([1])
        weightTarget.append(w0)
        biasTarget.append(b0)
        #2 layers approximating univariate function
        weightTarget.append(w1)
        biasTarget.append(b1)
        weightTarget.append(w2)
        biasTarget.append(b2)
        if depth > 4:
            biasTarget[-1][0] = biasTarget[-1][0] + 1
            for i in range(4,depth):
                w0 = np.array([1]).reshape((1,1))
                b0 = np.array([0])
                weightTarget.append(w0)
                biasTarget.append(b0)
            biasTarget[-1][0] = -1
    return weightTarget, biasTarget

def fun_ex1(x):
    y = 5*np.pi + 3*np.pi*x
    return y*np.cos(y)/(8*np.pi)

def fun_ex2(x):
    y = 5*np.pi + 3*np.pi*x
    return y*np.sin(y)/(8*np.pi)

def fun_ex3(x):
    return (5*np.pi + 3*np.pi*x)/(8*np.pi)

def lottery_helix_wide(fun1, fun2, fun3, depth, nknots):
    #sparsity: number of weights in lottery ticket, i.e., non-zero weights in lottery ticket
    #depth = number of layers (including input layer)
    #assume continuous function fun: [-1,1] -> [-1,1]
    #ReLU activation functions, except last layer, last layer linear
    #output: 3-dim with out1 = fun1, out2 = fun2, out3 = fun3
    nknots = int(nknots)
    if depth == 3:
        s = np.linspace(-1,1,nknots)
        pm = np.ones(nknots)
        pm[np.arange(1,nknots,2)] = -1
        wt, bt = param_outer(s, pm, fun1)
        (_, w2), (_,b2) = param_outer(s, pm, fun2)
        (_, w3), (_,b3) = param_outer(s, pm, fun3)
        weightTarget = list()
        biasTarget = list()
        weightTarget.append(wt[0])
        biasTarget.append(bt[0])
        weightTarget.append(np.hstack([wt[1],w2,w3]))
        biasTarget.append(np.array([bt[1][0], b2[0], b3[0]]))
        #lottery ticket of form relu(1+x) then wide univariate function (2 layers) + bias 1 followed by relu()
    if depth > 3:
        s = np.linspace(0,2,nknots)
        pm = np.ones(nknots)
        pm[np.arange(1,nknots,2)] = -1
        ff = lambda x: fun1(x-1)
        (w11, w12), (b11, b12) = param_outer(s, pm, ff)
        ff = lambda x: fun2(x-1)
        (_, w2), (_,b2) = param_outer(s, pm, ff)
        ff = lambda x: fun3(x-1)
        (_, w3), (_,b3) = param_outer(s, pm, ff)
        #lottery ticket of form relu(1+x) then wide univariate function (2 layers) + bias 1 followed by relu()
        weightTarget = list()
        biasTarget = list()
        w0 = np.array([1]).reshape((1,1))
        b0 = np.array([1])
        weightTarget.append(w0)
        biasTarget.append(b0)
        #2 layers approximating univariate function
        weightTarget.append(w11)
        biasTarget.append(b11)
        weightTarget.append(np.hstack([w12,w2,w3]))
        biasTarget.append(np.array([b12[0], b2[0], b3[0]]))
        if depth > 4:
            biasTarget[-1][:3] = biasTarget[-1][:3] + 1
            for i in range(4,depth):
                w0 = np.eye(3)
                b0 = np.zeros(3)
                weightTarget.append(w0)
                biasTarget.append(b0)
            biasTarget[-1][:3] = -1
    return weightTarget, biasTarget

def lottery_relu(depth, problemType):
    L = depth-1
    weightTarget = list()
    biasTarget = list()
    for i in range(L-1):
        w0 = np.array([1]).reshape((1,1))
        b0 = np.array([0])
        weightTarget.append(w0)
        biasTarget.append(b0)
    if problemType == "reg":
        w0 = np.array([1]).reshape((1,1))
        b0 = np.array([0])
        weightTarget.append(w0)
        biasTarget.append(b0)
    else:
        w0 = np.array([1,-2]).reshape((1,2))
        b0 = np.array([0, 0.001])
        weightTarget.append(w0)
        biasTarget.append(b0)
    return weightTarget, biasTarget

def fsq(x):
    return x**2

def lottery_sphere(depth, class_bound, nknots, problemType):
    L = int(depth-1)
    nknots = int(nknots)
    weightTarget = list()
    biasTarget = list()
    if depth == 3:
        if problemType == "reg":
            s = np.linspace(-1,1,nknots)
            pm = np.ones(nknots)
            pm[np.arange(1,nknots,2)] = -1
            pm[-1] = -1
            wt, bt = param_outer(s, pm, fsq)
            weightTarget = list()
            biasTarget = list()
            w1 = np.zeros((2,2*nknots))
            w1[0,:nknots] = wt[0].flatten()
            w1[1,nknots:] = wt[0].flatten()
            weightTarget.append(w1)
            biasTarget.append(np.concatenate([bt[0], bt[0]]))
            w2 = np.zeros((2*nknots, 1))
            w2[:nknots,0] = wt[1].flatten()
            w2[nknots:,0] = wt[1].flatten()
            weightTarget.append(w2)
            biasTarget.append(bt[1]*2)
        else:
            nclass = len(class_bound)+1
            s = np.linspace(-1,1,nknots)
            pm = np.ones(nknots)
            pm[np.arange(1,nknots,2)] = -1
            pm[-1] = -1
            ff = lambda x: fsq(x)#*5
            wt, bt = param_outer(s, pm, ff)
            #print(wt)
            #print(bt)
            weightTarget = list()
            biasTarget = list()
            w1 = np.zeros((2,2*nknots))
            w1[0,:nknots] = wt[0].flatten()
            w1[1,nknots:] = wt[0].flatten()
            weightTarget.append(w1)
            biasTarget.append(np.concatenate([bt[0], bt[0]]))
            #decision boundary: 0.2, 0.5, 0.7
            #scale = 50
            #y = np.exp(scale*np.array([0.1*x, (0.2*x-0.02), (0.4*x+((0.2-0.4)*0.5-0.02)), (0.8*x+((0.4-0.8)*0.7+(0.2-0.4)*0.5-0.02))]))
            w2 = np.zeros((2*nknots, nclass))
            b2 = np.zeros(nclass)
            # w2[nknots:,0] = 5*wt[1].flatten()
            # w2[:nknots,0] = 5*wt[1].flatten()
            # w2[nknots:,1] = 10*wt[1].flatten()
            # w2[:nknots,1] = 10*wt[1].flatten()
            # w2[nknots:,2] = 20*wt[1].flatten()
            # w2[:nknots,2] = 20*wt[1].flatten()
            # w2[nknots:,3] = 40*wt[1].flatten()
            # w2[:nknots,3] = 40*wt[1].flatten()
            # b2 = np.array([2*bt[1][0]*5, 2*bt[1][0]*10-1, 2*bt[1][0]*20-6, 2*bt[1][0]*40-20])
            a = 5
            aOld = 5
            bOld = 0
            w2[:nknots,0] = a*wt[1].flatten()
            w2[nknots:,0] = a*wt[1].flatten()
            b2[0] = 2*bt[1][0]*a
            for j in range(1,nclass):
                a = 2*a#50*class_bound[j]
                w2[:nknots,j] = a*wt[1].flatten()
                w2[nknots:,j] = a*wt[1].flatten()
                b2[j] = (aOld-a)*class_bound[j-1] + bOld + 2*bt[1][0]*a
                bOld = (aOld-a)*class_bound[j-1] + bOld
                aOld = a
            weightTarget.append(w2)
            biasTarget.append(b2)
    if depth == 4:
        #nknots = max(np.floor(nknots/2),2)
        nknots = int(nknots)
        s = np.linspace(0,1,nknots)
        pm = np.ones(nknots)
        pm[np.arange(1,nknots,2)] = -1
        pm[-1] = -1
        wt, bt = param_outer(s, pm, fsq)
        weightTarget = list()
        biasTarget = list()
        w0 = np.zeros((2,4))
        w0[0,0] = 1
        w0[0,1] = -1
        w0[1,2] = 1
        w0[1,3] = -1
        b0 = np.zeros(4)
        weightTarget.append(w0)
        biasTarget.append(b0)
        w1 = np.zeros((4,2*nknots))
        w1[0,:nknots] = wt[0].flatten()
        w1[1,:nknots] = wt[0].flatten()
        w1[2,nknots:] = wt[0].flatten()
        w1[3,nknots:] = wt[0].flatten()
        weightTarget.append(w1)
        biasTarget.append(np.concatenate([bt[0], bt[0]]))
        if problemType == "reg":
            w2 = np.zeros((2*nknots, 1))
            w2[:nknots,0] = wt[1].flatten()
            w2[nknots:,0] = wt[1].flatten()
            weightTarget.append(w2)
            biasTarget.append(bt[1]*2)
        else:
            nclass = len(class_bound)+1
            w2 = np.zeros((2*nknots, nclass))
            b2 = np.zeros(nclass)
            a = 5
            aOld = 5
            bOld = 0
            w2[:nknots,0] = a*wt[1].flatten()
            w2[nknots:,0] = a*wt[1].flatten()
            b2[0] = 2*bt[1][0]*a
            for j in range(1,nclass):
                a = 2*a#50*class_bound[j]
                w2[:nknots,j] = a*wt[1].flatten()
                w2[nknots:,j] = a*wt[1].flatten()
                b2[j] = (aOld-a)*class_bound[j-1] + bOld + 2*bt[1][0]*a
                bOld = (aOld-a)*class_bound[j-1] + bOld
                aOld = a
            weightTarget.append(w2)
            biasTarget.append(b2)
    if depth > 4:
        weightTarget = list()
        biasTarget = list()
        w0 = np.zeros((2,4))
        w0[0,0] = 1
        w0[0,1] = -1
        w0[1,2] = 1
        w0[1,3] = -1
        b0 = np.zeros(4)
        weightTarget.append(w0)
        biasTarget.append(b0)
        w0 = np.zeros((4,2))
        w0[0,0] = 1
        w0[1,0] = 1
        w0[2,1] = 1
        w0[3,1] = 1
        b0 = np.zeros(2)
        weightTarget.append(w0)
        biasTarget.append(b0)
        #
        #vold = np.zeros([2,4])
        #vold[0,:2] = 1
        #vold[1,2:] = 1
        vold = np.zeros([2,2])
        vold[0,0] = 1
        vold[1,1] = 1
        phi = np.pi/2
        for l in range(depth-5):#range(depth-4):
            ww = np.zeros((vold.shape[1],3))
            bb = np.zeros(3)
            phi = phi/2
            v = np.array([np.cos(phi), np.sin(phi)])
            ww[:,0] = v[0]*vold[0,:] + v[1]*vold[1,:]
            ww[:,1] = v[1]*vold[0,:] - v[0]*vold[1,:]
            ww[:,2] = -ww[:,1]
            vold = np.zeros([2,3])
            vold[0,:] = np.array([v[0], v[1], v[1]])
            vold[1,:] = np.array([v[1], -v[0], -v[0]])
            weightTarget.append(ww)
            biasTarget.append(bb)
        ##compute (v*x)**2 + (vortho*x)**2
        eps = 0.01
        cc = np.sqrt(2)
        if depth == 5:
            cortho = 1
            nknots = int(nknots)
            nk = nknots#int(max(nknots*cc,2))
            nko = nknots#int(max(nknots*cortho,2))
        else:
            cortho = np.abs(v[1])*np.sqrt(class_bound[-1])+eps
            nknots = int(nknots)
            nk = int(min(np.ceil(cc/(cortho+cc)*nknots*2), nknots*2-3))
            nko = int(max(2*nknots-nk,3))
        sc = np.linspace(0,cc,nk)
        sortho = np.linspace(0,cortho,nko)
        pm = np.ones(nk)
        pm[np.arange(1,nk,2)] = -1
        pm[-1] = 1
        pmo = np.ones(nko)
        pmo[np.arange(1,nko,2)] = -1
        pmo[-1] = 1
        wt, bt = param_outer(sc, pm, fsq)
        wto, bto = param_outer(sortho, pmo, fsq)
        if depth > 5:
            w1 = np.zeros((3,nk+nko))
            w1[0,:nk] = wt[0].flatten()
            w1[1,nk:] = wto[0].flatten()
            w1[2,nk:] = wto[0].flatten()
            weightTarget.append(w1)
            biasTarget.append(np.concatenate([bt[0], bto[0]]))
        else:
            w1 = np.zeros((2,nko+nko))
            w1[0,:nko] = wto[0].flatten()
            w1[1,nko:] = wto[0].flatten()
            weightTarget.append(w1)
            biasTarget.append(np.concatenate([bto[0], bto[0]]))
        if problemType == "reg":
            if depth > 5:
                w2 = np.zeros((nk+nko, 1))
                w2[:nk,0] = wt[1].flatten()
                w2[nk:,0] = wto[1].flatten()
                weightTarget.append(w2)
                biasTarget.append(np.array([bt[1][0]+bto[1][0]]))
            else:
                w2 = np.zeros((nko+nko, 1))
                w2[:nko,0] = wto[1].flatten()
                w2[nko:,0] = wto[1].flatten()
                weightTarget.append(w2)
                biasTarget.append(np.array([2*bto[1][0]]))
        else:
            nclass = len(class_bound)+1
            if depth > 5:
                w2 = np.zeros((nk+nko, nclass))
                b2 = np.zeros(nclass)
                a = 5
                aOld = 5
                bOld = 0
                w2[:nk,0] = a*wt[1].flatten()
                w2[nk:,0] = a*wto[1].flatten()
                b2[0] = (bt[1][0]+bto[1][0])*a
                for j in range(1,nclass):
                    a = 2*a
                    w2[:nk,j] = a*wt[1].flatten()
                    w2[nk:,j] = a*wto[1].flatten()
                    b2[j] = (aOld-a)*class_bound[j-1] + bOld + (bt[1][0]+bto[1][0])*a
                    bOld = (aOld-a)*class_bound[j-1] + bOld
                    aOld = a
                weightTarget.append(w2)
                biasTarget.append(b2)
            else:
                w2 = np.zeros((nko+nko, nclass))
                b2 = np.zeros(nclass)
                a = 5
                aOld = 5
                bOld = 0
                w2[:nko,0] = a*wto[1].flatten()
                w2[nko:,0] = a*wto[1].flatten()
                b2[0] = (bto[1][0]+bto[1][0])*a
                for j in range(1,nclass):
                    a = 2*a
                    w2[:nko,j] = a*wto[1].flatten()
                    w2[nko:,j] = a*wto[1].flatten()
                    b2[j] = (aOld-a)*class_bound[j-1] + bOld + (bto[1][0]+bto[1][0])*a
                    bOld = (aOld-a)*class_bound[j-1] + bOld
                    aOld = a
                weightTarget.append(w2)
                biasTarget.append(b2)
    return weightTarget, biasTarget
