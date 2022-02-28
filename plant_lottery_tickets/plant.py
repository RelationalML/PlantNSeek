import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm


def target_ind(w, v):
    #search for column in w that has highest positive correlation with v
    dd = w.shape
    v2 = np.sum(v**2)
    diff = 1000
    scale = 1
    ind = -1
    c = 0
    for i in range(dd[1]):
        scale = np.sum(v*w[:,i])/v2
        x = np.sum(np.abs(w[:,i]-v*scale))
        indsign = np.where(v!=0)[0]
        if x < diff and scale > 0 and all(np.sign(w[indsign,i]).flatten() == np.sign(v[indsign]).flatten()):
            diff = x
            ind = i
            c = scale
    return ind, c

def find_ind(scale, winit, binit, wt, bt):
    #improve for univariate targets to find lottery tickets without need of planting them
    din, dout = wt.shape
    ind = np.zeros(dout,dtype=int)
    scaleNew = np.ones(dout)
    for i in range(dout):
        #only non-zero elements need to be recovered
        iloc = np.where(wt[:,i]!= 0)[0]
        tk = np.concatenate([np.array([bt[i]]), wt[iloc,i]/scale[iloc]])
        indk, ck = target_ind(np.vstack([binit,winit[iloc,:]]), tk)
        if ck == 0:
            indk = i
            ck = np.abs(winit[0,i]/tk[1])
        if indk in ind:
            indList = np.setdiff1d(np.arange(winit.shape[1]), ind)
            if len(indList) > 0:
                indk = indList[0]
                ck = np.abs(winit[0,indk]/tk[1])
            else:
                indk = 0
                ck = np.abs(winit[0,indk]/tk[1])
        ind[i] = indk
        scaleNew[i] = ck
        winit[iloc,indk] = ck*tk[1:]
        binit[indk] = ck*tk[0]
    return winit, binit, ind, scaleNew

def plant_target(weightTarget, biasTarget, weightInit, biasInit):
    widthNet = np.array([len(biasInit[i]) for i in range(len(biasInit))])
    L = len(widthNet)
    weightPlant = list()
    biasPlant = list()
    din = weightInit[0].shape[0]
    scale = np.ones(din)
    indOld = np.arange(din)
    for l in range(L):
        wplant = weightInit[l]*0
        bplant = biasInit[l]*0
        wi, bi, ind, scale = find_ind(scale, weightInit[l][indOld, :], biasInit[l], weightTarget[l], biasTarget[l])
        #print(ind)
        #print(scale)
        weightInit[l][indOld,:] = wi
        biasInit[l][ind] = bi[ind]
        for i in range(weightTarget[l].shape[0]):
            for j in range(weightTarget[l].shape[1]):
                if weightTarget[l][i,j]!= 0:
                    wplant[indOld[i],ind[j]] = wi[i,ind[j]]
        bplant[ind] = bi[ind]
        weightPlant.append(wplant)
        biasPlant.append(bplant)
        indOld = ind
    weightInit[-1] = weightInit[-1][:,ind]
    biasInit[-1] = biasInit[-1][ind]
    weightPlant[-1] = weightPlant[-1][:,ind]
    biasPlant[-1] = biasPlant[-1][ind]
    #For convenience: keep the same order in output as in target
    #scaleOut = np.copy(scale)
    #scaleOut[ind] = scale
    return weightInit, biasInit, weightPlant, biasPlant, scale

def init_He(width):
    width = np.array(width, dtype=int)
    L = len(width)-1
    weight = list()
    bias = list()
    for i in range(L):
        w = np.array(np.random.normal(loc=0.0, scale=np.sqrt(2/(width[i]+1)), size=width[i]*width[i+1])).reshape((width[i],width[i+1]))
        b = np.array(np.random.normal(loc=0.0, scale=np.sqrt(2/(width[i]+1)), size=width[i+1]))
        weight.append(w)
        bias.append(b)
    return weight, bias

def init_He_scaled(width):
    width = np.array(width, dtype=int)
    L = len(width)-1
    weight = list()
    bias = list()
    sigmaB = 1
    for i in range(L):
        sigmaB *= np.sqrt(2/(width[i]))
        w = np.array(np.random.normal(loc=0.0, scale=np.sqrt(2/(width[i])), size=width[i]*width[i+1])).reshape((width[i],width[i+1]))
        b = np.array(np.random.normal(loc=0.0, scale=sigmaB, size=width[i+1]))
        weight.append(w)
        bias.append(b)
    return weight, bias
