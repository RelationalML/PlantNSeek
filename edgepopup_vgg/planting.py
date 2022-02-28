import torch

def setdiff(t1,t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return intersection

def target_ind(w, v):
    #search for column in w that has highest positive correlation with v
    dd = w.shape
    v2 = torch.sum(v**2)
    diff = 1000
    scale = 1
    ind = -1
    c = 0
    for i in range(dd[1]):
        scale = torch.sum(v*w[:,i])/v2
        x = torch.sum(torch.abs(w[:,i]-v*scale))
        indsign = torch.where(v!=0)[0]
        if x < diff and scale > 0 and all(torch.sign(w[indsign,i]).flatten() == torch.sign(v[indsign]).flatten()):
            diff = x
            ind = i
            c = scale
    return ind, c

def find_ind(scale, winit, binit, wt, bt, indinitin, indtin, dev):
    #improve for univariate targets to find lottery tickets without need of planting them
    degOut = torch.sum(torch.abs(wt),dim=1).to(dev)
    indOut = torch.where(degOut>0)[0].to(dev)
    din = wt.shape[1]
    dout = len(indOut)
    ind = torch.zeros(dout, dtype=int, device=dev)
    indTarget = torch.zeros(dout, dtype=int, device=dev)
    scaleNew = torch.ones(dout,device=dev)
    for ii in range(dout):
        i = indOut[ii]
        #only non-zero elements need to be recovered
        iloc = torch.where(wt[i,indtin]!= 0)[0]
        tk = torch.cat([torch.tensor([bt[i]],device=dev), wt[i,indtin[iloc]]/scale[iloc]]) #.to(dev)
        indk, ck = target_ind(torch.vstack([binit, torch.transpose(winit[:,indtin[iloc]],0,1)]), tk)
        if ck == 0:
            indk = i
            ck = torch.abs(winit[i,0]/tk[1]).to(dev)
        if indk in ind:
            indList = setdiff(torch.arange(winit.shape[0],device=dev), ind).to(dev)
            if len(indList) > 0:
                indk = indList[0]
                ck = torch.abs(winit[indk,indinitin[0]]/tk[1])
            else:
                indk = 0
                ck = torch.abs(winit[indk,indinitin[0]]/tk[1])
        ind[ii] = indk
        indTarget[ii] = indOut[ii]
        scaleNew[ii] = ck
        winit[indk,indtin[iloc]] = ck*tk[1:]
        binit[indk] = ck*tk[0]
    return winit, binit, scaleNew, ind, indTarget

def find_ind_conv(scale, winit, binit, wt, bt, indinitin, indtin, dev):
    degOut = torch.sum(torch.abs(wt),dim=1).to(dev)
    indOut = torch.where(degOut>0)[0]
    din = wt.shape[1]
    dout = len(indOut)
    scaleNew = torch.ones(dout, device=dev)
    #scaleNew = torch.ones(wt.shape[0])
    ind = torch.zeros(dout,dtype=int,device=dev)
    indTarget = torch.zeros(dout,dtype=int,device=dev)
    for ii in range(dout):
        i = indOut[ii]
        ind1, ind2, ind3 = torch.where(wt[i,indtin,:,:] != 0)
        #wloci = wt[iloc,i,:, :].reshape(wt[iloc,i,:, :].shape[0],-1)
        tk = torch.cat([torch.tensor([bt[i]], device=dev), wt[i,indtin[ind1],ind2,ind3]/scale[ind1]]).to(dev)
        indk, ck = target_ind(torch.vstack([binit, torch.transpose(winit[:,indinitin[ind1],ind2,ind3].reshape(winit.shape[0],-1),0,1)]), tk)
        if ck == 0:
            indk = i
            ck = torch.abs(winit[i,indinitin[ind1[0]],ind2[0],ind3[0]]/tk[1])
        if indk in ind:
            indList = setdiff(torch.arange(winit.shape[1],device=dev), ind)
            if len(indList) > 0:
                indk = indList[0]
                ck = torch.abs(winit[indk,indinitin[0],0,0]/tk[1])
            else:
                indk = 0
                ck = torch.abs(winit[0,0,0,0]/tk[1])
        ind[ii] = indk
        indTarget[ii] = indOut[ii]
        scaleNew[ii] = ck
        winit[indk,indinitin[ind1],ind2,ind3] = ck*tk[1:]
        binit[indk] = ck*tk[0]
    return winit, binit, scaleNew, ind, indTarget

def plant_target_torch(modelInit, pathTarget):
    target_dict = torch.load(pathTarget)
    target_params = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), target_dict.items()))
    init_dict = modelInit.state_dict()
    init_params = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), init_dict.items()))
    wt = target_params[list(target_params.keys())[0]]
    dev = wt.device
    lwprev = 0
    scale = torch.ones(wt.shape[1], device=dev)
    ind = torch.arange(wt.shape[1], device=dev)
    indTarget = torch.arange(wt.shape[1], device=dev)
    for ll in target_params.keys():
        print(ll)
        if ll.endswith("weight"):
            wt = target_params[ll].data.clone()
            wi = init_params[ll].data.clone()
            lw = wt.shape
            wkey = ll
            #print(wi[0,0,0,0])
        if ll.endswith("bias"):
            bt = target_params[ll].data.clone()
            bi = init_params[ll].data.clone()
            #plant target weight and bias into initial model:
            if len(lw) > 2:
                #Conv2d layer
                wi, bi, scale, ind, indTarget = find_ind_conv(scale, wi, bi, wt, bt, ind, indTarget, dev)
            else:
                #linear layer
                #adjust scale in case of previous flattening operation
                if len(lwprev) > 2:
                    dd = int(wi.shape[1]/lwprev[1])
                    scale = torch.tensor([i for i in scale for j in range(dd)], device=dev)
                    ind = torch.tensor([i for i in ind for j in range(dd)], dtype=int, device=dev)
                    indTarget = torch.tensor([i for i in indTarget for j in range(dd)], dtype=int, device=dev)
                wi, bi, scale, ind, indTarget = find_ind(scale, wi, bi, wt, bt, ind, indTarget, dev)
            #update model dictionary
                print(scale.shape)
                print(scale[:2])
            init_dict[wkey].data = wi
            init_dict[ll].data = bi
            indOld = ind
            lwprev = lw
    #update initial model
    modelInit.load_state_dict(init_dict)


#faster version without close matching of neurons
def find_ind_fast(scale, winit, binit, wt, bt, indIn, dev):
    #improve for univariate targets to find lottery tickets without need of planting them
    degOut = torch.sum(torch.abs(wt),dim=1).to(dev)
    indOut = torch.where(degOut>0)[0].to(dev)
    din = wt.shape[1]
    dout = len(indOut)
    ind = torch.zeros(dout, dtype=int, device=dev)
    indTarget = torch.zeros(dout, dtype=int, device=dev)
    scaleNew = torch.ones(dout,device=dev)
    for ii in range(dout):
        i = indOut[ii]
        #only non-zero elements need to be recovered
        iloc = torch.where(wt[i,indIn]!= 0)[0]
        ck = winit[i,indIn[0]]/wt[i,indIn[0]]*scale[0]
        binit[i] = bt[i]*ck
        winit[i,indIn] = wt[i,indIn[iloc]]*ck/scale[iloc]
        ind[ii] = i
        scaleNew[ii] = ck
    return winit, binit, scaleNew, ind

def find_ind_conv_fast(scale, winit, binit, wt, bt, indIn, dev):
    degOut = torch.sum(torch.abs(wt),dim=1).to(dev)
    indOut = torch.where(degOut>0)[0]
    dout = len(indOut)
    scaleNew = torch.ones(dout, device=dev)
    ind = torch.zeros(dout,dtype=int,device=dev)
    indTarget = torch.zeros(dout,dtype=int,device=dev)
    for ii in range(dout):
        i = indOut[ii]
        ind1, ind2, ind3 = torch.where(wt[i,indIn,:,:] != 0)
        ck = winit[i,indIn[ind1[0]],ind2[0],ind3[0]]/wt[i,indIn[ind1[0]],ind2[0],ind3[0]]*scale[ind1[0]]
        binit[i] = bt[i]*ck
        winit[i,indIn[ind1],ind2,ind3] = wt[i,indIn[ind1],ind2,ind3]*ck/scale[ind1]
        ind[ii] = i
        scaleNew[ii] = ck
    return winit, binit, scaleNew, ind

def plant_target_torch_fast(modelInit, pathTarget):
    target_dict = torch.load(pathTarget)
    target_params = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), target_dict.items()))
    init_dict = modelInit.state_dict()
    init_params = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), init_dict.items()))
    wt = target_params[list(target_params.keys())[0]]
    dev = wt.device
    lwprev = torch.ones(2,device=dev)
    scale = torch.ones(wt.shape[1], device=dev)
    ind = torch.arange(wt.shape[1], device=dev)
    for ll in target_params.keys():
        print(ll)
        if ll.endswith("weight"):
            wt = target_params[ll].data.clone()
            wi = init_params[ll].data.clone()
            lw = wt.shape
            wkey = ll
        if ll.endswith("bias"):
            bt = target_params[ll].data.clone()
            bi = init_params[ll].data.clone()
            #plant target weight and bias into initial model:
            if len(lw) > 2:
                #Conv2d layer
                wi, bi, scale, ind = find_ind_conv_fast(scale, wi, bi, wt, bt, ind, dev)
            else:
                #linear layer
                #adjust scale in case of previous flattening operation
                if len(lwprev) > 2:
                    dd = int(wi.shape[1]/lwprev[1]) #torch.prod(lwprev[2:]))
                    scale = torch.tensor([i for i in scale for j in range(dd)], device=dev)
                    ind = torch.tensor([i for i in ind for j in range(dd)], dtype=int, device=dev)
                wi, bi, scale, ind = find_ind_fast(scale, wi, bi, wt, bt, ind, dev)
            print(scale.shape)
            print(scale[:2])
            #update model dictionary
            init_dict[wkey].data = wi
            init_dict[ll].data = bi
            indOld = ind
            lwprev = lw
    #update initial model
    modelInit.load_state_dict(init_dict)
