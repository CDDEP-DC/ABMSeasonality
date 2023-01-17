import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import imageio

nReal = 200
path = './monte carlo/2021_12_11_20_08'
try:
    os.mkdir(path+'/gif')
except:
    pass

for w in ['colonization','infection']:
    if w == 'colonization':
        UC = pd.read_csv(path+'/monte carlo/UC_0.csv')
        DC = pd.read_csv(path+'/monte carlo/DC_0.csv')
        x = UC + DC
    else:
        x = pd.read_csv(path+'/monte carlo/I_0.csv')
    avgx = x.mean(1)
    maxX = x.max().max()
    sse = []
    fig, ax = plt.subplots()
    for i in range(nReal):
        ax.plot(x.iloc[:,i+1], color='grey')
        avgi = x.iloc[:,:i+1].mean(1)
        L = ax.plot(avgi, color='black')
        ax.set_ylim([0, maxX])
        ax.set_title(str(i+1).zfill(3)+' realizations')
        fig.savefig(path+'/gif/figure_'+str(i+1)+'.png',dpi=300)
        line = L.pop(0)
        line.remove()
        sse.append(sum((avgx-avgi)**2))
    
    fig,ax = plt.subplots(figsize=(16,6), ncols=2)
    ax[0].plot(sse)
    ax[1].plot(np.log(sse))
    ax[0].set_xlabel('# realizations', fontsize=14)
    ax[1].set_xlabel('# realizations', fontsize=14)
    ax[0].set_ylabel('SSE', fontsize=14)
    ax[1].set_ylabel('log SSE', fontsize=14)
    fig.tight_layout()
    plt.savefig(path+'/plots/convergenace_'+w+'.png', dpi=300)
    plt.close()
    
    images = []
    for i in range(nReal):
        filename = path+'/gif/figure_'+str(i+1)+'.png'
        images.append(imageio.imread(filename))
        os.remove(filename)
    
    imageio.mimsave(path+'/plots/'+w+'.gif', images, format='GIF', duration=0.3)
    
    fig, ax = plt.subplots(figsize=(8,12), nrows=3, sharey=True)
    for i in range(3):
        ax[i].plot(x.iloc[:,np.random.randint(1,x.shape[1])], color='grey')
    fig.savefig(path+'/plots/3_random_realizations_'+w+'.png', dpi=300)
    plt.close(fig)
