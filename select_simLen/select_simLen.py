import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path
import os
from matplotlib.ticker import FormatStrFormatter

it = [5,10,15,20,25,*np.arange(50,501,50)]
samples = 100
fig, ax = plt.subplots(figsize=(20,10), nrows=2, ncols=2)
for i, status in enumerate(['qcol', 'qinf']):
    data = pd.read_csv(status+'_rpd_0.csv')
    q = data.values.mean(0)
    s = []
    for j in it:
        s.append(np.random.choice(q,(samples,j), replace=True).mean(1)) 
    df = pd.DataFrame(s).T
    df.columns = it
    sns.boxplot(data=df, color='#beaed4', orient="v", ax=ax[i][0])
    
    means = df.mean(0)
    means.plot(ax=ax[i][1], linewidth=2, color='green')
    ax2 = ax[i][1].twinx()
    sds = df.std(0)
    sds.plot(ax=ax2, linewidth=2, color='black')
    ax[i][1].set_ylabel('Sample mean', color='green', fontsize=16)
    ax[i][1].set_ylim([int(means.min())-1,int(means.max())+2])
    ax[i][0].tick_params(labelsize=14)
    ax[i][1].tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax2.set_ylabel('Sample\nstandard deviation', color='black', fontsize=16)
    ax[i][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i][0].set_xlabel('Sample size', fontsize=16)
    # ax[3][1].set_xlabel('Sample size', fontsize=16)
for i in range(2):
    for j in range(2):
        ax[i][j].tick_params(labelsize=16)
ax[0][0].set_ylabel("Acquisition rate\nsampling distribution", fontsize=16)
ax[1][0].set_ylabel("Infection rate\nsampling distribution", fontsize=16)
fig.tight_layout()
fig.savefig('sampling_convergence.png', dpi=300)
plt.close(fig)
        

