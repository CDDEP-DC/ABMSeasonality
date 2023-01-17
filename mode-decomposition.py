import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD

abmUC = pd.read_csv('./output/2021_10_27_16_11/monte carlo/UC_0.csv').mean(1).values
abmDC = pd.read_csv('./output/2021_10_27_16_11/monte carlo/DC_0.csv').mean(1).values
abm = abmUC + abmDC
abm = abm[20:]

from pydmd import HODMD
window = 180
hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=window).fit(abm)
fig,ax = plt.subplots(figsize=(16,4*len(hodmd.modes.T)), nrows=len(hodmd.modes.T), ncols=1)
for i in range(len(hodmd.modes.T)):
    ax[i].plot(hodmd.modes.T[i].real)

plt.figure()
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.plot(abm)

