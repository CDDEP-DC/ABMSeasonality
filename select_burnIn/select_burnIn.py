import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('unit_0_stats_1000.csv')
d = 90
acq = []
for i in range(1000):
    t = data.iloc[max(0,i-d):i,[10,12]].sum(0).values
    acq.append(t[0]/t[1]*1000)
plt.plot(acq[:200])
# 50