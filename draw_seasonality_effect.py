import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

q = 3
t = np.arange(360)
x = np.ones(len(t)) + np.random.random(len(t)) * 0.2
t2 = np.linspace(-1, 1, 90, endpoint=False)
e = signal.gausspulse(t2, fc=1.5, retquad=True, retenv=True)[2]
x[(q-1)*90:q*90] += e * 0.3
plt.figure(figsize=(8,4))
plt.plot(x)
plt.ylim([0,2])
plt.xlim([0,len(t)])
plt.xlabel('Days', fontsize=12)
plt.ylabel('Admission prevalence', fontsize=12)
ax = plt.gca()
ax.axes.yaxis.set_ticks([])
plt.vlines(90, 0, 2, color='grey', linestyle='dashed')
plt.vlines(180, 0, 2, color='grey', linestyle='dashed')
plt.vlines(270, 0, 2, color='grey', linestyle='dashed')
plt.text(5, 1.8, 'Baseline', fontsize=10)
plt.text(95, 1.8, 'Baseline', fontsize=10)
plt.text(185, 1.8, 'High season', fontsize=10)
plt.text(275, 1.8, 'Post high season', fontsize=10)
plt.savefig('./output/seasonality_effect.png', dpi=300)