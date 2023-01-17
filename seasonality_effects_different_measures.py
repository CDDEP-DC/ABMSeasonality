import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial import ConvexHull

def getConvexHull(x, y):
    if int(sum(y)) == len(y):
        points = x
        acc = 1
    elif sum(y) < 1:
        points = []
        acc = 1
    else:
        model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
        acc = model.score(x, y)
        est = model.predict(x)
        points = x[est==True]
    try:
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices,0],points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],points[hull.vertices,1][0])
    except:
        x_hull = []
        y_hull = []
        acc = 0
    return [x_hull, y_hull, acc]

sig_levels = [0.01,0.05,0.1]
prev_range = ["0_5","5_10","10_15"]
nSims = [*np.arange(10,91,10), *np.arange(100,301,50)]   
output = pd.read_csv("./seasonality_results/seasonality_effects.csv")
output = output.loc[output['baseline_acquisition_rate']<=50,:]
output['sample_size'] = output['sample_size'].astype(int)
### t-test and mw-test
all data
for test in ['ttest_pvalue','mwtest_pvalue']:
    try:
        os.mkdir("./seasonality_results/"+test)
    except:
        pass
    fig, ax = plt.subplots(figsize=(10,15), nrows=len(sig_levels))
    for i, sigL in enumerate(sig_levels):
        output['test'] = (output[test] < sigL) * 1
        sns.scatterplot(data=output, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i], legend=False)
        x_hull, y_hull, acc = getConvexHull(output[['seasonality','baseline_acquisition_rate']].values, output['test'].values)
        ax[i].fill(x_hull, y_hull, alpha=0.3, c='grey')
        ax[i].set_title("sig. level = "+str(sigL)+", accuracy = "+str(int(acc*100))+"%")
        fig.savefig("./seasonality_results/"+test+"/"+test+".png", dpi=300)
        plt.close(fig)

# categorized by admission prevalence
for test in ['ttest_pvalue','mwtest_pvalue']:
    for sigL in sig_levels:
        output['test'] = (output[test] < sigL) * 1
        fig, ax = plt.subplots(figsize=(10,15), nrows=3)
        acc = [[] for i in range(len(prev_range))]
        for i, admC in enumerate(prev_range):
            subset = output.loc[output['admission_prevalence']==admC,:]
            sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i], legend=False)
            x_hull, y_hull, acc[i] = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['test'].values)
            ax[i].fill(x_hull, y_hull, alpha=0.3, c='grey')
            ax[i].set_ylim([0,50])
            ax[i].set_xlim([-5,105])
        ax[0].set_title('admission prevalence < 5%'+', accuracy = '+str(int(acc[0]*100))+'%')
        ax[1].set_title('5% < admission prevalence < 10%'+', accuracy = '+str(int(acc[1]*100))+'%')
        ax[2].set_title('admission prevalence > 10%, accuracy = '+str(int(acc[2]*100))+'%')
        fig.tight_layout()
        fig.savefig("./seasonality_results/"+test+"/admission_"+test+"_"+str(int(sigL*100))+".png", dpi=300)
        plt.close()

# by nSim & admission prevalence
nr = int(np.sqrt(len(nSims)))
nc = int(np.ceil(len(nSims)/nr))
for test in ['ttest_pvalue','mwtest_pvalue']:
    for admC in prev_range:
        for sigL in sig_levels:
            output['test'] = (output[test] < sigL) * 1
            fig, ax = plt.subplots(figsize=(nc*5,nr*5), nrows=nr, ncols=nc, sharex=True, sharey=True)
            for i, ns in enumerate(nSims):
                subset = output.loc[(output['admission_prevalence']==admC)&(output['sample_size']==ns)&(output['seasonality']<=100),:]
                subset['seasonality'] = subset['seasonality'].apply(int)
                sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i//nc][i%nc], legend=False)
                x_hull, y_hull, acc = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['test'].values)
                ax[i//nc][i%nc].fill(x_hull, y_hull, alpha=0.3, c='grey')
                ax[i//nc][i%nc].set_title('sample size: '+str(ns)+' ICUs, accuracy = '+str(int(acc*100))+'%')
            fig.tight_layout()
            fig.savefig("./seasonality_results/"+test+"/nSims_admission_"+admC+"_"+test+"_"+str(int(sigL*100))+".png", dpi=300)
            plt.close()

# by nSim & transmission probability
transm_range = [[1,3],[4,6],[7,9]]
for test in ['ttest_pvalue','mwtest_pvalue']:
    for transm in transm_range:
        for sigL in sig_levels:
            output['test'] = (output[test] < sigL) * 1
            fig, ax = plt.subplots(figsize=(nc*5,nr*5), nrows=nr, ncols=nc, sharex=True, sharey=True)
            for i, ns in enumerate(nSims):
                subset = output.loc[(output['transmission_probability']>=transm[0])&(output['transmission_probability']<=transm[1])&(output['sample_size']==ns)&(output['seasonality']<=100),:]
                subset['seasonality'] = subset['seasonality'].apply(int)
                sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i//nc][i%nc], legend=False)
                x_hull, y_hull, acc = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['test'].values)
                ax[i//nc][i%nc].fill(x_hull, y_hull, alpha=0.3, c='grey')
                ax[i//nc][i%nc].set_title('sample size: '+str(ns)+' ICUs, accuracy = '+str(int(acc*100))+'%')
            fig.tight_layout()
            fig.savefig("./seasonality_results/"+test+"/nSims_TransmissionPrabability_"+str(int(transm[0]))+"_"+str(int(transm[1]))+"_"+test+"_"+str(int(sigL*100))+".png", dpi=300)
            plt.close()

### seasonal_signal and seasonal_effect
# all data
for test in ['seasonal_signal_abs','seasonal_signal_rel','seasonal_effect_abs','seasonal_effect_rel']:
    try:
        os.mkdir("./seasonality_results/"+test)
    except:
        pass
    x = output[test].values
    x[x==np.inf] = np.nan
    x[np.abs(stats.zscore(x, nan_policy='omit'))>1.96] = np.nan
    x[x<0] = np.nan
    thresholds = [0, np.nanmedian(x)] 
    fig, ax = plt.subplots(figsize=(10,15), nrows=len(thresholds))
    for i, th in enumerate(thresholds):
        output['test'] = (output[test] >= th) * 1
        sns.scatterplot(data=output, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i], legend=False)
        x_hull, y_hull, acc = getConvexHull(output[['seasonality','baseline_acquisition_rate']].values, output['test'].values)
        ax[i].fill(x_hull, y_hull, alpha=0.3, c='grey')
        ax[i].set_title("threshold = "+str(np.round(th,2))+", accuracy = "+str(int(acc*100))+"%")
        fig.savefig("./seasonality_results/"+test+"/"+test+".png", dpi=300)
        plt.close(fig)
    
    # categorized by admission prevalence
    for th in thresholds:
        output['test'] = (output[test] >= th) * 1
        fig, ax = plt.subplots(figsize=(10,15), nrows=3)
        acc = [[] for i in range(len(prev_range))]
        for i, admC in enumerate(prev_range):            
            subset = output.loc[output['admission_prevalence']==admC,:]
            sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i], legend=False)
            x_hull, y_hull, acc[i] = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['test'].values)
            ax[i].fill(x_hull, y_hull, alpha=0.3, c='grey')
            ax[i].set_ylim([0,50])
            ax[i].set_xlim([-5,105])  
        ax[0].set_title('admission prevalence < 5%'+', accuracy = '+str(int(acc[0]*100))+'%')
        ax[1].set_title('5% < admission prevalence < 10%'+', accuracy = '+str(int(acc[1]*100))+'%')
        ax[2].set_title('admission prevalence > 10%, accuracy = '+str(int(acc[2]*100))+'%')
        fig.tight_layout()
        fig.savefig("./seasonality_results/"+test+"/admission_"+test+"_"+str(np.round(th,2)).replace('.','-')+".png", dpi=300)
        plt.close()
    
    # by nSim & admission prevalence
    for admC in prev_range:
        for th in thresholds:
            output['test'] = (output[test] >= th) * 1
            fig, ax = plt.subplots(figsize=(nc*5,nr*5), nrows=nr, ncols=nc, sharex=True, sharey=True)
            for i, ns in enumerate(nSims):
                subset = output.loc[(output['admission_prevalence']==admC)&(output['sample_size']==ns)&(output['seasonality']<=100),:]
                subset['seasonality'] = subset['seasonality'].apply(int)
                sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i//nc][i%nc], legend=False)
                x_hull, y_hull, acc = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['test'].values)
                ax[i//nc][i%nc].fill(x_hull, y_hull, alpha=0.3, c='grey')
                ax[i//nc][i%nc].set_title('sample size: '+str(ns)+' ICUs, accuracy = '+str(int(acc*100))+'%')
            fig.tight_layout()
            fig.savefig("./seasonality_results/"+test+"/nSims_admission_"+admC+"_"+test+"_"+str(np.round(th,2)).replace('.','-')+".png", dpi=300)
            plt.close()
    
    # by nSim & transmission probability
    transm_range = [[1,3],[4,6],[7,9]]
    for transm in transm_range:
        for th in thresholds:
            output['test'] = (output[test] >= th) * 1
            fig, ax = plt.subplots(figsize=(nc*5,nr*5), nrows=nr, ncols=nc, sharex=True, sharey=True)
            for i, ns in enumerate(nSims):
                subset = output.loc[(output['transmission_probability']>=transm[0])&(output['transmission_probability']<=transm[1])&(output['sample_size']==ns)&(output['seasonality']<=100),:]
                subset['seasonality'] = subset['seasonality'].apply(int)
                sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i//nc][i%nc], legend=False)
                x_hull, y_hull, acc = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['test'].values)
                ax[i//nc][i%nc].fill(x_hull, y_hull, alpha=0.3, c='grey')
                ax[i//nc][i%nc].set_title('sample size: '+str(ns)+' ICUs, accuracy = '+str(int(acc*100))+'%')
            fig.tight_layout()
            fig.savefig("./seasonality_results/"+test+"/nSims_TrPr_"+str(int(transm[0]))+"_"+str(int(transm[1]))+"_"+test+"_"+str(np.round(th,2)).replace('.','-')+".png", dpi=300)
            plt.close()

       
# explore all tests with different thresholds
# for test in tests:
#     if  'test' in test:
#         fig, ax = plt.subplots(figsize=(10,16), nrows=len(sig_levels))
#         for i, sigL in enumerate(sig_levels):
#             output['test'] = (output[test] < sigL) * 1
#             sns.scatterplot(data=output, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i], legend=False)
#             ax[i].set_title("sig. level = "+str(sigL))
#             fig.savefig("./seasonality_results/"+test+"/seasonality_effects_"+test+".png", dpi=300)
#             plt.close(fig)
#     else:
#         if 'abs' in test:
#             x = output[test].clip(0,output[test].max()).values
#         else:
#             x = output[test].clip(0,1).values
#         x[np.abs(stats.zscore(x, nan_policy='omit'))>1.96] = np.nan
#         thresholds = np.linspace(min(x),max(x),4)[:-1]
#         fig, ax = plt.subplots(figsize=(10,16), nrows=len(thresholds))
#         for i, thrs in enumerate(thresholds):
#             output['test'] = (x >= thrs) * 1
#             sns.scatterplot(data=output, x='seasonality', y='baseline_acquisition_rate', hue='test', size=1, ax=ax[i], legend=False)
#             ax[i].set_title("threshold = "+str(thrs))
#             fig.savefig("./seasonality_results/"+test+"seasonality_effects_"+test+".png", dpi=300)
#             plt.close(fig)

# points = subset.loc[subset['test']==0, ['seasonality', 'baseline_acquisition_rate']]
# xv = np.arange(5,100,10)
# yv = []
# for s in xv:
#     ps = points.loc[np.abs(points['seasonality']-s)<5, :]
#     ps = ps.loc[np.abs(stats.zscore(ps['baseline_acquisition_rate']))<1.96,'baseline_acquisition_rate'].values
#     if len(ps) > 0:
#         yv.append(max(ps))
#     else:
#         yv.append(0)
# ax[i//nc][i%nc].plot(xv, yv, c='red', linewidth = 2)