import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import os
import seaborn as sns
from glob import glob
from scipy.spatial import ConvexHull
from scipy import stats
import multiprocessing as mp
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as pl


def readMonteCarloResults(modelName, nUnits=1, burnIn=50):
    path = "./monte carlo/" + modelName + "/monte carlo/"
    files = glob(path + "*.csv")
    monteCarloResults = {}
    for file in files:
        c = file[len(path):-6]
        l = [[] for u in range(nUnits)]
        for u in range(nUnits):
            l[u].extend(pd.read_csv(file).T.values)
        monteCarloResults[c] = np.array(l)        
    monteCarloResults['burnIn'] = burnIn
    return monteCarloResults
        
    
    
def plotStats(hospital):
    try:
        os.mkdir(hospital.path+'/plots')
    except:
        pass
    for i in range(len(hospital.units)):
        stats = pd.read_csv(hospital.path+'/units/unit_'+str(i)+'_stats.csv', index_col=0)
        stats = stats.iloc[hospital.burnIn:,:].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(16,12), nrows=3, ncols=2, sharex=True)
        stats['S'].plot(ax=ax[0][0])
        stats['X'].plot(ax=ax[0][1])
        stats['UC'].plot(ax=ax[1][0])
        stats['DC'].plot(ax=ax[1][1])
        stats.loc[:,['UC','DC']].sum(1).plot(ax=ax[2][0])
        stats['I'].plot(ax=ax[2][1])
        ax[2][0].set_xlabel('days', fontsize=16)
        ax[2][1].set_xlabel('days', fontsize=16)
        ax[0][0].set_ylabel('susceptible', fontsize=16)
        ax[0][1].set_ylabel('highly susceptible', fontsize=16)
        ax[1][0].set_ylabel('undetected col.', fontsize=16)
        ax[1][1].set_ylabel('detected col.', fontsize=16)
        ax[2][0].set_ylabel('total col.', fontsize=16)
        ax[2][1].set_ylabel('infected', fontsize=16)
        lim = [stats['S'].min(), stats['S'].max()+1]
        ax[0][0].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [stats['X'].min(), stats['X'].max()+1]
        ax[0][1].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [stats['UC'].min(), stats['UC'].max()+1]
        ax[1][0].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [stats['DC'].min(), stats['DC'].max()+1]
        ax[1][1].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [stats.loc[:,['UC','DC']].sum(1).min(), stats.loc[:,['UC','DC']].sum(1).max()+1]
        ax[2][0].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [stats['I'].min(), stats['I'].max()+1]
        ax[2][1].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        for s in range(3):
            for k in range(2):
                ax[s][k].tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(hospital.path+'/plots/unit_'+str(i)+'_stats.png', dpi=300)
        plt.close(fig)

def plotPathogenLoad(hospital):
    for i in range(len(hospital.units)):
        load = pd.read_csv(hospital.path+'/units/unit_'+str(i)+'_load.csv').iloc[:,:-2]
        load = load.iloc[hospital.burnIn:,:].reset_index(drop=True)
        n = len(hospital.units[i].rooms)
        nc = int(np.sqrt(n))
        nr = int(np.ceil(n/nc))
        fig, ax = plt.subplots(figsize=(16,16), nrows=nr, ncols=nc, sharex=True, sharey=True)
        for j in range(n):
            load.iloc[:,j].plot(ax=ax[j//nc][j%nc])
            ax[j//nc][j%nc].set_title(load.columns[j] ,fontsize=14)
            ax[j//nc][j%nc].tick_params(labelsize=14)
        for j in range(n,nc*nr):
            ax[j//nc][j%nc].set_visible(False)
        fig.tight_layout()
        fig.savefig(hospital.path+'/plots/unit_'+str(i)+'_load.png', dpi=300)
        plt.close(fig)
     
def plotIncidence(hospital):
    try:
        os.mkdir(hospital.path+'/plots')
    except:
        pass
    for i in range(len(hospital.units)):
        data = pd.read_csv(hospital.path+'/units/unit_'+str(i)+'_log.csv')
        data = data.loc[data['day']>=hospital.burnIn,:].reset_index(drop=True)
        cols = ['colonized_admission', 'infected_admission', \
                'colonized_incidence', 'infected_incidence']
        incidence = pd.DataFrame(np.zeros((hospital.simLength-hospital.burnIn, len(cols))), columns=cols)
        for index, row in data.iterrows():
            if row['event'] == 'colonized':
                if row['source'] == 'admission':
                    ind = 0
                else:
                    ind = 2
            elif row['event'] == 'infected':
                if row['source'] == 'admission':
                    ind = 1
                else:
                    ind = 3
            incidence.iloc[row['day']-hospital.burnIn, ind] += 1
        incidence.to_csv(hospital.path+'/units/unit_'+str(i)+'_importation_incidence.csv')
        fig, ax = plt.subplots(figsize=(16,8), nrows=2, ncols=2, sharex=True)
        for c in range(len(cols)):
            incidence.iloc[:,c].plot(ax=ax[c//2][c%2])
            ax[c//2][c%2].set_ylabel(cols[c], fontsize=16)
            ax[c//2][c%2].tick_params(labelsize=16)
            lim = [incidence.iloc[:,c].min(), incidence.iloc[:,c].max()+1]
            ax[c//2][c%2].set_yticks(np.arange(*lim))
        ax[1][0].set_xlabel('days', fontsize=16)
        ax[1][1].set_xlabel('days', fontsize=16)
        fig.savefig(hospital.path+'/plots/unit_'+str(i)+'_incidence.png', dpi=300)
        plt.close(fig)
        # plot acquisition rate
        census = pd.read_csv(hospital.path+'/units/unit_'+str(i)+'_stats.csv', usecols=['S','X','UC','DC','I'])
        census = census.iloc[hospital.burnIn:,:].sum(1).values
        hospitalization = pd.read_csv(hospital.path+'/units/unit_'+str(i)+'_stats.csv', usecols=['admissions']).iloc[hospital.burnIn:,0].values
        contacts = pd.read_csv(hospital.path+'/units/unit_'+str(i)+'_stats.csv', usecols=['contacts']).iloc[hospital.burnIn:,0].values
        qcol_per_1000_patient_days = []
        qinf_per_1000_patient_days = []
        qcol_per_patient = []
        qinf_per_patient = []
        qcol_per_1000_contacts = []
        qinf_per_1000_contacts = []
        for q in range(4):
            qcol_per_1000_patient_days.append(incidence['colonized_incidence'].values[q*90:((q+1)*90)].sum() / census[q*90:((q+1)*90)].sum() * 1000)
            qinf_per_1000_patient_days.append(incidence['infected_incidence'].values[q*90:((q+1)*90)].sum() / census[q*90:((q+1)*90)].sum() * 1000)
            qcol_per_patient.append(incidence['colonized_incidence'].values[q*90:((q+1)*90)].sum() / hospitalization[q*90:((q+1)*90)].sum() * 1000)
            qinf_per_patient.append(incidence['infected_incidence'].values[q*90:((q+1)*90)].sum() / hospitalization[q*90:((q+1)*90)].sum() * 1000)
            qcol_per_1000_contacts.append(incidence['colonized_incidence'].values[q*90:((q+1)*90)].sum() / contacts[q*90:((q+1)*90)].sum() * 1000)
            qinf_per_1000_contacts.append(incidence['infected_incidence'].values[q*90:((q+1)*90)].sum() / contacts[q*90:((q+1)*90)].sum() * 1000)
        xlabel = ['Q1','Q2','Q3','Q4']
        fig, ax = plt.subplots(figsize=(16,12),nrows=3,ncols=2)
        ax[0][0].bar(xlabel, qcol_per_1000_patient_days, color='#d8b365')
        ax[0][1].bar(xlabel, qinf_per_1000_patient_days, color='#5ab4ac')
        ax[1][0].bar(xlabel, qcol_per_patient, color='#d8b365')
        ax[1][1].bar(xlabel, qinf_per_patient, color='#5ab4ac')
        ax[2][0].bar(xlabel, qcol_per_1000_contacts, color='#d8b365')
        ax[2][1].bar(xlabel, qinf_per_1000_contacts, color='#5ab4ac')
        ax[0][0].set_title('Colonization', fontsize=16)
        ax[0][1].set_title('Infection', fontsize=16)
        ax[0][0].set_ylabel('Acquisition rate per \n 1000 patient-days', fontsize=16)
        ax[1][0].set_ylabel('Acquisition rate per \n 1000 hospitalizations', fontsize=16)
        ax[2][0].set_ylabel('Acquisition rate per \n 1000 HCW contacts', fontsize=16)
        for i in range(3):
            for j in range(2):
                ax[i][j].tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(hospital.path+'/plots/unit_'+str(i)+'_quarterly_acquisition_rate.png', dpi=300)
        plt.close(fig)


def transmissionContribution(hospital):
    try:
        os.mkdir(hospital.path+'/plots')
    except:
        pass
    for i in range(len(hospital.units)):
        data = pd.read_csv(hospital.path+'/units/unit_'+str(i)+'_log.csv')
        data = data.loc[data['day']>=hospital.burnIn,:].reset_index(drop=True)
        ind = [any([s in data['source'][j] for s in ['env','HCW','admission']]) for j in range(data.shape[0])]   
        data = data.loc[ind,:]
        data.reset_index(drop=True, inplace=True)
        contribution = []
        for pathway in ['env','HCW','admission']:
            count = sum([pathway in data['source'][j] for j in range(data.shape[0])])
            contribution.append(count)
        contribution = np.array(contribution) / data.shape[0] * 100
        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(['Environmental', 'HCW-mediated', 'Importation'], contribution, width=0.75, color='green')
        ax.set_ylabel('contribution (%)', fontsize=16)
        ax.tick_params(labelsize=12)
        fig.savefig(hospital.path+'/plots/unit_'+str(i)+'_transmission.png', dpi=300)
        plt.close(fig)
 
def readMonteCarloResults(path):
    cols = ['S','X','UC','DC','I','N1','D1','background','env','hcw','import','admC', \
            'admI','transC','transI','incC','incI','roomLoad','bathroomLoad','stationLoad', \
            'qcol_rpd','qinf_rpd','qcol_rp', 'qinf_rp','qcol_rc','qinf_rc']
    monteCarloResults = {}
    nUnits = 1
    numproc = int(mp.cpu_count() / 1)
    for c in cols:
        l = [[] for u in range(nUnits)]
        for u in range(nUnits):
            for h in range(numproc):
                filename = path+'/monte carlo/'+c+'_'+str(u)+'.csv'
                l[u].extend(pd.read_csv(filename).T.values)
        monteCarloResults[c] = np.array(l)
    monteCarloResults['burnIn'] = 60
    return monteCarloResults
    
           
def plotMonteCarloResults(monteCarloResults, path):
    try:
        os.mkdir(path+'/plots')
    except:
        pass
    burnIn = monteCarloResults['burnIn']
    nUnits = len(monteCarloResults['S'])
    #stats
    nc = 2
    nr = 3
    for i in range(nUnits):
        fig, ax = plt.subplots(figsize=(nc*8,nr*4), nrows=nr, ncols=nc, sharex=True)
        ax[0][0].plot(monteCarloResults['S'][i].T[burnIn:], color='grey', alpha=0.5)
        ax[0][1].plot(monteCarloResults['X'][i].T[burnIn:], color='grey', alpha=0.5)
        ax[1][0].plot(monteCarloResults['UC'][i].T[burnIn:], color='grey', alpha=0.5)
        ax[1][1].plot(monteCarloResults['DC'][i].T[burnIn:], color='grey', alpha=0.5)
        ax[2][0].plot((monteCarloResults['UC'][i].T+monteCarloResults['DC'][i].T)[burnIn:], color='grey', alpha=0.5)
        ax[2][1].plot(monteCarloResults['I'][i].T[burnIn:], color='grey', alpha=0.5)
        ax[0][0].plot(monteCarloResults['S'][i].T.mean(1)[burnIn:], color='black')
        ax[0][1].plot(monteCarloResults['X'][i].T.mean(1)[burnIn:], color='black')
        ax[1][0].plot(monteCarloResults['UC'][i].T.mean(1)[burnIn:], color='black')
        ax[1][1].plot(monteCarloResults['DC'][i].T.mean(1)[burnIn:], color='black')
        ax[2][0].plot((monteCarloResults['UC'][i].T.mean(1)+monteCarloResults['DC'][i].T.mean(1))[burnIn:], color='black')
        ax[2][1].plot(monteCarloResults['I'][i].T.mean(1)[burnIn:], color='black')
        ax[nr-1][0].set_xlabel('days', fontsize=16)
        ax[nr-1][1].set_xlabel('days', fontsize=16)
        ax[0][0].set_ylabel('susceptible', fontsize=16)
        ax[0][1].set_ylabel('highly susceptible', fontsize=16)
        ax[1][0].set_ylabel('undetected col.', fontsize=16)
        ax[1][1].set_ylabel('detected col.', fontsize=16)
        ax[2][0].set_ylabel('total col.', fontsize=16)
        ax[2][1].set_ylabel('infected', fontsize=16)
        lim = [monteCarloResults['S'][i].T.min(), monteCarloResults['S'][i].T.max()+1]
        ax[0][0].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [monteCarloResults['X'][i].T.min(), monteCarloResults['X'][i].T.max()+1]
        ax[0][1].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [monteCarloResults['UC'][i].T.min(), monteCarloResults['UC'][i].T.max()+1]
        ax[1][0].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [monteCarloResults['DC'][i].T.min(), monteCarloResults['DC'][i].T.max()+1]
        ax[1][1].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [(monteCarloResults['UC'][i].T+monteCarloResults['DC'][i].T).min(), (monteCarloResults['UC'][i].T+monteCarloResults['DC'][i].T).max()+1]
        ax[2][0].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        lim = [monteCarloResults['I'][i].T.min(), monteCarloResults['I'][i].T.max()+1]
        ax[2][1].set_yticks(np.arange(*lim, (lim[1]-lim[0])//10+1))
        for s in range(nr):
            for k in range(nc):
                ax[s][k].tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(path+'/plots/unit_'+str(i)+'_stats.png', dpi=300)
        plt.close(fig)
    
    # incidence
    cols = ['admC','admI','incC','incI']
    labels = ['colonized admission', 'infected admission', \
              'colonized incidence', 'infected incidence']
    for i in range(nUnits):
        fig, ax = plt.subplots(figsize=(16,8), nrows=2, ncols=2, sharex=True)
        for j, c in enumerate(cols):
            ax[j//2][j%2].plot(monteCarloResults[c][i].T[burnIn:], color='grey', alpha=0.5)
            ax[j//2][j%2].plot(monteCarloResults[c][i].T.mean(1)[burnIn:], color='black')
            ax[j//2][j%2].set_ylabel(labels[j], fontsize=16)
            ax[j//2][j%2].tick_params(labelsize=16)
            lim = [monteCarloResults[c][i].T.min(), monteCarloResults[c][i].T.max()+1]
            ax[j//2][j%2].set_yticks(np.arange(*lim))
        ax[1][0].set_xlabel('days', fontsize=16)
        ax[1][1].set_xlabel('days', fontsize=16)
        fig.savefig(path+'/plots/unit_'+str(i)+'_incidence.png', dpi=300)
        plt.close(fig)
    
    # contribution
    contribution = [[np.hstack(monteCarloResults[c][u]) for c in ['env','hcw','import']] for u in range(nUnits)]
    for i in range(nUnits):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.boxplot(data=contribution[i], orient="v", ax=ax)
        ax.set_xticklabels(['Environmental', 'HCW-mediated', 'Importation'])
        ax.set_ylabel('contribution (%)', fontsize=16)
        ax.tick_params(labelsize=12)
        fig.savefig(path+'/plots/unit_'+str(i)+'_transmission.png', dpi=300)
        plt.close(fig)
    
    # acquisition histogram
    for u in range(nUnits):
        fig, ax = plt.subplots(figsize=(16,10), nrows=2, ncols=2)
        qcol_rpd = monteCarloResults['qcol_rpd'][u].mean(1)
        qinf_rpd = monteCarloResults['qinf_rpd'][u].mean(1)
        qcol_rp = monteCarloResults['qcol_rp'][u].mean(1)
        qinf_rp = monteCarloResults['qinf_rp'][u].mean(1)
        sns.histplot(data=qcol_rpd, binrange=(np.floor(min(qcol_rpd)),np.ceil(max(qcol_rpd))), binwidth=2, color='#beaed4', ax=ax[0][0])
        sns.histplot(data=qinf_rpd, binrange=(np.floor(min(qinf_rpd)),np.ceil(max(qinf_rpd))), binwidth=0.5, color='#beaed4', ax=ax[0][1])
        sns.histplot(data=qcol_rp, binrange=(np.floor(min(qcol_rp)),np.ceil(max(qcol_rp))), binwidth=2, color='#beaed4', ax=ax[1][0])
        sns.histplot(data=qinf_rp, binrange=(np.floor(min(qinf_rp)),np.ceil(max(qinf_rp))), binwidth=0.5, color='#beaed4', ax=ax[1][1])
        ax[1][0].set_xlabel('Colonization acquisition per 1000 hospitalization', fontsize=18)
        ax[1][1].set_xlabel('Infection incidence per 1000 hospitalization', fontsize=18)
        ax[1][0].set_ylabel('Frequency', fontsize=18)
        ax[1][1].set_ylabel('Frequency', fontsize=18)
        ax[0][0].set_xlabel('Colonization acquisition per 1000 patient-days', fontsize=18)
        ax[0][1].set_xlabel('Infection incidence per 1000 patient-days', fontsize=18)
        ax[0][0].set_ylabel('Frequency', fontsize=18)
        ax[0][1].set_ylabel('Frequency', fontsize=18)
        for i in range(2):
            for j in range(2):
                ax[i][j].tick_params(labelsize=18)
        fig.tight_layout()
        fig.savefig(path+'/plots/unit_'+str(u)+'_acquisition.png', dpi=300)
        plt.close(fig)
    
    # acquisition rate
    cols = ['qcol_rpd','qinf_rpd','qcol_rp', 'qinf_rp','qcol_rc','qinf_rc']
    colors = ['#d8b365','#5ab4ac']
    fig, ax = plt.subplots(figsize=(16,12),nrows=3,ncols=2)
    for i, c in enumerate(cols):
        sns.boxplot(data=monteCarloResults[c][0], color=colors[i%2], orient="v", ax=ax[i//2][i%2])
        ax[i//2][i%2].set_xticklabels(['Q1','Q2','Q3','Q4'])
    ax[0][0].set_ylabel('Acquisition rate per \n 1000 patient-days', fontsize=16)
    ax[1][0].set_ylabel('Acquisition rate per \n 1000 hospitalizations', fontsize=16)
    ax[2][0].set_ylabel('Acquisition rate per \n 1000 HCW contacts', fontsize=16)
    for i in range(3):
        for j in range(2):
            ax[i][j].tick_params(labelsize=16)
    fig.tight_layout()
    fig.savefig(path+'/plots/acquisition_rate.png', dpi=300)
    plt.close(fig)
    
# def analyzeSeasonalityEffects2(sig_level=0.05):
#     path = './monte carlo/'
#     q = 3 # seasonality_quarter
#     nSamples = 100
#     nSims = np.arange(10,201,10)
#     output = []
#     folders = ['p_2','p_4','p_6','p_8','p_10','simulations']
#     for folder in folders:
#         contents = glob(path+folder+"/*/")
#         for sim in contents:
#             for ns in nSims:
#                 data = pd.read_csv(sim+"monte carlo/qcol_rpd_0.csv")
#                 baseline = np.hstack(data.iloc[0:(q-1),:].values)
#                 high_season = np.hstack(data.iloc[(q-1),:].values)
#                 baseline_samples = np.random.choice(baseline,(nSamples,ns))
#                 high_season_samples = np.random.choice(high_season,(nSamples,ns))
#                 # tt = stats.ttest_ind(baseline_samples, high_season_samples, equal_var=True)
#                 mw = stats.mannwhitneyu(baseline_samples, high_season_samples, method='auto', axis=1)
#                 seasonality = int(sim[(sim.index('s_')+2):-1])
#                 if folder == 'simulations':
#                     admC = int(sim[(sim.index('p_')+2):sim.index('_t')])
#                 else:
#                     admC = int(folder[2:])
#                 output.append([np.ones(nSamples)*seasonality, np.ones(nSamples)*admC, np.ones(nSamples)*ns, baseline_samples.mean(1), (mw.pvalue < sig_level)])
#     output = [pd.DataFrame(np.transpose(c)) for c in output]
#     output = pd.concat(output)
#     output.reset_index(drop=True, inplace=True)
#     output.columns = ['seasonality','admission_prev','sample_size','acquisition_rate','ttest']
#     output.to_csv("./monte carlo/seasonality_effects.csv", index=False)

def analyzeSeasonalityEffects():
    folders = ["_admission_0_5","_admission_5_10","_admission_10_15"]
    q = 3 # seasonality_quarter
    nSamples = 100
    nSims = [*np.arange(10,91,10), *np.arange(100,301,50)]
    output = []
    for folder in folders:
        admC = folder[folder.index("n_")+2:]
        for p in ["randomizedMC_withIter","systematicMC"]:
            path = './monte carlo/'+p+folder
            contents = glob(path+"/*/")
            for sim in contents:
                data = pd.read_csv(sim+"monte carlo/qcol_rpd_0.csv")
                incidence = pd.read_csv(sim+"monte carlo/incC_0.csv")
                for ns in nSims:
                    baseline_signal = []
                    seasonal_signal_abs_diff = []
                    seasonal_signal_rel_diff = []
                    for i in range(nSamples):
                        incidence_sampled = incidence.iloc[:,np.random.choice(np.arange(incidence.shape[1]),ns)].mean(1).values
                        cycle, trend = sm.tsa.filters.hpfilter(incidence_sampled, 1600)
                        ss_abs = max(trend)-np.mean(trend[:(q-1)*90])
                        ss_rel = ss_abs / np.mean(trend[:(q-1)*90])
                        baseline_signal.append(np.mean(trend[:(q-1)*90]))
                        seasonal_signal_abs_diff.append(ss_abs)
                        seasonal_signal_rel_diff.append(ss_rel)
                    baseline = np.hstack(data.iloc[0:(q-1),:].values)
                    high_season = np.hstack(data.iloc[(q-1),:].values)
                    baseline_samples = np.random.choice(baseline,(nSamples,ns))
                    high_season_samples = np.random.choice(high_season,(nSamples,ns))
                    mean_acq_baseline = baseline_samples.mean(1)
                    mean_acq_high = high_season_samples.mean(1)
                    seasonal_effect_abs = mean_acq_high - mean_acq_baseline
                    seasonal_effect_rel = seasonal_effect_abs / mean_acq_baseline
                    tt = stats.ttest_ind(baseline_samples, high_season_samples, equal_var=True, axis=1)
                    mw = stats.mannwhitneyu(baseline_samples, high_season_samples, method='auto', axis=1)
                    seasonality = int(sim[(sim.index('s_')+2):-1])
                    transmissionProb = int(sim[(sim.index('t_')+2):(sim.index('s_')-1)])
                    output.append([[admC]*nSamples, np.ones(nSamples)*seasonality, np.ones(nSamples)*transmissionProb, np.ones(nSamples)*ns, mean_acq_baseline, baseline_signal, tt.pvalue, mw.pvalue, seasonal_signal_abs_diff, seasonal_signal_rel_diff, seasonal_effect_abs, seasonal_effect_rel])
    output = [pd.DataFrame(np.transpose(c)) for c in output]
    output = pd.concat(output)
    output.reset_index(drop=True, inplace=True)
    output.columns = ['admission_prevalence','seasonality','transmission_probability','sample_size','baseline_acquisition_rate','baseline_signal','ttest_pvalue','mwtest_pvalue','seasonal_signal_abs','seasonal_signal_rel','seasonal_effect_abs','seasonal_effect_rel']
    output.to_csv("./seasonality_results/seasonality_effects.csv", index=False)
    output.describe().to_csv("./seasonality_results/seasonality_effects_description.csv")

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

def plotSeasonalityEffects(sig_level=0.05):
    prev_range = ["0_5","5_10","10_15"]
    nSims = [*np.arange(10,91,10), *np.arange(100,301,50)]   
    output = pd.read_csv("./seasonality_results/seasonality_effects.csv")
    output = output.loc[output['baseline_acquisition_rate']<=50,:]
    output['sample_size'] = output['sample_size'].astype(int)
    ### mw-test
    test = 'mwtest_pvalue'
    output['U-test'] = (output[test] < sig_level)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(data=output, x='seasonality', y='baseline_acquisition_rate', hue='U-test', s=15, ax=ax)
    lgd = ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left', title='U-test')
    ax.set_xlabel('Seasonality strength of admission prevalence (%)', fontsize=14)
    ax.set_ylabel('Baseline acquisition rate', fontsize=14)
    ax.tick_params(labelsize=12)
    x_hull, y_hull, acc = getConvexHull(output[['seasonality','baseline_acquisition_rate']].values, output['U-test'].values)
    ax.fill(x_hull, y_hull, alpha=0.3, c='grey')
    ax.set_title("LR accuracy = "+str(int(acc*100))+"%")
    fig.savefig("./output/"+test+".png", bbox_inches='tight', bbox_extra_artists=(lgd,), dpi=300)
    plt.close(fig)
    
    # categorized by admission prevalence
    alp = 'abc'
    acc = [[] for i in range(len(prev_range))]
    fig, ax = plt.subplots(figsize=(10,15), nrows=3)
    for i, admC in enumerate(prev_range):
        subset = output.loc[(output['admission_prevalence']==admC)&(output['sample_size']==300),:]
        sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='U-test', s=15, ax=ax[i], legend=False)
        ax[i].text(-3, 45, alp[i], fontsize=14, weight='bold')
        if i == 0:
            sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='U-test', s=15, ax=ax[i])
            lgd = ax[i].legend(bbox_to_anchor=(1, 1.02), loc='upper left', title='U-test')
        x_hull, y_hull, acc[i] = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['U-test'].values)
        ax[i].fill(x_hull, y_hull, alpha=0.3, c='grey')
        ax[i].set_ylim([0,50])
        ax[i].set_xlim([-5,105])
        ax[i].set_xlabel('Seasonality strength (%)', fontsize=14)
        ax[i].set_ylabel('Baseline acquisition rate', fontsize=14)
        ax[i].tick_params(labelsize=12)
    ax[0].set_title('admission prevalence < 5%'+', LR accuracy = '+str(int(acc[0]*100))+'%')
    ax[1].set_title('5% < admission prevalence < 10%'+', LR accuracy = '+str(int(acc[1]*100))+'%')
    ax[2].set_title('10% < admission prevalence < 15%, LR accuracy = '+str(int(acc[2]*100))+'%')
    fig.tight_layout()
    fig.savefig("./output/admission_"+test+"_.png", bbox_inches='tight', bbox_extra_artists=(lgd,), dpi=300)
    plt.close()
    
    # by nSim & admission prevalence
    nr = int(np.sqrt(len(nSims)))
    nc = int(np.ceil(len(nSims)/nr))
    for admC in prev_range:
        fig, ax = plt.subplots(figsize=(nc*5,nr*5), nrows=nr, ncols=nc, sharex=True, sharey=True)
        for i, ns in enumerate(nSims):
            subset = output.loc[(output['admission_prevalence']==admC)&(output['sample_size']==ns)&(output['seasonality']<=100),:]
            subset['seasonality'] = subset['seasonality'].apply(int)
            sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='U-test', s=15, ax=ax[i//nc][i%nc], legend=False)
            if i == len(nSims)-1:
                sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='U-test', s=15, ax=ax[i//nc][i%nc])
                lgd = ax[i//nc][i%nc].legend(bbox_to_anchor=(1, 1.02), loc='upper left', title='U-test')
            x_hull, y_hull, acc = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['U-test'].values)
            ax[i//nc][i%nc].fill(x_hull, y_hull, alpha=0.3, c='grey')
            ax[i//nc][i%nc].set_title('sample size: '+str(ns)+' ICUs, LR accuracy = '+str(int(acc*100))+'%')
            ax[i//nc][i%nc].set_xlabel('Seasonality strength (%)')
            ax[i//nc][i%nc].set_ylabel('Baseline acquisition rate')
        ax[i//nc][i%nc+1].set_visible(False)
        # fig.tight_layout()
        fig.savefig("./output/nSims_admission_"+admC+"_"+test+"_.png", bbox_inches='tight', bbox_extra_artists=(lgd,), dpi=300)
        plt.close()
    
    # by nSim & transmission probability
    transm_range = [[1,3],[4,6],[7,9]]
    for transm in transm_range:
        fig, ax = plt.subplots(figsize=(nc*5,nr*5), nrows=nr, ncols=nc, sharex=True, sharey=True)
        for i, ns in enumerate(nSims):
            subset = output.loc[(output['transmission_probability']>=transm[0])&(output['transmission_probability']<=transm[1])&(output['sample_size']==ns)&(output['seasonality']<=100),:]
            subset['seasonality'] = subset['seasonality'].apply(int)
            sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='U-test', s=15, ax=ax[i//nc][i%nc], legend=False)
            if i == len(nSims)-1:
                sns.scatterplot(data=subset, x='seasonality', y='baseline_acquisition_rate', hue='U-test', s=15, ax=ax[i//nc][i%nc])
                lgd = ax[i//nc][i%nc].legend(bbox_to_anchor=(1, 1.02), loc='upper left', title='U-test')
            x_hull, y_hull, acc = getConvexHull(subset[['seasonality','baseline_acquisition_rate']].values, subset['U-test'].values)
            ax[i//nc][i%nc].fill(x_hull, y_hull, alpha=0.3, c='grey')
            ax[i//nc][i%nc].set_title('sample size: '+str(ns)+' ICUs, LR accuracy = '+str(int(acc*100))+'%')
            ax[i//nc][i%nc].set_xlabel('Seasonality strength (%)')
            ax[i//nc][i%nc].set_ylabel('Baseline acquisition rate')
        ax[i//nc][i%nc+1].set_visible(False)
        fig.tight_layout()
        fig.savefig("./output/nSims_TransmissionPrabability_"+str(int(transm[0]))+"_"+str(int(transm[1]))+"_"+test+".png", bbox_inches='tight', bbox_extra_artists=(lgd,), dpi=300)
        plt.close()

def plotCDFSeasonality(sig_level=0.05):
    prev_range = ["0_5","5_10","10_15"]
    ns=300
    output = pd.read_csv("./seasonality_results/seasonality_effects.csv")
    output = output.loc[output['baseline_acquisition_rate']<=50,:]
    output['sample_size'] = output['sample_size'].astype(int)
    output['seasonality'] = output['seasonality'].astype(int)
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99']
    seasonality = [25,50,75,100]
    # colors = pl.cm.binary(np.linspace(0,1,20))[5::2]
    # for different seasonality levels
    max_x0 = 1
    max_x1 = 1
    fig, ax = plt.subplots(figsize=(16,12), nrows=3, ncols=2)
    for i, p in enumerate(prev_range):
        for j, s in enumerate(seasonality):
            subset = output.loc[(output['admission_prevalence']==p)&(output['seasonality']==s)&(output['sample_size']==ns),:]
            x0 = np.sort(subset.seasonal_effect_abs.values)
            x0[np.where(x0<0)] = 0
            max_x0 = max(max_x0, max(x0))
            y = np.arange(len(x0)) / float(len(x0)) * 100
            ax[i][0].plot(x0, y, color=colors[j], label=s)
            ax[i][0].set_ylabel("Admission prevalence = "+prev_range[i].replace("_","-")+"%\nCumulative probability", fontsize=14)            
            ax[i][0].tick_params(labelsize=14)
            # ax[i][0].set_title("Admission prevalence = "+prev_range[i].replace("_","-")+"%", fontsize=14)
            x1 = np.sort(subset.seasonal_effect_rel.values) * 100
            x1[np.isneginf(x1)] = np.nan
            x1[np.isinf(x1)] = np.nan
            x1[np.where(x1<0)] = 0
            x1 = x1[~np.isnan(x1)]
            max_x1 = max(max_x1, max(x1))
            y = np.arange(len(x1)) / float(len(x1)) * 100
            ax[i][1].plot(x1, y, color=colors[j], label=s)
            ax[i][1].tick_params(labelsize=14)
            # ax[i][1].set_title("Admission prevalence = "+prev_range[i].replace("_","-")+"%", fontsize=14)
    for i in range(3):
        ax[i][0].set_xlim([0, np.ceil(max_x0)])
        ax[i][1].set_xlim([0, np.ceil(max_x1)])
    ax[i][0].set_xlabel('Absolute seasonal increase in acquisition rate (cases per 1000 patient-days)', fontsize=12)
    ax[i][1].set_xlabel('Relative seasonal increase in acquisition rate (%)', fontsize=12)
    fig.tight_layout()
    lgd = ax[i][0].legend(ncol=len(seasonality), bbox_to_anchor=(0.2, -0.4), loc="lower left", title='Seasonality (%)')
    fig.savefig("./output/seasonality_detection_CDF.png", bbox_inches='tight', bbox_extra_artists=(lgd,), dpi=300)
    plt.close()
    # for different number of datasets
    nSims = [10, 50, 250] 
    max_x0 = 1
    max_x1 = 1
    s = 100
    xmax = 20
    fig, ax = plt.subplots(figsize=(8,12), nrows=3, ncols=1, sharex=True)
    for i, p in enumerate(prev_range):
        for j, ns in enumerate(nSims):
            subset = output.loc[(output['admission_prevalence']==p)&(output['seasonality']==s)&(output['sample_size']==ns),:]
            x0 = np.sort(subset.seasonal_effect_abs.values)
            x0[np.where(x0<0)] = 0
            if x0[-1] < xmax:
                x0 = np.append(x0, xmax)
            y = np.arange(len(x0)) / float(len(x0)) * 100
            max_x0 = max(max_x0, x0[np.where(y>99)[0][0]]) 
            ax[i].plot(x0, y, color=colors[j], linewidth=3, label=ns)
            ax[i].vlines(5, 0, 100, color='black', linestyle='dashed')
            ax[i].set_ylabel("Baseline adm. prev. = "+prev_range[i].replace("_","-")+"%\nCumulative probability", fontsize=14)            
            ax[i].tick_params(labelsize=14)
            print(p, ns, np.round(y[np.where(x0>=5)[0][0]]), np.mean(x0), np.std(x0))
            # x1 = np.sort(subset.seasonal_effect_rel.values) * 100
            # x1[np.isneginf(x1)] = np.nan
            # x1[np.isinf(x1)] = np.nan
            # x1[np.where(x1<0)] = 0
            # x1 = x1[~np.isnan(x1)]
            # max_x1 = max(max_x1, max(x1))
            # y = np.arange(len(x1)) / float(len(x1)) * 100
            # ax[i][1].plot(x1, y, color=colors[j], linewidth=2, label=ns)
            # ax[i][1].tick_params(labelsize=14)
    ax[i].set_xlim([0, np.ceil(max_x0)])
    ax[i].set_xlabel('Absolute seasonal increase in acquisition rate (cases per 1000 patient-days)', fontsize=12)
    # ax[i][1].set_xlabel('Relative seasonal increase in acquisition rate (%)', fontsize=12)
    fig.tight_layout()
    lgd = ax[i].legend(ncol=len(nSims), bbox_to_anchor=(0.25, -0.3), loc="lower left", title='# datasets')
    fig.savefig("./output/seasonality_detection_CDF_2.png", bbox_inches='tight', bbox_extra_artists=(lgd,), dpi=300)
    plt.close()
        