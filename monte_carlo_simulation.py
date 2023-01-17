import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import multiprocessing as mp
import copy
import os
import time
from tqdm import tqdm
import random
from agents import Hospital
from post_processing import plotMonteCarloResults
from datetime import datetime

def runMonteCarlo(path, iterations, MConSamples=0):
    days = 410
    burnIn = 50
    if MConSamples > 0:
        iterations = MConSamples
    numproc = int(mp.cpu_count() / 1)
    if iterations <= numproc:
        numproc = iterations
        iter_per_proc = 1
    else:
        iter_per_proc = int(np.floor(iterations/numproc))
    pools = mp.Pool(processes=numproc)
    inputList = []
    for i in range(numproc):
        inputList.append([path, iter_per_proc, i, days, burnIn, MConSamples])
    for i in range(iterations-iter_per_proc*numproc):
        inputList[i][1] += 1
    pools.map(mc, inputList)
    # aggregate
    cols = ['S','X','UC','DC','I','N1','D1','background','env','hcw','import','admC', \
            'admI','transC','transI','incC','incI','roomLoad','bathroomLoad','stationLoad', \
            'qcol_rpd','qinf_rpd','qcol_rp', 'qinf_rp','qcol_rc','qinf_rc']
    monteCarloResults = {}
    nUnits = 1
    for c in cols:
        l = [[] for u in range(nUnits)]
        for u in range(nUnits):
            for h in range(numproc):
                filename = path+'/monte carlo/'+c+'_'+str(u)+'_'+str(h)+'.csv'
                l[u].extend(pd.read_csv(filename).T.values)
                os.remove(filename)
            pd.DataFrame(l[u]).T.to_csv(path+'/monte carlo/'+c+'_'+str(u)+'.csv', index=False)
        monteCarloResults[c] = np.array(l)
    
    monteCarloResults['burnIn'] = burnIn
    return monteCarloResults
      
def mc(inputList):
    path, iter_per_proc, h, days, burnIn, MConSamples = inputList
    mc_range = pd.read_csv("./data/mc_parameters_ranges.csv")
    hospitalsData = pd.read_csv("./data/hospitals_data.csv")
    hospital = Hospital(0, *hospitalsData.iloc[0,:].values, days, burnIn)
    hospital.monteCarlo = True
    hospital.path = path
    hospital.monteCarloResults = {}
    nUnits = len(hospital.units)
    S, X, UC, DC, I, N1, D1, background, env, hcw, importation, admC, admI, transC, transI, incC, incI, roomLoad, bathroomLoad, stationLoad = [[[] for i in range(nUnits)] for j in range(20)]
    qcol_per_1000_patient_days = []
    qinf_per_1000_patient_days = []
    qcol_per_1000_patients = []
    qinf_per_1000_patients = []
    qcol_per_1000_contacts = []
    qinf_per_1000_contacts = []
    for i in tqdm(range(iter_per_proc)):
        hospital.reset()
        admC_range = mc_range.loc[mc_range['parameters']=='admission_prev',['min','max']].values[0]
        # transmission_range = mc_range.loc[mc_range['parameters']=='transmission',['min','max']].values[0]
        # seasonality_range = mc_range.loc[mc_range['parameters']=='seasonality',['min','max']].values[0]
        admCol = np.random.uniform(*admC_range)
        # transmission = np.random.uniform(*transmission_range)
        # seasonality = np.random.uniform(*seasonality_range)
        admXS = np.random.uniform(0.5,1.5)
        admS = (1 - admCol) / (1 + admXS)
        admX = admXS * admS
        hospital.units[0].admissionStatus = [admS, admX, admCol, 0]
        # for p in ['beta_ph','beta_eh','beta_he']:
        #     hospital.transmissionParams[p] = transmission
        # hospital.transmissionParams['beta_hp'] = transmission / 2
        # hospital.transmissionParams['seasonality_strength'] = seasonality
        hospital.simulate()
        for j, u in enumerate(hospital.units):
            S[j].append(np.array(u.stats)[:,0])
            X[j].append(np.array(u.stats)[:,1])
            UC[j].append(np.array(u.stats)[:,2])
            DC[j].append(np.array(u.stats)[:,3])
            I[j].append(np.array(u.stats)[:,4])
            N1[j].append(np.array(u.stats)[:,5])
            D1[j].append(np.array(u.stats)[:,6])
            
            data = pd.DataFrame(u.log, columns=['day','hour','patient_ID','event','source'])
            data = data.loc[data['day']>=burnIn,:].reset_index(drop=True)
            ind = [any([s in data['source'][k] for s in ['back','env','HCW','admission']]) for k in range(data.shape[0])]   
            data = data.loc[ind,:]
            data.reset_index(drop=True, inplace=True)
            contribution = []
            for pathway in ['back','env','HCW','admission']:
                count = sum([pathway in data['source'][k] for k in range(data.shape[0])])
                contribution.append(count)
            nEvents = max(data.shape[0], 1)
            contribution = np.array(contribution) / nEvents * 100
            background[j].append(contribution[0])
            env[j].append(contribution[1])
            hcw[j].append(contribution[2])
            importation[j].append(contribution[3])
            
            data = pd.DataFrame(u.log, columns=['day','hour','patient_ID','event','source'])
            data = data.loc[data['day']>=burnIn,:].reset_index(drop=True)
            cols = ['colonized_admission', 'infected_admission', 'colonized_transfer', \
                    'infected_transfer', 'colonized_incidence', 'infected_incidence']
            incidence = pd.DataFrame(np.zeros((days-burnIn, len(cols))), columns=cols)
            for index, row in data.iterrows():
                if row['event'] == 'colonized':
                    if row['source'] == 'admission':
                        ind = 0
                    elif row['source'] == 'external transfer':
                        ind = 2
                    else:
                        ind = 4
                elif row['event'] == 'infected':
                    if row['source'] == 'admission':
                        ind = 1
                    elif row['source'] == 'external transfer':
                        ind = 3
                    else:
                        ind = 5
                incidence.iloc[row['day']-burnIn, ind] += 1
            admC[j].append(incidence.iloc[:,0].values)
            admI[j].append(incidence.iloc[:,1].values)
            transC[j].append(incidence.iloc[:,2].values)
            transI[j].append(incidence.iloc[:,3].values)
            incC[j].append(incidence.iloc[:,4].values)
            incI[j].append(incidence.iloc[:,5].values)
            #
            roomLoad[j].append(sum(np.array([u.rooms[r].contaminationHistory for r in range(len(u.rooms))])))
            bathroomLoad[j].append(sum(np.array([u.bathrooms[b].contaminationHistory for b in range(len(u.bathrooms))])))
            stationLoad[j].append(u.station.contaminationHistory)
            # incidence rate
            stats = pd.DataFrame(u.stats, columns=['S','X','UC','DC','I','admissions','contacts','N1','D1'])
            census = stats[['S','X','UC','DC','I']].iloc[burnIn:,:].sum(1).values
            hospitalization = stats[['admissions']].iloc[burnIn:,0].values
            contacts = stats[['contacts']].iloc[burnIn:,0].values
            r1, r2, r3, r4, r5, r6 = [[] for i in range(6)]
            for q in range(4):
                r1.append(incidence['colonized_incidence'].values[q*90:((q+1)*90)].sum() / census[q*90:((q+1)*90)].sum() * 1000)
                r2.append(incidence['infected_incidence'].values[q*90:((q+1)*90)].sum() / census[q*90:((q+1)*90)].sum() * 1000)
                r3.append(incidence['colonized_incidence'].values[q*90:((q+1)*90)].sum() / hospitalization[q*90:((q+1)*90)].sum() * 1000)
                r4.append(incidence['infected_incidence'].values[q*90:((q+1)*90)].sum() / hospitalization[q*90:((q+1)*90)].sum() * 1000)
                r5.append(incidence['colonized_incidence'].values[q*90:((q+1)*90)].sum() / contacts[q*90:((q+1)*90)].sum() * 1000)
                r6.append(incidence['infected_incidence'].values[q*90:((q+1)*90)].sum() / contacts[q*90:((q+1)*90)].sum() * 1000)
            qcol_per_1000_patient_days.append(r1)
            qinf_per_1000_patient_days.append(r2)
            qcol_per_1000_patients.append(r3)
            qinf_per_1000_patients.append(r4)
            qcol_per_1000_contacts.append(r5)
            qinf_per_1000_contacts.append(r6)
            
    # write
    for i in range(nUnits):
        pd.DataFrame(np.array(S[i]).T).to_csv(path+'/monte carlo/S_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(X[i]).T).to_csv(path+'/monte carlo/X_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(UC[i]).T).to_csv(path+'/monte carlo/UC_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(DC[i]).T).to_csv(path+'/monte carlo/DC_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(I[i]).T).to_csv(path+'/monte carlo/I_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(N1[i]).T).to_csv(path+'/monte carlo/N1_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(D1[i]).T).to_csv(path+'/monte carlo/D1_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(background[i]).T).to_csv(path+'/monte carlo/background_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(env[i]).T).to_csv(path+'/monte carlo/env_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(hcw[i]).T).to_csv(path+'/monte carlo/hcw_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(importation[i]).T).to_csv(path+'/monte carlo/import_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(admC[i]).T).to_csv(path+'/monte carlo/admC_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(admI[i]).T).to_csv(path+'/monte carlo/admI_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(transC[i]).T).to_csv(path+'/monte carlo/transC_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(transI[i]).T).to_csv(path+'/monte carlo/transI_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(incC[i]).T).to_csv(path+'/monte carlo/incC_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(incI[i]).T).to_csv(path+'/monte carlo/incI_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(roomLoad[i]).T).to_csv(path+'/monte carlo/roomLoad_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(bathroomLoad[i]).T).to_csv(path+'/monte carlo/bathroomLoad_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(stationLoad[i]).T).to_csv(path+'/monte carlo/stationLoad_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(qcol_per_1000_patient_days).T).to_csv(path+'/monte carlo/qcol_rpd_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(qinf_per_1000_patient_days).T).to_csv(path+'/monte carlo/qinf_rpd_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(qcol_per_1000_patients).T).to_csv(path+'/monte carlo/qcol_rp_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(qinf_per_1000_patients).T).to_csv(path+'/monte carlo/qinf_rp_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(qcol_per_1000_contacts).T).to_csv(path+'/monte carlo/qcol_rc_'+str(i)+'_'+str(h)+'.csv', index=False)
        pd.DataFrame(np.array(qinf_per_1000_contacts).T).to_csv(path+'/monte carlo/qinf_rc_'+str(i)+'_'+str(h)+'.csv', index=False)

def systematicMC():
    iterations = 200
    path = './monte carlo/systematicMC_' + datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
    try:
        os.mkdir(path)
    except:
        pass
    mc_range = pd.read_csv("./data/mc_parameters_ranges.csv")
    transmParams = ['beta_hp','beta_ph','beta_eh','beta_he']
    for transmission in [0.02]:#np.arange(0.01,0.15,0.02):
        params = pd.read_csv('./data/transmission_parameters.csv')
        params.loc[params['parameters'].isin(transmParams),'value'] = [transmission/2, *(np.ones(3)*transmission)]
        params.to_csv('./data/transmission_parameters.csv', index=False)
        for seasonality in np.arange(1,2.01,0.05):
            params = pd.read_csv('./data/transmission_parameters.csv')
            params.loc[params['parameters']=='seasonality_strength','value'] = seasonality
            params.to_csv('./data/transmission_parameters.csv', index=False)
            path2 = path+"/"+"_".join(['t',str(int(np.round(transmission*100))),'s',str(int(np.round((seasonality-1)*100)))])
            try:
                os.mkdir(path2)
                os.mkdir(path2+'/monte carlo')
            except:
                pass
            print(transmission, seasonality)
            monteCarloResults = runMonteCarlo(path2, iterations)
            plotMonteCarloResults(monteCarloResults, path2)

def randomizedMCWithIteration():
    samples = 200     # number of scenarios
    iterations = 200   # number of ICUs simulated per scenario
    path = './monte carlo/randomizedMC_withIter_' + datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
    mc_range = pd.read_csv("./data/mc_parameters_ranges.csv")
    try:
        os.mkdir(path)
    except:
        pass
    transmParams = ['beta_hp','beta_ph','beta_eh','beta_he']
    transmission_range = mc_range.loc[mc_range['parameters']=='transmission',['min','max']].values[0]
    seasonality_range = mc_range.loc[mc_range['parameters']=='seasonality',['min','max']].values[0]
    # admC_range = [0.01,0.15]
    # transmission_range = [0.01,0.1]
    # seasonality_range = [1,1]
    for i in tqdm(range(samples)):
        # admC = np.random.uniform(*admC_range)
        transmission = np.random.uniform(*transmission_range)
        seasonality = np.random.uniform(*seasonality_range)
        # admXS = np.random.uniform(0.5,1.5)
        # admS = (1 - admC) / (1 + admXS)
        # admX = admXS * admS
        # unitParams = pd.read_csv('./data/units_parameters.csv')
        # unitParams['admission_susceptible'] = admS
        # unitParams['admission_highly_susceptible'] = admX
        # unitParams['admission_colonized'] = admC
        # unitParams.to_csv('./data/units_parameters.csv', index=False)
        params = pd.read_csv('./data/transmission_parameters.csv')
        params.loc[params['parameters'].isin(transmParams),'value'] = [transmission/2, *(np.ones(3)*transmission)]
        params.loc[params['parameters']=='seasonality_strength','value'] = seasonality
        params.to_csv('./data/transmission_parameters.csv', index=False)
        path2 = path+"/"+"_".join(['t',str(int(np.round(transmission*100))),'s',str(int(np.round((seasonality-1)*100)))])
        try:
            os.mkdir(path2)
            os.mkdir(path2+'/monte carlo')
        except:
            pass
        monteCarloResults = runMonteCarlo(path2, iterations)
        plotMonteCarloResults(monteCarloResults, path2)

def randomizedMConSample():
    samples = 500     # number of ICUs generated
    path = './monte carlo/randomizedMConSample_' + datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
    try:
        os.mkdir(path)
        os.mkdir(path+'/monte carlo')
    except:
        pass
    monteCarloResults = runMonteCarlo(path, 1, MConSamples=samples)
    plotMonteCarloResults(monteCarloResults, path)

def singleScenario():
    iterations = 200
    path = './monte carlo/singleScenario_' + datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
    try:
        os.mkdir(path)
    except:
        pass
    transmParams = ['beta_hp','beta_ph','beta_eh','beta_he']
    
    admC = 0.1
    transmission = 0.08
    seasonality = 3
    admXS = 1
    admS = (1 - admC) / (1 + admXS)
    admX = admXS * admS
    unitParams = pd.read_csv('./data/units_parameters.csv')
    unitParams['admission_susceptible'] = admS
    unitParams['admission_highly_susceptible'] = admX
    unitParams['admission_colonized'] = admC
    unitParams.to_csv('./data/units_parameters.csv', index=False)
    params = pd.read_csv('./data/transmission_parameters.csv')
    params.loc[params['parameters'].isin(transmParams),'value'] = [transmission/2, *(np.ones(3)*transmission)]
    params.loc[params['parameters']=='seasonality_strength','value'] = seasonality
    params.to_csv('./data/transmission_parameters.csv', index=False)
    path2 = path+"/"+"_".join(['p',str(int(np.round(admC*100))),'t',str(int(np.round(transmission*100))),'s',str(int(np.round((seasonality-1)*100)))])
    try:
        os.mkdir(path2)
        os.mkdir(path2+'/monte carlo')
    except:
        pass
    monteCarloResults = runMonteCarlo(path2, iterations)
    plotMonteCarloResults(monteCarloResults, path2)

if __name__ == '__main__':
    try:
        os.mkdir('monte carlo')
    except:
        pass
    systematicMC()     
