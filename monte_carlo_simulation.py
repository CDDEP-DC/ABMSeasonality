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
from scipy.stats import qmc

def runMonteCarlo(inputList):
    samples_split, mc_params, path, iterations, days, burnIn, pn = inputList
    hospitalsData = pd.read_csv("./data/hospitals_data.csv")
    nUnits = hospitalsData['number_units'].values[0]
    S, X, UC, DC, I, N1, D1, background, env, hcw, importation, admC, admI, transC, transI, incC, incI, roomLoad, bathroomLoad, stationLoad = [[[] for i in range(nUnits)] for j in range(20)]
    qcol_per_1000_patient_days = []
    qinf_per_1000_patient_days = []
    qcol_per_1000_patients = []
    qinf_per_1000_patients = []
    qcol_per_1000_contacts = []
    qinf_per_1000_contacts = []
    UA = mc_params.copy()
    # UA.columns = UA.loc['parameter',:].values
    # UA.drop(index=['parameter','owner'], inplace=True)
    for c in ['qcol_per_1000_patient_days','qinf_per_1000_patient_days','qcol_per_1000_patients',\
              'qinf_per_1000_patients','qcol_per_1000_contacts','qinf_per_1000_contacts']:
        UA.loc[len(UA)] = [c, 'ICU', c]
    UA = UA.T.values.tolist()
    for rvs in samples_split:
        mc_params['value'] = rvs
        hospital = Hospital(0, *hospitalsData.iloc[0,:].values, days, burnIn)
        hospital.monteCarlo = True
        hospital.path = path
        hospital.monteCarloResults = {}
        # replace sampled ICU parameters
        icu_mc_params = mc_params.loc[mc_params['owner']=='ICU',:].to_dict('records')
        for r in icu_mc_params:
            exec('hospital.' + r['model_param_name'] + '=' + str(r['value']))       
        # replace sampled unit parameters
        unit_mc_params = mc_params.loc[mc_params['owner']=='unit',:].to_dict('records')
        for r in unit_mc_params:
            for unit in hospital.units:
                exec('unit.' + r['model_param_name'] + '=' + str(r['value']))
        # replace sampled transmission parameters
        transm_mc_params = mc_params.loc[mc_params['owner']=='transmission',:].to_dict('records')
        for r in transm_mc_params:
            hospital.transmissionParams[r['model_param_name']] = r['value']
        # replace other parameters (manually)
        compliance_variability = mc_params.loc[mc_params['parameter']=='HCW compliance variability from mean','value'].values[0]
        contact_variability = mc_params.loc[mc_params['parameter']=='Contact rate variability from mean','value'].values[0]
        hospital.calculateUniformDistBoundaries(compliance_variability, contact_variability)
        for unit in hospital.units:
            unit.calculateUniformDistBoundaries(compliance_variability, contact_variability)
        
        for i in tqdm(range(iterations)):
            hospital.reset()
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
                ind = hospital.transmissionParams['seasonality_quarter'].astype(int)
                UA.append([*rvs, r1[ind], r2[ind], r3[ind], r4[ind], r5[ind], r6[ind]])
            
    # write to file
    for i in range(nUnits):
        pd.DataFrame(np.array(S[i]).T).to_csv(path+'/S_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(X[i]).T).to_csv(path+'/X_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(UC[i]).T).to_csv(path+'/UC_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(DC[i]).T).to_csv(path+'/DC_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(I[i]).T).to_csv(path+'/I_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(N1[i]).T).to_csv(path+'/N1_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(D1[i]).T).to_csv(path+'/D1_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(background[i]).T).to_csv(path+'/background_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(env[i]).T).to_csv(path+'/env_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(hcw[i]).T).to_csv(path+'/hcw_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(importation[i]).T).to_csv(path+'/import_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(admC[i]).T).to_csv(path+'/admC_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(admI[i]).T).to_csv(path+'/admI_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(transC[i]).T).to_csv(path+'/transC_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(transI[i]).T).to_csv(path+'/transI_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(incC[i]).T).to_csv(path+'/incC_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(incI[i]).T).to_csv(path+'/incI_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(roomLoad[i]).T).to_csv(path+'/roomLoad_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(bathroomLoad[i]).T).to_csv(path+'/bathroomLoad_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(stationLoad[i]).T).to_csv(path+'/stationLoad_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(qcol_per_1000_patient_days).T).to_csv(path+'/qcol_rpd_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(qinf_per_1000_patient_days).T).to_csv(path+'/qinf_rpd_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(qcol_per_1000_patients).T).to_csv(path+'/qcol_rp_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(qinf_per_1000_patients).T).to_csv(path+'/qinf_rp_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(qcol_per_1000_contacts).T).to_csv(path+'/qcol_rc_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(qinf_per_1000_contacts).T).to_csv(path+'/qinf_rc_'+str(i)+'_'+str(pn)+'.csv', index=False)
        pd.DataFrame(np.array(UA).T).to_csv(path+'/UA_'+str(i)+'_'+str(pn)+'.csv', index=False)

def randomizedMCWithIteration():
    numproc = mp.cpu_count()
    nsamples = 100 # number of scenarios (draws from distributions)
    iterations = 25   # number of ICUs simulated per set of sampled variables
    days = 410
    burnIn = 50
    path = './monte carlo/' + datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
    mc_range = pd.read_csv("./data/mc_parameters_ranges.csv")
    mc_params = mc_range.loc[:,['parameter','owner','model_param_name']]
    try:
        os.mkdir(path)
    except:
        pass
    lower_bounds = mc_range.loc[:,'min'].values
    upper_bounds = mc_range.loc[:,'max'].values
    sampler = qmc.LatinHypercube(d=len(mc_range))
    samples = sampler.random(n=nsamples)
    samples = qmc.scale(samples, lower_bounds, upper_bounds)
    samples_split = np.array_split(samples, numproc)
    inputList = []
    for i in range(len(samples_split)):
        inputList.append([samples_split[i], mc_params, path, iterations, days, burnIn, i])
    pools = mp.Pool(processes=numproc)
    pools.map(runMonteCarlo, inputList)
    # aggregate
    cols = ['S','X','UC','DC','I','N1','D1','background','env','hcw','import','admC', \
            'admI','transC','transI','incC','incI','roomLoad','bathroomLoad','stationLoad', \
            'qcol_rpd','qinf_rpd','qcol_rp', 'qinf_rp','qcol_rc','qinf_rc','UA']
    monteCarloResults = {}
    nUnits = 1
    for c in cols:
        l = [[] for u in range(nUnits)]
        for u in range(nUnits):
            for pn in range(numproc):
                filename = path+'/'+c+'_'+str(u)+'_'+str(pn)+'.csv'
                l[u].extend(pd.read_csv(filename).T.values)
                os.remove(filename)
            pd.DataFrame(l[u]).T.to_csv(path+'/'+c+'_'+str(u)+'.csv', index=False)
        monteCarloResults[c] = np.array(l)
    
    monteCarloResults['burnIn'] = burnIn

if __name__ == '__main__':
    try:
        os.mkdir('monte carlo')
    except:
        pass
    randomizedMCWithIteration()     
