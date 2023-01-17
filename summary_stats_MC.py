import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

for model in ['singleScenario', 'randomizedMConSample']:
    for status in ['qcol', 'qinf']:
        for seasonality in ['no_seasonality','2x_seasonality']:
            output = []
            path = "./monte carlo/" + model + "_" + seasonality 
            if model == 'singleScenario':
                path = glob(path+"/*/")[0]
            path += "/monte carlo/" + status+'_rpd_0.csv'
            data = pd.read_csv(path)
            print(model, status, seasonality)
            print(pd.DataFrame(data.values.flatten()).describe())
            print("-------------------")
