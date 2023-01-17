import numpy as np
import pandas as pd
from agents import Hospital
from post_processing import plotStats, plotPathogenLoad, plotIncidence, transmissionContribution
import time

def main():
    days = 410
    burnIn = 50
    hospitalsData = pd.read_csv("./data/hospitals_data.csv")
    hospital = Hospital(0, *hospitalsData.iloc[0,:].values, days, burnIn)
    hospital.simulate()
    plotStats(hospital)
    plotPathogenLoad(hospital)
    plotIncidence(hospital)
    transmissionContribution(hospital)

if __name__ == '__main__':
    main()
