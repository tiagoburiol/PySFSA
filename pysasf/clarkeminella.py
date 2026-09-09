#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd
import numpy as np
from IPython.display import clear_output
from scipy.spatial import ConvexHull
import concurrent.futures
import time


#PySASF imports
from pysasf import solvers
from pysasf import stats

def run_repetitions_and_reduction (bd, key, reductions, percents = False,
                                   repetitions = 50, target = None):
    inicio = time.time()
    cv = lambda x: np.std(x) / np.mean(x) *100
    filename = bd.filename+'_'+str(key)
    
    # create the columns names for return dataframe
    df_out_cols = ['nSamp','CV','Mean','Std', 'Total', 'Feas']
    nSources = len(bd.sources)
    for i in range(nSources):
        df_out_cols.append('MeanP'+str(i+1))
    df_out_data = []


    # apenas para calcular o numero total de Ps (Melhorar isso)
    prod =1
    for k in bd.sources:
        if k!=key:
            prod = prod*len(bd.df_dict[k])

    #calcula as reduções com base nos percentuais
    if percents == True:
        percents_reductions = reductions 
        ssize = len(bd.df_dict[key])
        percent = ssize/100
        reductions = np.round(percent*np.array(reductions))
        reductions = list(reductions.astype(int))
        print ("Number of samples:", key, reductions)

    def compute_area(pts):
        hull = ConvexHull(pts)
        area = hull.volume # area for 2d points
        return area

    # loop over the sample-size reductions
    for n in reductions:
        points_set = []
        areas = []
        filename = filename+'-'+str(n)

        t1 = time.time()
        for i in range(repetitions):
            t2 = time.time()
            if t2-t1>0.5:
                print ('Processing for', n, 'subsamples of',key,
                   ', repetition number', i+1,end='\r', flush=True)
                t1=t2

            _,Pfea = stats.randon_props_subsamples(bd, key, n,
                                                   only_feasebles=True, target=target)

            if Pfea.shape[0]>=4:
                Pcr = stats.confidence_region(Pfea[:,0:2], p = 95)
                points_set.append(Pcr)

        # areas of the confidence regions, computed in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
             areas = list(executor.map(compute_area, points_set))


        # insert data for df_out
        if percents == True:
            idx = np.argwhere(reductions == n)[0][0]
            nSamp = percents_reductions[idx]
        else:
            nSamp = n
            
        CV = np.round(cv(areas),4)
        Mean = np.round(np.mean(areas),4)
        Std = np.round(np.std(areas),4)
        Total=prod*n
        Feas = len(Pfea)
        df_out_data_n = [nSamp,CV,Mean,Std,Total,Feas]
        for i in range(nSources):
            df_out_data_n.append(np.mean(Pfea, axis=0)[i])
        df_out_data.append(df_out_data_n)
        bd.cm_df = df_out_data

    # Clean the terminal and print some infos
    print('Done!')
    clear_output(wait=True)
    fim = time.time()
    print ("Time for all runs:",fim-inicio)

    # create return dataframe
    df_out = pd.DataFrame(df_out_data, columns=df_out_cols)
    bd.cm_df = df_out

    # Saving file in cvs
    print ('Saving in', filename+'.csv')
    df_out.to_csv(bd.output_folder+'/'+filename+'.csv')
    return (df_out)
