#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd
import numpy as np

from pysasf import distances
'''
Just get basic infos and print
'''
def infos(bd):
    info = pd.DataFrame()
    for key in list(bd.df_dict):
        size = bd.df_dict[key].shape[0]
        values = np.full(len(bd.tracers),size).reshape(1,len(bd.tracers))
        df_info= pd.DataFrame(values, columns=bd.tracers)
        df_info = df_info.rename(index={0: key})
        info = pd.concat((info, df_info))
    info.columns.name = 'Sample Sizes'
    #display(info)
    return(info)

def means(bd):
    means = pd.DataFrame()
    for key in list(bd.df_dict):
        df_mean = bd.df_dict[key].mean(axis=0).to_frame().transpose()
        df_mean = df_mean.rename(index={0: key})
        means = pd.concat((means, df_mean))
    means.columns.name = 'Means'
    #display(means.round(2))
    return(means.astype(float).round(2))

           
def std(bd):
    std = pd.DataFrame()
    for key in list(bd.df_dict):
        df_std = bd.df_dict[key].std(axis=0).to_frame().transpose()
        df_std = df_std.rename(index={0: key})
        std = pd.concat((std, df_std))
    std.columns.name = 'STD'
    #display(std.round(2))
    return(std.astype(float).round(2))


def randon_props_subsamples(bd, key, n, only_feasebles=False, target=None):

    key_idx = list(bd.df_dict.keys()).index(key) # coluna da fonte em combs
    size = len(bd.df_dict[key]) # tamanho da amostra daquela fonte
    rand = np.random.choice(np.arange(size), n, replace=False) # sorteia n indices

    Ps = bd.props                    # proporções já calculadas
    combs = bd.combs                 # combinações de indices de amostras
    feas = np.asarray(bd.feas, dtype=bool)  # máscara booleana das soluções factíveis

    # linhas cujo indice de amostra da fonte 'key' está no subconjunto sorteado
    mask = np.isin(combs[:, key_idx], rand)

    # se 'target' for informado, restringe às combinações daquele(s) alvo(s) em Y
    if target is not None:
        targets_idx = list(bd.df_dict.keys()).index('Y')
        mask &= np.isin(combs[:, targets_idx], target)

    # se only_feasebles, mantém apenas as soluções factíveis
    if only_feasebles:
        mask &= feas

    return combs[mask], Ps[mask]



'''
Calculate de confidence region
'''    
def confidence_region(P, p = 95, space_dist='mahalanobis'):
    P0 = np.mean(P, axis=0)
    #print('Pm=',Pm)
    if space_dist=='mahalanobis':
        dist = distances.mahalanobis_dist(P, P0)
    elif space_dist=='mahalanobis2d':
        dist = distances.mahalanobis2d_dist(P, P0)
    else:
        print('The confidence region space distance',space_dist,'not avaliable')

    
    sorted_idx = np.argsort(dist)
    Psorted = P[sorted_idx]

    # em ordem crescente
    end_idx = int((p/100)*len(Psorted))
    #print ("Os 95% mais próximos:", Psorted[:,:end_idx])
    return (Psorted[:end_idx,:])

'''
Get a randon subsample set
n_list is a list containing a number of elements of each source
'''
def random_subsamples(bd,nlist):
    df_dict = bd.df_dict
    ss_dict = {}
    for i, key in enumerate(df_dict.keys()):
        data = df_dict[key].values.astype(float)
        ss_dict[key]=data[np.random.choice(len(data), nlist[i], replace=False)]
    return ss_dict

'''
Get a randon props subset
'''
def random_props_subset(arr,n):
    substet = arr[np.random.choice(len(arr), n, replace=False)]
    return substet
