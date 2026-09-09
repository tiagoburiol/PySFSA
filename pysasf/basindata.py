#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""
import pandas as pd
import numpy as np
import time
import os

from IPython.display import clear_output

# PySASF imports
from pysasf import readers
from pysasf import stats
from pysasf import solvers


class BasinData:
    """
    A class to store and process a basin sediments dataset. 

    Attributes:
    ----------
    df_dict: str
        Dictionary of Pandas dataframes for store sources and suspended sediments data.
    tracers: str list
        List of tracer names.   
    sources: str list
        List of sediment sources names. 
    combs: numpy.array
        Array of indexes (int) for all possible combinations of data samples.
    props: numpy.array
        Array of all proportions solutions for all combinations (combs) of data samples.
    feas: boolean list
        List of feaseble propportions ('props[feas]' return only feasebles). 
    solver_option = srt
        Solver options ('ols' or 'gls') to calculate the proportions. Default is 'ols'.
    cm_df: pandas DataFrame
        The dataframe containing the results from clarkeminella analysis scheme.
        
    Methods:
    -------
    infos():
        Returns a dataframe whith basic dataset infos (names, sources and sample sizes).
    means():
        Returns a dataframe whith means of dataset.
    std():
        Returns a dataframe whith std of dataset.
    set_solver_option(opt):
        To set the solver option for overdeterminated system. Options are 'ols'(default) and 'gls'.
    set_output_folder(path):
        Set the output folder (str) to save files. 
    load_combs_and_props_from_files(fc,fp):
        Function to load the combinations and the proportions saved files.
    
    
    Internal Methods:
    -----------------
    _save_array_in_file(self,array,output_folder,filename,fileformat,numberformat):
        Used to save a array in a output file.
        
    """
    def __init__(self, filename):
        df = readers.read_datafile(filename)
        
        self.df_dict = {}
        self.tracers = list(df.columns[1:])
        self.sources = []
        self.combs = np.array([])
        self.props = np.array([])
        self.feas = None
        self.S_inv = None
        self.solver_option = 'ols'
        self.output_folder= os.path.join(os.path.expanduser("~"), "PySASF/output")
        self.cm_df = None
        self.cm_all_Pfea = None

        
        # split data on a dataframes dictionary
        names = df[df.columns[0]].unique()
        df_temp2 = pd.DataFrame()
        for name in names:
            data = df[df[df.columns[0]]==name].values[:,1:]
            if name[0:6]!='Target':
                df_temp = pd.DataFrame(data.astype(float), columns=self.tracers)
                self.df_dict[name]=df_temp
                self.sources.append(name)
            else:
                df_temp = pd.DataFrame(data.astype(float),columns=self.tracers)
                df_temp2 = pd.concat((df_temp2, df_temp))
        self.df_dict['Y']=df_temp2.reset_index(drop=True)

        Y = self.df_dict['Y'].values.astype(float)
        
        self.S_inv = np.linalg.inv(np.cov(Y.T))
        return None

    def infos(self):
        return (stats.infos(self))
        
    def means(self):
        return (stats.means(self))
    
    def std(self):
        return (stats.std(self))
        
    def set_solver_option(self,solver_option):
        self.solver_option = solver_option

    def get_feasebles(self,Ps, option=1):
        if option==1:
            feas = Ps[[np.all(P>0) for P in Ps]]
        if option==2:
            feas = Ps[[np.all(P>0) and np.sum(P)<1 for P in Ps]]
        if option==3:
            feas = Ps[[np.all(P>0) and np.sum(P)<=1 for P in Ps]]
        return feas

    '''
    Identify feasebles ans store in the 'self.feas' boolean array  
    '''
    def set_feasebles(self, Ps, option=1):
        if option==1:
            feas_id = [np.all(P>0) for P in Ps]
        if option==2:
            feas_id =[np.all(P>0) and np.sum(P)<1 for P in Ps]
        if option==3:
            feas_id = [np.all(P>0) and np.sum(P)<=1 for P in Ps]
        self.feas = feas_id
        


    # SET OUTPUT SOLVER ##############################################################
    def set_output_folder(self, path):
      
        if path!=self.output_folder:
            self.output_folder = path
            print ('Setting output folder as:', path)
            # Setting the folder for output saves
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                print(f"Folder '{self.output_folder}' criated succesfully.")
            else:
                print(f"Folder to save output files is: '{self.output_folder}'.")

    # SAVE FILE ########################################################################
    def _save_array_in_file(self,array,output_folder,filename,fileformat,numberformat):
        self.set_output_folder(output_folder)
        if fileformat == 'txt':          
            if numberformat =='int':
                np.savetxt(self.output_folder+'/'+filename+'.'+fileformat,array, fmt='%1.0f')
            if numberformat =='float32':
                np.savetxt(self.output_folder+'/'+filename+'.'+fileformat,array, fmt='%1.4f')
        if fileformat == 'gzip':
            import pyarrow as pa
            import pyarrow.parquet as pq
            if numberformat =='int':
                array = np.array(array).astype(np.uint16)
            if numberformat =='float32':
                array = np.array(array).astype(np.float32)
            t = pa.Table.from_arrays([array.ravel()],['col0'])
            pq.write_table(t, self.output_folder+'/'+filename+'.'+fileformat,
                           compression='gzip')
            
    # LOAD FILES ########################################################################
    def load_combs_and_props_from_files(self,fc,fp):
        print('Loading combs and props files from:', self.output_folder)
        if fc[-3:]=='txt' and fp[-3:]=='txt':
            combs = np.loadtxt(fc).astype(int)
            Ps = np.loadtxt(fp)
            self.combs = combs
            self.props = Ps
            self.filename = str(fc[len(self.output_folder):-10])
        if fc[-4:]=='gzip' and fp[-4:]=='gzip':
            import pyarrow.parquet as pq
            combs = np.array(pq.read_table(fc))
            array_shape = (int(len(combs)/4),4)
            combs = combs.reshape(array_shape)
            self.combs = combs
            Ps = np.array(pq.read_table(fp))
            array_shape = (int(len(Ps)/3),3)
            Ps = Ps.reshape(array_shape)
            self.props = Ps
            self.filename = str(fc[len(self.output_folder):-11])
        self.set_feasebles(Ps) 
        return self.combs, self.props

    def load_feasebles_from_file(self,ff):
        print('Loading feasebles proportion indexes from:', self.output_folder)
        if ff[-3:]=='txt':
            feas = np.loadtxt(ff)
            self.feas = feas.ravel().astype(bool)
        if ff[-4:]=='gzip':
            import pyarrow.parquet as pq
            feas = np.array(pq.read_table(ff))
            #array_shape = (int(len(feas)))
            #feas = feas.reshape(array_shape)
            self.feas = feas.ravel().astype(bool)
        return self.feas

    def _save_feasebles(self, feas, output_folder,filename,format):
        #booleans = [np.all(P>0) for P in Ps]
        self._save_array_in_file(self.feas,output_folder,filename,format,'int')
        print ('Feasebles boolean array is saded in:',output_folder+'/'+filename+'.'+format)
        #self.feas=booleans
        return None

    
   ################################################################################
    def calcule_all_props(self, solve_opt='ols'):
        from itertools import product
        df_dict = self.df_dict
        S_inv = self.S_inv
    
        filename = ''
        idx = []
        keys = list(df_dict.keys())
        for key in keys:
            size = len(df_dict[key])
            filename=filename+key+str(size)
            idx.append(list(np.arange(size)))
        self.filename = filename
    
        # Gera todas as combinações possíveis
        combs = list(product(*idx))

        #cria um array vazio para todos os Ps
        Ps = np.empty((len(combs),len(keys)-1)).astype(float)

        t1 = time.time()
        for k, comb in enumerate(combs):
            t2 = time.time()
            if t2-t1>0.5:
                clear_output(wait=True)
                percent = np.round(100*k/len(combs),2)
                print('Calculating proportion:', k, 'of', len(combs),
                      '(', percent ,'%)', end='\r', flush=True)
                t1=t2

            X = []
            for i in range(len(comb)-1):
                key = keys[i]
                pos = comb[i]
                data = df_dict[key].values[pos]
                X.append(data.astype(float)) 
            y = df_dict['Y'].values[comb[-1]].astype(float)
            X = np.array(X).T
            #SOLVE
            if solve_opt == 'ols0':
                P = solvers.solve_ols(y,X)
            if solve_opt == 'gls':
                P = solvers.solve_gls_4x4(y,X,S_inv)
            if solve_opt == 'ols':
                P = solvers.solve_ols_cm(y,X)
            if solve_opt == 'opt':
                P = solvers.solve_minimize(y,X)
            Ps[k] = P
        self.set_feasebles(Ps)
        clear_output(wait=True)
        return combs, Ps
    ###############################################################################

    
    
    '''
    calculate_and_save_all_proportions
    '''
    def calculate_and_save_all_proportions(self, format='txt', key='Y', load=True):
        if key!='Y':
            data=self.df_dict[key].values.astype(float)
            self.S_inv = np.linalg.inv(np.cov(data.T))
        
        inicio = time.time()
        # Calculating...
        combs, Ps = self.calcule_all_props(solve_opt = self.solver_option)
        
        
        fim = time.time()
        print ("Done! Time processing:",fim-inicio)
        print('Total combinations:',len(combs),', shape of proportions:', Ps.shape)
        inicio = time.time()

        # Saving
        if not os.path.exists(self.output_folder):
           os.makedirs(self.output_folder)
           print(f"Folder '{self.output_folder}' criated succesfully.")
        filename = ''
        for key in self.df_dict.keys():
            filename = filename+key+str(len(self.df_dict[key]))
        self.combs_filename = filename+'_combs'      
        self.props_filename = filename+'_props'
        self.feas_filename = filename+'_feas'
        print('Saving combinations indexes in:',
              self.output_folder+'/'+self.combs_filename+'.'+format)
        print('Saving proportions calculated in:',
              self.output_folder+'/'+self.props_filename+'.'+format)
        self._save_array_in_file(combs, self.output_folder, self.combs_filename, format,'int')
        self._save_array_in_file(Ps, self.output_folder, self.props_filename, format,'float32')
        self._save_feasebles(self.feas,self.output_folder,self.feas_filename, format)
        fim = time.time()
        print ("Time for save files:",fim-inicio)

        
        # Loading if load option is choosed
        if load:
            c_name = self.output_folder+'/'+self.combs_filename+'.'+format
            p_name = self.output_folder+'/'+self.props_filename+'.'+format
            c,p = self.load_combs_and_props_from_files(c_name,p_name)
            self.combs = c
            self.props = p
        return None




    
