#!/usr/bin/env python3

'''
PCA is data dependent, no use to calculate an index! You won't be able to easily recreate it
SELF: how to?
'''

import subprocess
import pandas as pd
import argparse
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize, scale
from sklearn.decomposition import PCA


def preprocess_vc(data_fp):
    d = pd.read_csv(data_fp, sep='\t')
    d.index = d.Animal
    resp = [12 for i in range(15)]+[6 for i in range(10)]+[14 for i in range(20)]
    d.insert(0, 'vc', resp)
    # Unnamed cols contain NaNs
    d.drop(['Experiment', 'Unnamed: 152', 'Unnamed: 171',
                'NumberOfRunsUsedForCalculatingTrialStatistics']
            , 1, inplace=True)

    md = d.ix[:, :4]
    df = d.ix[:, 4:]

    return df, md, d


def normalize_pd(d, axis = 1, how='normalize'):
    if how=="normalize":
        return pd.DataFrame(normalize(d, axis=axis),
                        columns = d.columns, index = d.index)

    if how=="scale":
        return pd.DataFrame(scale(d, axis=axis),
                        columns = d.columns, index = d.index)

def sel_var(d, threshold):
    #d = normalize_pd(d, 1)
    sel = VarianceThreshold(threshold = threshold)
    d_sel = sel.fit_transform(d)
    o = pd.DataFrame(d_sel, columns = d.columns[sel.get_support()],
                        index = d.index)
    variances = pd.Series(sel.variances_, index = d.columns)
    return o, variances



def get_pc(d, n_comps=10):
    pca = PCA(n_components=n_comps)
    pca.fit(d.T)
    colnames = ['PC'+str(n) for n in range(1,n_comps+1)]
    comps = pd.DataFrame(pca.components_.T, columns = colnames, index = d.index)
    exp_var = pd.Series(pca.explained_variance_ratio_, index = colnames)

    return comps, exp_var, pca


def get_plot_comps(comps, md):
    md_, comps_ = md.align(comps, axis=0)
    o = pd.concat([md_, comps_], axis=1)
    return o


if __name__ == "__main__":
    data_fp = '/P/vasc/data_ss31.tsv'




