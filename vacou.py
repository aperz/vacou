#!/usr/bin/env python3

'''
PCA is data dependent, no use to calculate an index! You won't be able to easily recreate it
SELF: how to?
'''

import subprocess
import pandas as pd
import numpy as np
import argparse
import os

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import normalize, scale
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

def preprocess_vc(data_fp, version):
    d = pd.read_csv(data_fp, sep='\t')
    d.Animal = [a.strip('.') for a in d.Animal]
    d.index = d.Animal
    #ATT: dummy data in lieu NVC!
    if version == 1:
        resp = [12 for i in range(15)]+[6 for i in range(10)]+[14 for i in range(20)]
        d.insert(0, 'dummy_NVC', resp)

    # Unnamed cols contain NaNs
    d.drop(['Experiment', 'Unnamed: 151', 'Unnamed: 152', 'Unnamed: 171',
                'NumberOfRunsUsedForCalculatingTrialStatistics']
            , 1, inplace=True, errors='ignore')

    if version == 1:
        md = d.ix[:, :4]
        df = d.ix[:, 4:]

    if version == 2:
        md = d.ix[:, (0,1,-1)]
        df = d.ix[:, 2:-1]

    return df, md, d


def normalize_pd(d, axis = 1, how='normalize'):
    if how=="normalize":
        return pd.DataFrame(normalize(d, axis=axis),
                        columns = d.columns, index = d.index)

    if how=="scale":
        return pd.DataFrame(scale(d, axis=axis),
                        columns = d.columns, index = d.index)

def sel_var(X, threshold):
    #d = normalize_pd(d, 1)
    var_filter = VarianceThreshold(threshold = threshold)
    var_filter.fit(X)
    X_sel = var_filter.transform(X)
    o = pd.DataFrame(X_sel, columns = X.columns[var_filter.get_support()],
                        index = X.index)
    variances = pd.Series(var_filter.variances_, index = X.columns)
    return o, variances


def sel_anova(X, y, k=10):
    '''
    "Univariate linear regression tests.
    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors."
    '''
    anova_filter = SelectKBest(f_regression, k=k)
    anova_filter.fit(X, y)
    X_sel = anova_filter.transform(X)
    o = pd.DataFrame(X_sel, columns = X.columns[anova_filter.get_support()],
                        index = X.index)
    return o


def sel_mi(X, y, k=10):
    '''
    Mutual information based feature selection.
    '''
    mi_filter = SelectKBest(mutual_info_regression, k=k)
    mi_filter.fit(X, y)
    X_sel = mi_filter.transform(X)
    o = pd.DataFrame(X_sel, columns = X.columns[mi_filter.get_support()],
                        index = X.index)
    return o


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


def compare_dim_red_methods(X, y):
    '''
    http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py
    '''

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', LinearSVC())
    ])

    N_FEATURES_OPTIONS = [2, 4, 8, 10, 100]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(f_regression)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(mutual_info_regression)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    grid = GridSearchCV(pipe, cv=5, n_jobs=8, param_grid=param_grid)

    reducer_labels = ['PCA', 'KBest(f_reg)', 'KBest(mi_reg)']

    grid.fit(X, y)

    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    #std_scores = np.array(grid.cv_results_['std_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                (len(reducer_labels) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparison of feature reduction methods")
    plt.xlabel('Number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('Regression accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')
    plt.show()

def select_columns_matching(X, pattern):
    return X.ix[:, X.columns.str.match(pattern)]


def glms():
    #TODO Explore glm / mixed model possibilities
    pass

if __name__ == "__main__":
    data_fp = '/P/vacou/data_ss31.tsv'




