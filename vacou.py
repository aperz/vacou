#!/usr/bin/env python3


'''
PCA is data dependent, no use to calculate an index! You won't be able to easily recreate it
SELF: how to?
'''
# VERY IMPORTANT what have I done with NVC measurement? Should be included in the PCA

import subprocess
import pandas as pd
import numpy as np
import argparse
import os

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression, f_classif, mutual_info_classif
from sklearn.preprocessing import normalize, scale, maxabs_scale, minmax_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

# rpy2
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()

from preprocess import *
from toolbox import *


def preprocess_vc(version, data_fp, sep_runs_data_fp=None, use_vars_fp='use_all'):
    '''
    version =   1: dummy NVC
                2: actual NVC
                3: all separate runs
                4: merged 2 and 3
    '''
    if version == 4:
        d2, md2, full_df2 = preprocess_vc(version = 2,
                    data_fp = data_fp, use_vars_fp=use_vars_fp)
        #for d in [d2, md2, full_df2]:
        #    d['from_version'] = 'd2'
        #    d.set_index('from_version', append=True, inplace=True)

        d3, md3, full_df3 = preprocess_vc(version = 3,
                    data_fp = sep_runs_data_fp, use_vars_fp=use_vars_fp)
        #for d in [d3, md3, full_df3]:
        #    d['from_version'] = 'd3'
        #    d.set_index('from_version', append=True, inplace=True)

        d       = pd.concat([i for i in d2.align(d3, axis=1, join='inner')])
        md      = pd.concat([i for i in md2.align(md3, axis=1, join='inner')])
        full_df = pd.concat([i for i in full_df2.align(full_df3, axis=1, join='inner')])

        return d, md, full_df

    else:
        d = pd.read_csv(data_fp, sep='\t')
        if version == 1 or version == 2:
            try:
                d.Animal = [a.strip('.') for a in d.Animal]
            except AttributeError as e:
                None
                #print(e)
        d = d.T.drop_duplicates().T
        #ATT: dummy data in lieu NVC!
        if version == 1:
            resp = [12 for i in range(15)]+[6 for i in range(10)]+[14 for i in range(20)]
            d.insert(0, 'dummy_NVC', resp)

        # Unnamed cols contain NaNs
        d.drop(['Experiment', 'Unnamed: 151', 'Unnamed: 152', 'Unnamed: 171',
                    'NumberOfRunsUsedForCalculatingTrialStatistics']
                , 1, inplace=True, errors='ignore')


        if version == 1:
            d = d.set_index(['Group', 'Animal'], inplace=False, drop=False)
            md = d.ix[:, :4]
            df = d.ix[:, 4:]

        if version == 2:
            d['Group'].replace('ss31 aged injected', 'AT', inplace=True)
            d['Group'].replace('young ctrl', 'YC', inplace=True)
            d['Group'].replace('Aged Ctrl', 'AC', inplace=True)
            d = d.set_index(['Group', 'Animal'], inplace=False, drop=False)
            md = d.ix[:, (0,1)]
            df = d.ix[:, 2:]

        if version == 3:
            d['Group'].replace('ss31 aged injected', 'AT', inplace=True)
            d['Group'].replace('PUMPS AII LNAME SPIKES', 'YC', inplace=True)
            d['Group'].replace('aged control sal injected', 'AC', inplace=True)
            d = d.set_index(['Group', 'Animal'], inplace=False, drop=False)
            md = d[['Group', 'Animal', 'Run', 'Trial']]
            #md.replace
            df = d

        # subset variables
        if not use_vars_fp == 'use_all':
            use_vars = []
            with open(use_vars_fp, "r") as o:
                for l in o.readlines():
                    use_vars.append(l.strip())

            use_vars = set(use_vars)
            use_vars = [i for i in use_vars if i in d.columns]
            df = df[use_vars]

        if version == 1:
            d = d.T.drop_duplicates().T
            df = df.T.drop_duplicates().T

        return df, md, d


def sel_var(X, threshold):
    #d = normalize_pd(d, 1)
    var_filter = VarianceThreshold(threshold = threshold)
    var_filter.fit(X)
    X_sel = var_filter.transform(X)
    o = pd.DataFrame(X_sel, columns = X.columns[var_filter.get_support()],
                        index = X.index)
    variances = pd.Series(var_filter.variances_, index = X.columns)
    return o, variances


def sel_anova(X, y, mode='classif'):
    '''
    "Univariate linear regression tests.
    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors."

    mode: 'classif' or 'regression'
    '''
    if mode == 'classif':
        anova_filter = SelectKBest(f_classif, k='all')
    if mode == 'regression':
        anova_filter = SelectKBest(f_regression, k='all')

    anova_filter.fit(X, y)
    X_sel = anova_filter.transform(X)
    support = pd.DataFrame(X_sel, columns = X.columns[anova_filter.get_support()],
                        index = X.index)
    pvals = anova_filter.pvalues_
    o = pd.concat([
                    pd.DataFrame(support.columns, columns=['param']),
                    pd.DataFrame(pvals, columns=['pval'])
                ], axis=1)\
            .sort_values('pval')
    return o, support


def sel_mi(X, y, k=10, mode='classif'):
    '''
    Mutual information based feature selection.
    mode: 'classif' or 'regression'
    '''
    if mode == 'classif':
        mi_filter = SelectKBest(mutual_info_classif, k=k)
    if mode == 'regression':
        mi_filter = SelectKBest(mutual_info_regression, k=k)

    mi_filter.fit(X, y)
    X_sel = mi_filter.transform(X)
    o = pd.DataFrame(X_sel, columns = X.columns[mi_filter.get_support()],
                        index = X.index)
    pvals = mi_filter.pvalues_
    return o, pvals


def get_pc(d, **kwargs):
    pca = PCA(**kwargs)
    pca.fit(d.T)
    colnames = ['PC'+str(n) for n in range(1,pca.components_.shape[0]+1)]
    comps = pd.DataFrame(pca.components_.T, columns = colnames, index = d.index)
    exp_var = pd.Series(pca.explained_variance_ratio_, index = colnames)

    return comps, exp_var, pca


def get_plot_comps(comps, md):
    md_, comps_ = md.align(comps, axis=0)
    o = pd.concat([md_, comps_], axis=1)
    return o


def compare_dim_red_methods(X, y, mode='classif'):
    '''
    http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py
    '''

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', LinearSVC())
    ])

    N_FEATURES_OPTIONS = [1, 2, 4, 8, 10, 60]
    C_OPTIONS = [1, 10, 100, 1000]
    if mode == 'regression':
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

    if mode == 'classif':
        param_grid = [
            {
                'reduce_dim': [PCA(iterated_power=7)],
                'reduce_dim__n_components': N_FEATURES_OPTIONS,
                'classify__C': C_OPTIONS
            },
            {
                'reduce_dim': [SelectKBest(f_classif)],
                'reduce_dim__k': N_FEATURES_OPTIONS,
                'classify__C': C_OPTIONS
            },
            {
                'reduce_dim': [SelectKBest(mutual_info_classif)],
                'reduce_dim__k': N_FEATURES_OPTIONS,
                'classify__C': C_OPTIONS
            },
        ]
    grid = GridSearchCV(pipe, cv=5, n_jobs=8, param_grid=param_grid)

    if mode == 'regression':
        reducer_labels = ['PCA', 'KBest(f_regression)', 'KBest(mi_regression)']
    if mode == 'classif':
        reducer_labels = ['PCA', 'KBest(f_classif)', 'KBest(mi_classif)']

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
    if mode == 'regression':
        plt.ylabel('Regression accuracy')
    if mode == 'classif':
        plt.ylabel('Classification accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')
    plt.show()



def corr_L_R_(A):
    L = pd.concat([select_columns_matching(A, "LH"), select_columns_matching(A, 'LF')],
                    axis=1, join='inner')
    R = pd.concat([select_columns_matching(A, "RH"), select_columns_matching(A, 'RF')],
                    axis=1, join='inner')
    L.columns = [i.lstrip('L') for i in L.columns]
    R.columns = [i.lstrip('R') for i in R.columns]

    corrs = {}
    for var in L.columns:
        #return L[[var]], R[[var]]
        corrs[var] = pairwise_distances(
                                pd.merge(L[[var]], R[[var]],
                                        right_index=True, left_index=True).T,
                    metric='correlation')[0,1]

    #return pd.DataFrame.from_dict(corrs)
    return pd.DataFrame(pd.Series(corrs))

def corr_L_R(A):
    results = A.groupby(A.index.get_level_values(0)).apply(lambda x: corr_L_R_(x))\
                .reset_index()
    results.columns = ['group', 'parameter', 'correlation']
    #results = A.groupby(A.index.get_level_values(0))\
                #.apply(lambda x: x.sort_values('correlation', ascending=False))
    return results


def plot_L_R_(A):
    L = pd.concat([select_columns_matching(A, "LH"), select_columns_matching(A, 'LF')],
                    axis=1, join='inner')
    R = pd.concat([select_columns_matching(A, "RH"), select_columns_matching(A, 'RF')],
                    axis=1, join='inner')
    L.columns = [i.lstrip('L') for i in L.columns]
    R.columns = [i.lstrip('R') for i in R.columns]

    L.index = L.index.droplevel(0)
    L = L.reset_index().pivot(columns='Animal').sum(0)
    R.index = R.index.droplevel(0)
    R = R.reset_index().pivot(columns='Animal').sum(0)

    o = pd.concat(L.align(R, join='inner'), 1)
    o.columns = ['L', 'R']
    return o


def plot_L_R(A):
    o = A.groupby(A.index.get_level_values(0)).apply(lambda x: plot_L_R_(x))\
                .reset_index()
    o.rename(columns={'level_1': 'parameter', 'level_2':'Animal'}, inplace=True)
    return o


def glms():
    #TODO Explore glm / mixed model possibilities
    pass


def anovaR(data_fp, formula):
    tmpfile='.tmpvacou'
    #if not os.path.isfile(tmpfile):
        #os.mknod(tmpfile)
    #r.assign('d', data)
    r("d <- read.csv('"+data_fp+"', sep='\\t')")
    r('anova <- aov('+formula+', data=d)')
    call = 'anova <- aov('+formula+', data=d)'
    r("capture.output(summary(anova), file='"+tmpfile+"')")
    #r("o = capture.output(summary(anova))")
    with open(tmpfile, "r") as f:
        o = []
        o.append(call)
        o.append('\n')
        for i,l in enumerate(f.readlines()):
            o.append(l)
    return o#, call


def manovaR(data_fp = 'data/PCs.tsv',
            formula = 'cbind(PC1, PC2, PC3)~ Group', groups='all'):
    tmpfile='.tmpvacou'
    r("data <- read.csv('"+data_fp+"', sep='\\t')") #, check.names=FALSE)")

    if groups !='all':
        assert len(groups) == 2
        r("data <- droplevels(data[data$Group %in% c('"+ groups[0] +"', '"+groups[1]+"'),])")

    r('ma <- manova('+formula+', data=data)')
    call = 'ma <- manova('+formula+', data=data)'

    r("capture.output(summary(ma), file='"+tmpfile+"')")
    with open(tmpfile, "r") as f:
        o = {}
        o['summary'] = []
        o['summary.aov'] = []
        o['summary'].append(call)
        o['summary'].append('\n')
        o['summary.aov'].append(call)
        o['summary.aov'].append('\n')
        for i,l in enumerate(f.readlines()):
            o['summary'].append(l.rstrip())

    r("capture.output(summary.aov(ma), file='"+tmpfile+"')")
    with open(tmpfile, "r") as f:
        for i,l in enumerate(f.readlines()):
            o['summary.aov'].append(l.rstrip())
    return o#, call


def manova_contrastsR(data_fp = 'data/PCs.tsv',
                    formula = 'cbind(PC1, PC2, PC3)~ Group'):
    tmpfile='.tmpvacou'
    r("pcs <- read.csv('"+data_fp+"', sep='\\t')")

    r('ma <- manova('+formula+', data=pcs, contrasts = contrasts(pcs$Group))')
    call = 'ma <- manova('+formula+', data=pcs)'

    r("capture.output(summary(ma), file='"+tmpfile+"')")
    with open(tmpfile, "r") as f:
        o = {}
        o['summary'] = []
        o['summary.aov'] = []
        o['summary'].append(call)
        o['summary'].append('\n')
        o['summary.aov'].append(call)
        o['summary.aov'].append('\n')
        for i,l in enumerate(f.readlines()):
            o['summary'].append(l.rstrip())

    r("capture.output(summary.aov(ma), file='"+tmpfile+"')")
    with open(tmpfile, "r") as f:
        for i,l in enumerate(f.readlines()):
            o['summary.aov'].append(l.rstrip())
    return o#, call


def manova_py():
    #https://www.coursera.org/learn/machine-learning-data-analysis/lecture/XJJz2/running-a-k-means-cluster-analysis-in-python-pt-2
    pass

def compare_groups():
    '''
    https://www.coursera.org/learn/machine-learning-data-analysis/lecture/XJJz2/running-a-k-means-cluster-analysis-in-python-pt-2

    Significance for group difference.
    1)  a) PCA, take all coponents (but can assume independence - ?; what about normality?)
        b) Factorial analysis of variance
            >1 groups, >1 measurements; requires independent variables
            main effects, interaction effects

    2) (M)ANOVA?
        http://www.math.wustl.edu/~victor/classes/ma322/r-eg-12.txt
        a) test for outliers (susceptible) - at least graphically
            R > mshapiro.test() from mvnormtest package - test for multivariate normality
            treating nvasc coupling variable as x, too?
    '''
    pass


def Hotelling():
    pass



def cluster_separation(PCs):
    '''
    PCA - top 3 PCs
    Mahalanobis dist / Jaccard?
    Hotelling's T2
    return p-values
    '''
    dist = pd.DataFrame(
                pairwise_distances(PCs.T, metric='mahalanobis'), #TINKER
            columns=PCs.columns, index=PCs.columns)




if __name__ == "__main__":
    data_fp = '/P/vacou/data_ss31.tsv'




