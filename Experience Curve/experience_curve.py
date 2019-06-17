import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm

import sympy as sym

def get_kinked_year(df, yr_col, ln_x, ln_y):
    df = df.set_index(yr_col)
    y = df[ln_y]
    r_sq = []
    for idx, year in enumerate(df.index.values):
        if idx > 1:
            d = pd.Series(np.where(df.index.values < year, 0, 1), 
                          index=df.index)
            dln_x = d * df[ln_x]
            X = pd.concat((df[ln_x], d, dln_x), axis=1)
            X = sm.add_constant(X)
            r_sq.append(smf.OLS(y, X).fit().rsquared)
        else:
            r_sq.append(0)
    kinked_year = df.index[np.argmax(r_sq)]
    return kinked_year, r_sq    
    
f = lambda x : "{:.4f}".format(x)
g = lambda x : np.where(x<0.01,'***',np.where(x<0.05,'**',np.where(x<0.1,'*','')))
h = lambda x : "\n({:.3f})".format(x)

def format_params(model, coef):
    return '{:.4f}'.format(model.params[coef])\
            +g(model.pvalues[coef]).tolist()+'\n'\
            +'({:.3f})'.format(model.bse[coef])

def format_r_2(model):
    return '{:.4f}'.format(model.rsquared)

def format_pr(model, coef):
    return '{:.4f}'.format(2**model.params[coef])

def model_select(model2, model3):
    return ['Kinked' if (model2.pvalues[3] < 0.1)&(model3.pvalues[1] < 0.1)
                    else 'Classical'][0]

def get_exp_curve_table(ind):
    array = [
        ['Classical experience Eq','Classical experience Eq',
         'Classical experience Eq','Classical experience Eq',
         'Kinked','Kinked experience Eq','Kinked experience Eq',
         'Kinked experience Eq','Kinked experience Eq',
         'Kinked experience Eq','Kinked experience Eq',
         'Kinked experience Eq','Model'],
        ['log a','b','R_sq','PR','Year','log a1','b1','log a2','b2',
         'b2-b1','R_sq','PR2','selection']
    ]
    col = pd.MultiIndex.from_arrays(array)
    table = pd.DataFrame(columns=col,
                        index=ind)
    return table

def add_table_row(table, country, region, kinked_year, model1, model2, model3):
    table.loc[country, region] = [
        format_params(model1,0), # log a
        format_params(model1,1), # b
        format_r_2(model1),      # R2
        format_pr(model1,1),     # PR
        kinked_year[(country, region)],    # Kinked Year
        format_params(model2,0), # log a1
        format_params(model2,1), # b1
        format_params(model3,0), # log a2
        format_params(model3,1), # b2
        format_params(model2,3), # b2 - b1
        format_r_2(model2),      # R2
        format_pr(model3,1),     # PR2
        model_select(model2, model3)
#         model_select(model2,3)   # Model Selection (Either Classical or Kinked)
    ]