import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.stats import norm


class ConvergenceAnalysis(object):
    def __init__(self, df, verbose = True, r = 0):
        self.verbose = verbose
        self.df = df
        self.t_df = pd.concat([df[i] for i in df.columns if any((df[i].dtype == np.float, df[i].dtype == np.int))], axis=1)
        df = self.t_df
        
        if verbose:
            print("Version 1.0.0")
            print("Welcome. This Class works for ConvergenceAnalysis.")
            print("You can access dataframe by these METHODS")
            print("  : df, t_df")

        self.mean = df.apply(np.mean) # Series
        self.mean.name = 'mean'
        self.var = df.apply(np.var) # Series
        self.std = df.apply(np.std) # Series
        self.std_s = np.std(df, ddof=1) # Series
        self.cv = self.std / self.mean # Series
        self.cv_s = self.std_s / self.mean # Series
        self.sigma = self.cv / self.cv.values[0] # Series
        self.sigma.name = 'sigma'
        self.df_rank = df.rank(ascending=False, method="min")

        self.rank_var = self.df_rank.apply(np.var) # Series

        self.df_gamma = pd.DataFrame(self.df_rank.values + self.df_rank.values.T[0].reshape(-1,1), columns = self.df_rank.columns)

        self.gamma_var = self.df_gamma.apply(np.var) # Series
        self.gamma = self.gamma_var / self.rank_var.values[0] / 4
        self.gamma.name = 'gamma'
        if verbose:
            print("also, You can access as pd.Series by these METHODS")
            print("  : mean, var, std, cv, sigma, std_s, cv_s (_s means, \"sample\")")
            print("    rank_var, gamma_var, gamma")
            print("and You can access as pd.DataFrame by thes METHODS")
            print("  : df_rank, df_gamma")

        self.period = int(df.columns[-1]) - int(df.columns[0])

        if r:
            self.mean_cagr = round((math.pow(self.mean.values[-1] / self.mean.values[0], 1 / self.period)-1), r)
            self.sigma_cagr = round((math.pow(self.sigma.values[-1] / self.sigma.values[0], 1 / self.period)-1), r)
            self.gamma_cagr = round((math.pow(self.gamma.values[-1] / self.gamma.values[0], 1 / self.period)-1), r)
        else:
            self.mean_cagr = math.pow(self.mean.values[-1] / self.mean.values[0], 1 / self.period)-1
            self.sigma_cagr = math.pow(self.sigma.values[-1] / self.sigma.values[0], 1 / self.period)-1
            self.gamma_cagr = math.pow(self.gamma.values[-1] / self.gamma.values[0], 1 / self.period)-1
        if verbose:
            print("Finally, You can get three CAGR by these METHODS")
            print("  : mean_cagr, sigma_cagr, gamma_cagr")

        self.dof = len(df)-1
        if verbose:
            print("\nThis Class has 4 Functions following as")
            print("  : t_test(alpha=0.05, r=0), chi_square_test(r=0)")
            print("    get_measure_table(name, measure), get_index_table(name)")
            
        self.cagr_year = (self.mean.index[0], self.mean.index[-1])

    def t_test(self, alpha = 0.05, r = 0):
        df = self.t_df
        self.alpha = alpha
        self.z_crit = norm.ppf(1-alpha/2)
        self.z = pd.Series(index=list(df.columns))
        self.p_value = pd.Series(index=list(df.columns))
        self.t_lower = pd.Series(index=list(df.columns))
        self.t_upper = pd.Series(index=list(df.columns))

        for i in df.columns:
            if i == df.columns[0]:
                sd = self.cv_s.values[0]
                self.z[i] = np.NaN
                self.p_value[i] = np.NaN
                self.t_lower[i] = np.NaN
                self.t_upper[i] = np.NaN
            else :
                cp = self.cv_s[i]
                v_pool = (sd + cp) / 2
                new_mean = sd - cp
                new_stdev = np.sqrt(2*(v_pool**2)*(v_pool**2+0.5)/self.dof)
                self.z[i] = new_mean / new_stdev
                self.p_value[i] = 2*(1-norm.cdf(self.z[i]))
                d = np.sqrt((sd *(sd+0.5)+cp*(cp+0.5))/self.dof)
                self.t_lower[i] = new_mean - self.z_crit * d
                self.t_upper[i] = new_mean + self.z_crit * d
        if r:
            self.z = self.z.map(lambda x:round(x,r))
            self.p_value = self.p_value.map(lambda x:round(x,r))
            self.t_lower = self.t_lower.map(lambda x:round(x,r))
            self.t_upper = self.t_upper.map(lambda x:round(x,r))
        if self.verbose:
            print("Do T-Test. Get a P-value.")
            print("If you want, You can get statistic values by these METHODS")
            print("  : alpha, z_crit, dof(degree of freedom)")
            print("And You can get pd.Series by these METHODS")
            print("  : z, p_value, t_lower, t_upper")

    def chi_square_test(self, r=0):
        df = self.df_rank
        
        self.W1 = pd.Series(index=list(df.columns))
        self.r = pd.Series(index=list(df.columns))
        self.chi_squared = pd.Series(index=list(df.columns))
        self.chi_p_value = pd.Series(index=list(df.columns))
        self.W2 = pd.Series(index=list(df.columns))
        
        k = len(self.df)
        m = 2
        for i in df.columns:
            if i == df.columns[0]:
                sd_yr = df[i]
                self.chi_p_value[i] = np.NaN
                self.W1[i] = np.NaN
            else:
                sum_beg_yr = sd_yr + df[i]
                DEVSQ = sum((sum_beg_yr - np.mean(sum_beg_yr))**2)
                self.W1[i] = 12*DEVSQ/((m**2)*(k**3-k))
                self.r[i] = (m * self.W1[i] - 1) / (m-1)
                self.chi_squared[i] = m * (k-1) * self.W1[i]
                self.chi_p_value[i] = stats.chi2.pdf(self.chi_squared[i], self.dof)
                SUMEQ = sum(sum_beg_yr**2)
                self.W2[i] = 12*SUMEQ/((m**2)*(k**3-k)) - 3*(k+1)/(k-1)
        if r:
            self.r = self.r.map(lambda x:round(x,r))
            self.chi_squared = self.chi_squared.map(lambda x:round(x,r))
            self.chi_p_value = self.chi_p_value.map(lambda x:round(x,r))
        if self.verbose:
            print("Do Chi-Square Test. Get a P_value.")
            print("You can get pd.Series by these METHODS")
            print("  : W1, r, chi_squared, chi_p_value, W2")

    def get_measure_table(self, name=False, measure=False):
        tu = [["{}".format(name)], ["{}".format(measure)]]
        tuples = list(zip(*tu))
        index = pd.MultiIndex.from_tuples(tuples)
        table = pd.DataFrame(columns=index, index=self.mean.index)
        table[("{}".format(name), "{}".format(measure))] = ["%.2f" % i for i in round(self.mean, 2).values]
        table.loc["CAGR"] = "%.2f" % (self.mean_cagr*100) + "%"
        return table

    def get_index_table(self, name=False):
        tu = [["{}".format(name),"{}".format(name),"{}".format(name),"{}".format(name)], \
              ["sigma", "p_value", "gamma", "chi_p_value"]]
        tuples = list(zip(*tu))
        index = pd.MultiIndex.from_tuples(tuples)
        table = pd.DataFrame(columns=index, index=self.sigma.index)
        table[("{}".format(name), "sigma")] = ["%.4f" % i for i in round(self.sigma, 4).values]
        table[("{}".format(name), "p_value")] = np.where(self.p_value<0.01, "***", \
                                                np.where(self.p_value<0.05, "**", np.where(self.p_value<0.1, "*", "")))
        table[("{}".format(name), "gamma")] = ["%.4f" % i for i in round(self.gamma, 4).values]
        table[("{}".format(name), "chi_p_value")] = np.where(self.chi_p_value<0.01, "***", \
                                                np.where(self.chi_p_value<0.5, "**", np.where(self.chi_p_value<0.1, "*", "")))
        table.loc["CAGR"] = ["%.2f" % (self.sigma_cagr*100) + "%", "", "%.2f" % (self.gamma_cagr*100) + "%", ""]

        return table

    def get_index_table2(self, name=False):
        tu = [["{}".format(name),"{}".format(name)], ['Sigma', 'Gamma']]
        tuples = list(zip(*tu))
        index = pd.MultiIndex.from_tuples(tuples)
        table = pd.DataFrame(columns=index, index=self.sigma.index)
        f = lambda x : '{:.4f}'.format(x)
        h = lambda x : np.where(x<0.01, '***', 
                        np.where(x<0.05, '**', 
                         np.where(x<0.1, '*', '')))
        table[("{}".format(name), "Sigma")] = self.sigma.map(f) + self.p_value.map(h)
        table[("{}".format(name), "Gamma")] = self.gamma.map(f) + self.chi_p_value.map(h)
        table.loc["CAGR"] = ["%.2f" % (self.sigma_cagr*100) + "%", "%.2f" % (self.gamma_cagr*100) + "%"]
        return table
    
    def calc_cagr(self, start, end, measure='mean'):
        if measure.upper()=='SIGMA':
            sr = self.sigma.loc[start:end]
        elif measure.upper()=='GAMMA':
            sr = self.gamma.loc[start:end]
        else:
            sr = self.mean.loc[start:end]
        return math.pow(sr.values[-1] / sr.values[0], 1 / (len(sr)-1))-1
    
    def reset_cagr(self, start, end):
        """
        Caution!
        이 함수는 CAGR을 건듭니다.
        """
        self.cagr_year = (start, end)
        self.mean_cagr = self.calc_cagr(start, end, 'mean')
        self.sigma_cagr = self.calc_cagr(start, end, 'Sigma')
        self.gamma_cagr = self.calc_cagr(start, end, 'Gamma')
        print('Change cagr_year : {}~{}'.format(*self.cagr_year))
        
    
    
"""
20190214 수정
"""