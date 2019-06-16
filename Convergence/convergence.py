import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.stats import norm

def load_excel(filename):
    try:
        if filename.split('.')[-1] == 'csv':
            res = pd.read_csv(filename)
        elif filename.split('.')[-1] == 'xlsx':
            res = pd.read_excel(filename)
        else:
            print('Error: csv나 xlsx 파일을 넣어주세요')
            res = None
    except:
        print('Error: input_data 파일에 데이터를 넣었는지 확인해주세요')
        return None
    return res

class ConvergenceAnalysis(object):

    def __init__(self, df, income=False, region=False, month=False):
        self.error = False
        self.df = df
        self.income = income
        self.region = region
        self.t_df = pd.concat(
            [df[i] for i in df.columns
             if any((df[i].dtype == np.float, df[i].dtype == np.int))],
            axis=1)
        if income:
            try:
                df[income]
            except:
                print('Error: 정확한 income 변수명을 넣어주세요.')
                self.error = True
                return None
            self.income_df = {
                _income : pd.concat(
                [_df[i] for i in _df.columns
                 if any((_df[i].dtype == np.float, _df[i].dtype == np.int))],
                axis=1
                )
                for _income, _df in df.groupby(income)
            }

        if region:
            try:
                df[region]
            except:
                print('Error: 정확한 Region 변수명을 넣어주세요.')
                self.error = True
                return None
            self.region_df = {
                _region : pd.concat(
                    [_df[i] for i in _df.columns
                     if any((_df[i].dtype == np.float, _df[i].dtype == np.int))],
                    axis=1
                    )
                for _region, _df in df.groupby(region)
            }

        self.mean = self.t_df.apply(np.mean)
        self.mean.name = 'mean'
        self.var = self.t_df.apply(np.var)
        self.std = self.t_df.apply(np.std)
        self.std_s = np.std(self.t_df, ddof=1)
        self.cv = self.std / self.mean
        self.cv_s = self.std_s / self.mean
        self.sigma = self.cv / self.cv.values[0]
        self.sigma.name = 'sigma'
        self.df_rank = self.t_df.rank(ascending=False, method="min")
        self.rank_var = self.df_rank.apply(np.var)
        self.df_gamma = pd.DataFrame(
            self.df_rank.values + self.df_rank.values.T[0].reshape(-1, 1),
            columns = self.df_rank.columns)
        self.gamma_var = self.df_gamma.apply(np.var)
        self.gamma = self.gamma_var / self.rank_var.values[0] / 4
        self.gamma.name = 'gamma'
        if month:
            self.period = len(self.t_df.columns) - 1
        else:
            self.period = int(self.t_df.columns[-1]) - int(self.t_df.columns[0])
        self.mean_cagr = math.pow(self.mean.values[-1] / self.mean.values[0],
                                  1 / self.period) - 1
        self.sigma_cagr = math.pow(self.sigma.values[-1] / self.sigma.values[0],
                                  1 / self.period) - 1
        self.gamma_cagr = math.pow(self.gamma.values[-1] / self.gamma.values[0],
                                  1 / self.period) - 1
        self.dof = len(self.t_df) - 1
        self.p_value = None
        self.chi_p_value = None

        if (self.income != False) | (self.region != False):
            self.calc_sub_convergence()

    def calc_sub_convergence(self):
        if self.income:
            self.income_mean = {}
            self.income_sigma = {}
            self.income_gamma = {}
            self.income_cagr = {}
            for _income, df in self.income_df.items():
                mean = df.apply(np.mean)
                mean.name = 'mean'
                self.income_mean[_income] = mean
                std = df.apply(np.std)
                cv = std / mean
                sigma = cv / cv.values[0]
                sigma.name = 'sigma'
                self.income_sigma[_income] = sigma
                df_rank = df.rank(ascending=False, method="min")
                rank_var = df_rank.apply(np.var)
                df_gamma = pd.DataFrame(
                    df_rank.values + df_rank.values.T[0].reshape(-1, 1),
                    columns = df_rank.columns)
                gamma_var = df_gamma.apply(np.var)
                gamma = gamma_var / rank_var.values[0] / 4
                gamma.name = 'gamma'
                self.income_gamma[_income] = gamma
                cagr = {}
                cagr['mean'] = math.pow(mean.values[-1] / mean.values[0],
                                      1 / self.period) - 1
                cagr['sigma'] = math.pow(sigma.values[-1] / sigma.values[0],
                                      1 / self.period) - 1
                cagr['gamma'] = math.pow(gamma.values[-1] / gamma.values[0],
                                      1 / self.period) - 1
                self.income_cagr[_income] = cagr
        if self.region:
            self.region_mean = {}
            self.region_sigma = {}
            self.region_gamma = {}
            self.region_cagr = {}
            for _region, df in self.region_df.items():
                mean = df.apply(np.mean)
                mean.name = 'mean'
                self.region_mean[_region] = mean
                std = df.apply(np.std)
                cv = std / mean
                sigma = cv / cv.values[0]
                sigma.name = 'sigma'
                self.region_sigma[_region] = sigma
                df_rank = df.rank(ascending=False, method="min")
                rank_var = df_rank.apply(np.var)
                df_gamma = pd.DataFrame(
                    df_rank.values + df_rank.values.T[0].reshape(-1, 1),
                    columns = df_rank.columns)
                gamma_var = df_gamma.apply(np.var)
                gamma = gamma_var / rank_var.values[0] / 4
                gamma.name = 'gamma'
                self.region_gamma[_region] = gamma
                cagr = {}
                cagr['mean'] = math.pow(mean.values[-1] / mean.values[0],
                                      1 / self.period) - 1
                cagr['sigma'] = math.pow(sigma.values[-1] / sigma.values[0],
                                      1 / self.period) - 1
                cagr['gamma'] = math.pow(gamma.values[-1] / gamma.values[0],
                                      1 / self.period) - 1
                self.region_cagr[_region] = cagr

    def t_test(self, alpha=.05, df=None):
        if df is not None:
            dof = len(df) - 1
            pass
        else:
            df = self.t_df
            dof = self.dof
        z_crit = norm.ppf(1-alpha/2)
        z = pd.Series(index=list(df.columns))
        p_value = pd.Series(index=list(df.columns))
        t_lower = pd.Series(index=list(df.columns))
        t_upper = pd.Series(index=list(df.columns))

        for i in df.columns:
            if i == df.columns[0]:
                cv_s = np.std(df, ddof=1) / df.apply(np.mean)
                sd = cv_s.values[0]
                z[i] = np.NaN
                p_value[i] = np.NaN
                t_lower[i] = np.NaN
                t_upper[i] = np.NaN
            else :
                cp = cv_s[i]
                v_pool = (sd + cp) / 2
                new_mean = sd - cp
                new_stdev = np.sqrt(2*(v_pool**2)*(v_pool**2+0.5)/dof)
                z[i] = new_mean / new_stdev
                p_value[i] = 2*(1-norm.cdf(z[i]))
                d = np.sqrt((sd *(sd+0.5)+cp*(cp+0.5))/dof)
                t_lower[i] = new_mean - z_crit * d
                t_upper[i] = new_mean + z_crit * d

        if self.p_value is None:
            self.p_value = p_value
        else:
            return p_value

        if (self.income != False) | (self.region != False):
            if self.income:
                self.income_p_value = {}
                for _income, _df in self.income_df.items():
                    p_value = self.t_test(df=_df)
                    self.income_p_value[_income] = p_value
            if self.region:
                self.region_p_value = {}
                for _region, _df in self.region_df.items():
                    p_value = self.t_test(df=_df)
                    self.region_p_value[_region] = p_value

    def chi_square_test(self, r=0, df=None):
        if df is not None:
            df = df.rank(ascending=False, method='min')
            dof = len(df) - 1
        else:
            df = self.df_rank
            dof = self.dof

        if dof == 0:
            print('Error: 2개국 미만인 Subgroup 존재')
            self.error = True
            return None

        W1 = pd.Series(index=list(df.columns))
        r = pd.Series(index=list(df.columns))
        chi_squared = pd.Series(index=list(df.columns))
        chi_p_value = pd.Series(index=list(df.columns))
        W2 = pd.Series(index=list(df.columns))

        k = len(df)
        m = 2
        for i in df.columns:
            if i == df.columns[0]:
                sd_yr = df[i]
                chi_p_value[i] = np.NaN
                W1[i] = np.NaN
            else:
                sum_beg_yr = sd_yr + df[i]
                DEVSQ = sum((sum_beg_yr - np.mean(sum_beg_yr))**2)
                W1[i] = 12*DEVSQ/((m**2)*(k**3-k))
                r[i] = (m * W1[i] - 1) / (m-1)
                chi_squared[i] = m * (k-1) * W1[i]
                chi_p_value[i] = stats.chi2.pdf(chi_squared[i], dof)
                SUMEQ = sum(sum_beg_yr**2)
                W2[i] = 12*SUMEQ/((m**2)*(k**3-k)) - 3*(k+1)/(k-1)

        if self.chi_p_value is None:
            self.chi_p_value = chi_p_value
        else:
            return chi_p_value

        if (self.income != False) | (self.region != False):
            if self.income:
                self.income_chi_p_value = {}
                for _income, _df in self.income_df.items():
                    chi_p_value = self.chi_square_test(df=_df)
                    self.income_chi_p_value[_income] = chi_p_value
            if self.region:
                self.region_chi_p_value = {}
                for _region, _df in self.region_df.items():
                    chi_p_value = self.chi_square_test(df=_df)
                    self.region_chi_p_value[_region] = chi_p_value

    def calc_cagr(self, series):
        return math.pow(series.iloc[-1]/series.iloc[0],
                    1/(len(series)-1))-1

    def get_measure_table(self):
        ind = self.mean.index
        col = ['Total' + ' ({})'.format(len(self.t_df))]
        val = np.array(self.mean).reshape(-1,1)
        if (self.income != False) | (self.region != False):
            if self.income:
                for _income, _df in self.income_df.items():
                    col.append('{} ({})'.format(_income, len(_df)))
                    val = np.concatenate((val,
                      np.array(self.income_mean[_income]).reshape(-1,1)),
                                         axis=1)
            if self.region:
                for _region, _df in self.region_df.items():
                    col.append('{} ({})'.format(_region, len(_df)))
                    val = np.concatenate((val,
                      np.array(self.region_mean[_region]).reshape(-1,1)),
                                         axis=1)
        table = pd.DataFrame(data=val, columns=col, index=ind)
        table.loc['CAGR'] = table.apply(self.calc_cagr).map(
            lambda x : '{:.02%}'.format(x))
        table.iloc[:-1] = table.iloc[:-1].applymap(lambda x : '{:.4f}'.format(x))
        return table

    def get_index_table(self):
        ind = self.mean.index
        levels = ['Total' + ' ({})'.format(len(self.t_df))]
        measure = ['Sigma', 'Gamma']
        val = np.concatenate((
            np.array(self.sigma).reshape(-1,1),
            np.array(self.gamma).reshape(-1,1)), axis=1)
        prob = np.concatenate((
            np.array(self.p_value).reshape(-1,1),
            np.array(self.chi_p_value).reshape(-1,1)), axis=1)
        if (self.income != False) | (self.region != False):
            if self.income:
                for _income, _df in self.income_df.items():
                    levels.append('{} ({})'.format(_income, len(_df)))
                    val = np.concatenate((val,
                      np.concatenate((
                          np.array(self.income_sigma[_income]).reshape(-1,1),
                          np.array(self.income_gamma[_income]).reshape(-1,1)),
                          axis=1)), axis=1)
                    prob = np.concatenate((prob,
                      np.concatenate((
                      np.array(self.income_p_value[_income]).reshape(-1,1),
                      np.array(self.income_chi_p_value[_income]).reshape(-1,1)),
                          axis=1)), axis=1)
            if self.region:
                for _region, _df in self.region_df.items():
                    levels.append('{} ({})'.format(_region, len(_df)))
                    val = np.concatenate((val,
                      np.concatenate((
                          np.array(self.region_sigma[_region]).reshape(-1,1),
                          np.array(self.region_gamma[_region]).reshape(-1,1)),
                          axis=1)), axis=1)
                    prob = np.concatenate((prob,
                      np.concatenate((
                      np.array(self.region_p_value[_region]).reshape(-1,1),
                      np.array(self.region_chi_p_value[_region]).reshape(-1,1)),
                          axis=1)), axis=1)
        codes = [[i // 2 for i in range(len(levels)*2)],
                 [0, 1] * len(levels)]
        col = pd.MultiIndex(levels=[levels, measure],
                            labels=codes)
        table = pd.DataFrame(data=val, columns=col, index=ind)
        table2 = pd.DataFrame(data=prob, columns=col, index=ind)
        table.loc['CAGR'] = table.apply(self.calc_cagr).map(
            lambda x : '{:.02%}'.format(x))
        f = lambda x: '{:.4f}'.format(x)
        h = lambda x : np.where(x<0.01, '***',
                np.where(x<0.05, '**',
                 np.where(x<0.1, '*', '')))
        table.iloc[:-1] = table.iloc[:-1].applymap(f) + table2.applymap(h)
        return table

    def swap_index_table(self):
        table = self.get_index_table()
        return table.T.swaplevel().sort_index(ascending=False).T
