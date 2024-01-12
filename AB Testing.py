#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[4]:


df = pd.read_csv('AB_Test_Results.csv')

def perform_EDA(dataframe):
    """
    Perform Exploratory Data Analysis (EDA) on the given dataframe.

    Parameters:
        dataframe (pandas.DataFrame): The dataframe to perform EDA on.

    Prints:
        - Dataset Sample: Displays a sample of the dataset.
        - Dataset Information: Provides information about the dataset.
        - Duplicated Values: Calculates and displays the sum of duplicated values in the dataset.
        - Unique Values: Displays the count of unique values in each column of the dataset.
        - Dataset Description: Provides descriptive statistics of the dataset.

    Returns:
     None
    """
    print("---- Dataset Sample ".ljust(50, '-'))
    print('')
    print(dataframe.sample(n = 10))
    print('') 
    print("---- Dataset Information ".ljust(50, '-'))
    print('')
    print(dataframe.info())
    print('') 
    print("---- Duplicated Values ".ljust(50, '-'))
    print('')
    print('Sum of Duplicated Values:', df.duplicated().sum())
    print('')
    print('') 
    print("---- Unique Values ".ljust(50, '-'))
    print('')
    print(df.nunique())
    print('')
    print("---- Dataset Description ".ljust(50, '-'))
    print('')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print('')
    
perform_EDA(df)


# In[5]:


variant_group = df[df['VARIANT_NAME'] == 'variant'].drop_duplicates(keep = 'first')
control_group = df[df['VARIANT_NAME'] == 'control'].drop_duplicates(keep = 'first')
df = pd.concat([variant_group, control_group], ignore_index = True)
df.reset_index(drop=True, inplace=True)


# In[6]:


conf_int = sms.DescrStatsW(df['REVENUE']).tconfint_mean()

print(f'95% confidence interval for Revenue: {conf_int}')


# In[7]:


def outlier_thresholds(dataframe, variable):
    """
    Calculate the lower and upper outlier thresholds for a given variable in the dataframe.

    Parameters:
        dataframe (pandas.DataFrame): The dataframe containing the variable.
        variable (str): The name of the variable for which outlier thresholds will be calculated.

    Returns:
        tuple: A tuple containing the lower and upper outlier thresholds.
    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit.round(), up_limit.round()
def replace_with_thresholds(dataframe, variable):
    """
    Replace the outliers in the given variable of the dataframe with the lower and upper thresholds.

    Parameters:
        dataframe (pandas.DataFrame): The dataframe containing the variable.
        variable (str): The name of the variable for which outliers will be replaced.

    Returns:
        None  
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
replace_with_thresholds(df, "REVENUE")


# In[8]:


conf_int = sms.DescrStatsW(df['REVENUE']).tconfint_mean()

print(f'95% confidence interval for Revenue: {conf_int}')


# In[9]:


df.groupby('VARIANT_NAME').agg({'REVENUE': 'mean'})


# In[10]:


test_stat, pvalue = shapiro(df.loc[df["VARIANT_NAME"] == "variant", "REVENUE"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["VARIANT_NAME"] == "control", "REVENUE"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# In[11]:


test_stat, pvalue = mannwhitneyu(df.loc[df["VARIANT_NAME"] == "variant", "REVENUE"],
                                 df.loc[df["VARIANT_NAME"] == "control", "REVENUE"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# In[14]:


df = pd.read_csv('ab_data.csv')

perform_EDA(df)


# In[15]:


df.drop_duplicates( subset = 'user_id', keep = 'first', inplace = True)

df.nunique()


# In[16]:


confidence_level = 0.95 # --> The confidence level specified when formulating the hypothesis.
alpha = 1 - confidence_level # --> The significance level or alpha value, representing the maximum risk level for the hypothesis.
power = 0.8 # --> The measure of the probability of not rejecting the null hypothesis when it is false, often set to 0.8 for medical research.
effect_size = 0.02 # --> A small effect size that is considered acceptable for our analysis, set to 0.02.
random_state = 42 # --> A reference to Douglas Adams' number 42, related to the meaning of the universe.

n = sms.NormalIndPower().solve_power(
                                     effect_size = effect_size,
                                     alpha = alpha,
                                     power = power,
                                     ratio = 1
                                    )

sample_control = df.loc[df['group'] == 'control'].sample(n = round(n), random_state = random_state) 
sample_treatment = df.loc[df['group'] == 'treatment'].sample(n = round(n), random_state = random_state)
ab_test = pd.concat([sample_control, sample_treatment], axis = 0).reset_index(drop = True)
ab_test


# In[17]:


perform_EDA(ab_test)


# In[18]:


conversion_rates = ab_test.groupby('group')['converted'].agg(['mean','std','sem'])
conversion_rates.columns = ['conversion_rate', 'std_deviation','std_error']
conversion_rates.style.format('{:.3f}')


# In[19]:


import plotly.graph_objs as go
import plotly.offline as pyo

data = [go.Bar(x=conversion_rates.index, y=conversion_rates['conversion_rate'],
              error_y=dict(type='data', array=conversion_rates['std_error'],visible=True), marker_color = ['#7149C6', '#FC2947'])]

layout = go.Layout(title='Conversion rate by group', xaxis=dict(title='Group'),yaxis=dict(title='Converted (proportion)',
                                                                                        range=[0, 0.17]))

fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)


# In[20]:


c_results = ab_test.loc[ab_test['group'] == 'control', 'converted']
t_results = ab_test.loc[ab_test['group'] == 'treatment', 'converted']


test_stat, pvalue = proportions_ztest([c_results.sum(), t_results.sum()],
                                nobs = [len(c_results), len(t_results)])

conf_int_c = sms.DescrStatsW(c_results).tconfint_mean()
conf_int_t = sms.DescrStatsW(t_results).tconfint_mean()

print(f'Test Stat: {test_stat:.4f}')
print(f'p-value: {pvalue:.4f}')
print(f'95% confidence interval for control group: {conf_int_c}')
print(f'95% confidence interval for treatment group: {conf_int_t}')


# In[ ]:




