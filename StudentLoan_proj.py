# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 09:13:00 2018

@author: ericg
"""
#==============================================================================
# Load/Inspect Data
#==============================================================================
#%%Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFECV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import stats
from scipy.stats import randint as sp_randint
#%% Import data
df=pd.read_csv('train_values.csv',
               index_col='row_id')

df1=pd.read_csv('train_labels.csv',
                index_col='row_id')

dft=pd.read_csv('test_values.csv',
                index_col='row_id')

#%% check train data
df.info()
df.head()

#%% check train labels
df1.info()
df1.head()

#%%plot histogram of repayment_rate
sns.distplot(df1.repayment_rate, bins=20)
plt.title('Distribution of Repayment Rate')

#==============================================================================
# Process Data
#==============================================================================

#%% Concatenate train_values and test_values
df_comb=pd.concat([df, dft],
                  axis='rows')

#%% view info of combined data set
df_comb.info()

#%%  count of missing in combined data set
df_comb.isnull().sum().plot(kind='hist', bins=20)

#%% remove features with >10% missing
df_comb_NAfilter = df_comb.dropna(axis=1, thresh=13500)

#%% info
df_comb_NAfilter.info()

###################
''' Categorical '''
###################
#%% inspect objects
df_comb_NAfilter.describe(include=[np.object])

#%% create data frame containing only object class
df_obj=df_comb_NAfilter.select_dtypes(['object'])

#%%display object features to be converted to category
df_obj.describe()
df_obj.head()

#%% convert object features to categorical
df_cat=df_obj.apply(pd.Series.astype, dtype='category')
df_cat.info()

#%%check missin value count 
df_cat.isnull().sum()

#%%list of categorical features to be removed
drop_cols=['report_year',
           'school__online_only',
           'school__religious_affiliation', 
           'school__state',
           'school__main_campus',
           'school__degrees_awarded_predominant']

#%%drop select categorical columns
df_cat.drop(drop_cols, axis=1, inplace=True)

#%%fill missing value in region_id with mode
df_cat['school__region_id'].fillna(df_cat['school__region_id'].mode()[0], inplace=True)

#%% generate dummy variables for remaining categorical features
df_catDum=pd.get_dummies(df_cat, 
                         columns=['school__degrees_awarded_highest', 
                                  'school__institutional_characteristics_level', 
                                  'school__ownership',
                                  'school__region_id'],
                                  prefix=['degree',
                                          'level',
                                          'ownership',
                                          'region'], 
                                  drop_first=True)

#%% info and head of categorical df
df_catDum.info()
df_catDum.head()

###############
 '''Numeric'''
###############

#%% missing values in numeric
df_num=df_comb_NAfilter.select_dtypes(['float', 'int64'])

#%%Plot missing values in numeric df
df_num.isnull().sum().plot(kind='hist')

#%%impute missing numeric values with mean 
df_numImp=df_num.fillna(df_num.mean())
df_numImp.isnull().sum().sum()

#########################
''' Feature Selection '''
#########################

##-Variance Threshold-##
#%% remove features with <5% variance
vt=VarianceThreshold(threshold=0.05)
vt.fit(df_numImp)

#%%select out chosen features based on variance threshold
df_numImpvar=df_numImp.iloc[:,vt.get_support()]
df_numImpvar.info()

#%% combine transformed categorical and numeric dataframes
df_combTsf=pd.concat([df_catDum, df_numImpvar], axis='columns')
df_combTsf.info()  #removes 125 features

#%%split back into train and test dataframes
df_trainTsf=df_combTsf[:8705]
df_testTsf=df_combTsf[8705:]

#%%inspect transformed train set
df_trainTsf.info()
df_trainTsf.head()

#%%inspect transformed test set
df_testTsf.info()
df_testTsf.head()

#%% append train values to train set
trainTsf=pd.concat([df_trainTsf, df1], axis='columns')
y=trainTsf['repayment_rate'].copy()
X=trainTsf.drop('repayment_rate', axis='columns')

##-RFE-##

#%% 
estimator = RandomForestRegressor()
selector = RFECV(estimator,step=1, scoring='neg_mean_absolute_error', cv=3)
selector=selector.fit(X, y)
print("Optimal number of features : %d" % selector.n_features_)
#Optimal number of features : 178

#%%dataframe of features selected by RFECV
trainRFE=X.iloc[:,selector.get_support()]
trainRFE.info()

#%%Create train and test set from
X_rfe=trainRFE.drop('repayment_rate', axis='columns')

#%% 

skb = SelectKBest(score_func=f_regression)

et_reg = ExtraTreesRegressor(n_jobs=-1,
                             random_state=123,
                             n_estimators=1500)

k=sp_randint(20,100)
n_iter=10

steps=[('skb',skb),
       ('et_reg', et_reg)]

pipeline=Pipeline(steps)
parameters=dict(skb__k=k)

rand_grid = RandomizedSearchCV(pipeline,
                               param_distributions=parameters,
                               n_iter=n_iter,
                               scoring='neg_mean_absolute_error',
                               cv=3,)

rand_grid.fit(trainRFE, y)
''' {'skb__k': 90} '''
#%%
train_kbest=trainRFE.iloc[:, skb.get_support()]

#%% Sample data set
#trainTsf_3K = trainTsf.sample(n=3000, random_state=7811)
#y=trainTsf_3K['repayment_rate'].copy()
#X=trainTsf_3K.drop('repayment_rate', axis=1)

#==============================================================================
# EDA
#==============================================================================
#%%
sns.color_palette('deep')
sns.set_context("poster")
sns.set_style("ticks")

###############
# Categorical
##############

#%%
catTrain=df_cat[:8705]
catPlot=pd.concat([catTrain, df1], axis='columns')

#%% Repayment rate vs school ownership
ax = sns.barplot(x='school__ownership',
                 y='repayment_rate',
                 data=catPlot)

#%% Repayment rate vs school region id
ax = sns.barplot(x='school__region_id',
                 y='repayment_rate',
                 data=catPlot)

#%% Repayment rate vs school__degrees_awarded_highest
ax = sns.barplot(x='school__degrees_awarded_highest',
                 y='repayment_rate',
                 data=catPlot)

#%% Repayment rate vs school__institutional_characteristics_level
sns.barplot(x='school__institutional_characteristics_level',
            y='repayment_rate',
            data=catPlot)

############
# Numeric
############

#==============================================================================
# Build predictive models
#==============================================================================

#%%
X_train, X_test, y_train, y_test = train_test_split(trainRFE, y, test_size=0.3, random_state=7811)




#%%

skb = SelectKBest(score_func=f_regression)

et_reg = ExtraTreesRegressor(n_jobs=-1,
                             random_state=7811,
                             n_estimators=10)

k=sp_randint(20,100)
n_iter=10

steps=[('skb',skb),
       ('et_reg', et_reg)]

pipeline=Pipeline(steps)
parameters=dict(skb__k=k)

rand_grid = RandomizedSearchCV(pipeline,
                               param_distributions=parameters,
                               n_iter=n_iter,
                               scoring='neg_mean_absolute_error',
                               cv=3,)

rand_grid.fit(X_train, y_train)

#%%
y_pred = rand_grid.predict(X_test)
print(rand_grid.score(X_test, y_test))

#%%
seed=7811
model = ExtraTreesRegressor(random_state=seed,
                             n_estimators=100)

scoring='neg_mean_squared_error'
kfold=KFold(n_splits=10, random_state=seed)
results=cross_val_score(model,
                      trainRFE,
                      y, 
                      cv=kfold,
                      scoring=scoring)
#%%
mse_scores= -results
rmse_scores=np.sqrt(mse_scores)
print(rmse_scores.mean())