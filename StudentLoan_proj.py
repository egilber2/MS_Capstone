# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 09:13:00 2018

@author: ericg
"""
#==============================================================================
# Load/Inspect Data
#==============================================================================
#%%Import packages
#Import Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFECV, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Normalizer, Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import stats as st
from scipy.stats import randint as sp_randint
from scipy.stats import beta
from scipy.special import log1p
from xgboost import XGBRegressor as xgb
from sklearn.externals import joblib

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
#histogram of 'label'#
sns.color_palette(palette='deep')
sns.set_context("notebook")
sns.set_style("ticks")
sns.distplot(df1.repayment_rate, bins=20)
plt.title('Distribution of Repayment Rate');


#==============================================================================
# Process Data
#==============================================================================


#%% #Combine the training and testing data for preprocessing 
df_comb=pd.concat([df, dft],
                  axis='rows')

#%% view info of combined data set
df_comb.info()

#%%  plot of missing counts in combined data set
df_comb.isnull().sum().plot(kind='hist', bins=20);

#%% remove features with >10% missing
df_comb_NAfilter = df_comb.dropna(axis=1,
                                  thresh=12100)

#%% info
df_comb_NAfilter.info()

#%%

###################
# Categorical 
###################


# create data frame containing only object class
df_obj=df_comb_NAfilter.select_dtypes(['object'])

#%%display object features to be converted to category
df_obj.describe()
df_obj.head()

#%% convert object features to categorical
df_cat=df_obj.apply(pd.Series.astype,
                    dtype='category')

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
df_cat.drop(drop_cols,
            axis=1,
            inplace=True)


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

#%% 

###############
# Numeric
###############


#missing values in numeric
df_num=df_comb_NAfilter.select_dtypes(['float', 'int64'])

#%%Plot missing values in numeric df
df_num.isnull().sum().plot(kind='hist')

#%%impute missing numeric values with mode 

fill_NaN=Imputer(strategy='most_frequent', axis=0)
imputed_num=pd.DataFrame(fill_NaN.fit_transform(df_num))
imputed_num.columns=df_num.columns
imputed_num.index=df_num.index

#%%
imputed_num.isnull().sum().sum()
imputed_num.info()
        
#%%
imputed_num.select_dtypes(['float64']).skew().hist(bins=30);

#%%
df_trainTsf=pd.concat([df_catDum, imputed_num], axis='columns')

#%%

# =============================================================================
# Feature Selection
# =============================================================================

#####################
# Variance Threshold
#####################

#remove features with <5% variance
vt=VarianceThreshold(threshold=0.05)
vt.fit(imputed_num)

#%%select out chosen features based on variance threshold
df_numImpvar=imputed_num.iloc[:,vt.get_support()]
df_numImpvar.info()

#%% combine transformed categorical and numeric dataframes
df_combTsf=pd.concat([df_catDum, df_numImpvar], axis='columns')
df_combTsf.info()  #removes 185 features

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


#%%

#########
# RFE
#########

estimator = GradientBoostingRegressor(random_state=7811)
selector = RFECV(estimator,step=1, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
selector=selector.fit(X, y)
print("Optimal number of features : %d" % selector.n_features_)
#Optimal number of features : 93- fillna 'most_frequent'

#%%
trainRFE_gboost=X.iloc[:,selector.get_support()]

#%%
trainRFE_gboost.to_csv('trainRFE_gboost.csv')

#%%
X_train, X_test, y_train, y_test = train_test_split(trainRFE_gboost,
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=7811)

#%%
estimator.fit(X_train, y_train)

#%%
y_pred = estimator.predict(X_test)

mse_scores= mean_squared_error(y_test, y_pred)
rmse_scores=np.sqrt(mse_scores)
print(rmse_scores.mean())

#%%
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(estimator.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.to_csv('importantRFE_gboost')

#%%

#==============================================================================
# EDA
#==============================================================================


sns.color_palette('deep')
sns.set_context("poster")
sns.set_style("ticks")

#%%

#####################
# Feature Importance
#####################


#Read in ranked feature importances from RFE estimator
importances=pd.read_csv('importantRFE_gboost')
importances = importances.sort_values('importance',ascending=False).set_index('feature')

#%%
#Have a look at top 10 features ranked by importance
importances.info()
importances.head(n=10)

#%%
importances=pd.read_csv('importantRFE_gboost')
importances = importances.sort_values('importance',ascending=True).set_index('feature')
importances.plot(kind='barh', figsize=(12,30));

#%%

###################
# Numeric Features-EDA
###################

#read in data selected by RFECV 
trainRFE_gboost=pd.read_csv('trainRFE_gboost.csv', index_col='row_id')

#%%
trainRFE_full=pd.concat([trainRFE_gboost, df1], axis='columns')

#%%
trainRFE_full.to_csv('trainRFE_full.csv', index='row_id')
trainRFE_full.info()

#%%
trainRFE_full.select_dtypes(['float64']).skew().hist(bins=30);


#%%
#####################
# Categorical-EDA
#####################
catRFE=trainRFE_full.select_dtypes(['int64'])
catRFE_plot=pd.concat([catRFE, df1], axis='columns')

#%%
catRFE_plot.head()
catRFE_plot.info()

#%%
for c in catRFE_plot.columns[:8]:
    sns.boxplot(x=catRFE_plot[c], y=catRFE_plot['repayment_rate'])
    plt.show()
    
#%% Repayment rate vs school ownership
ax = sns.barplot(x='school__ownership',
                 y='repayment_rate',
                 data=catRFE_plot)

#%% Repayment rate vs school region id
ax = sns.barplot(x='school__region_id',
                 y='repayment_rate',
                 data=catRFE_plot)

#%% Repayment rate vs school__degrees_awarded_highest
ax = sns.barplot(x='school__degrees_awarded_highest',
                 y='repayment_rate',
                 data=catRFE_plot)


#%% Repayment rate vs school__institutional_characteristics_level
sns.barplot(x='school__institutional_characteristics_level',
            y='repayment_rate',
            data=catRFE_plot)
#%%

#-Correlation heatmap
corr=X_sfm_40.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#%%
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

#%%
# Draw the heatmap with the mask and correct aspect ratio
sns.set_context('poster')
g=sns.heatmap(corr, cmap=cmap, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Heatmap of Selected Features', fontweight='bold',fontsize=40)
plt.rcParams["axes.labelsize"] = 50



#%%
#==============================================================================
# Build predictive models
#==============================================================================

trainRFE_full=pd.read_csv('trainRFE_full.csv', index_col='row_id')

#%%

y=trainRFE_full['repayment_rate'].copy()
X=trainRFE_full.drop('repayment_rate', axis='columns')

#%%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.3,
                                                    random_state=7811)

#%%
#Extra Trees Regressor
et_reg=ExtraTreesRegressor(n_jobs=-1, random_state=7811)

steps=[('et_reg', et_reg)]
pipeline=Pipeline(steps)

n_estimators=sp_randint(20,1700)
n_iter=10

parameters=dict(et_reg__n_estimators=n_estimators)

rand_grid= RandomizedSearchCV(pipeline,
                              param_distributions=parameters,
                              n_iter=n_iter,
                              cv=5,
                              scoring='neg_mean_squared_error')



#%%
rand_grid.fit(X_train, y_train)

#%%
y_pred = rand_grid.predict(X_test)

mse_scores= mean_squared_error(y_test, y_pred)
rmse_scores=np.sqrt(mse_scores)
print(rmse_scores.mean())
print (rand_grid.best_params_)

#%%
y_pred = est.predict(X_test)
print(est.score(X_test, y_test))

#%%


#%%




#%%



#%%
seed=7811
n_est=sp_randint(10, 1500)
n_iter=10
scoring='neg_mean_squared_error'
et_reg=ExtraTreesRegressor()

parameters=dict(n_estimators=n_est)
model = RandomizedSearchCV(estimator=et_reg,
                           param_distributions=parameters,
                           n_iter=n_iter,
                           scoring=scoring)

kfold=RepeatedKFold(n_splits=10, n_repeats=5)
results=cross_val_score(model,
                      trainRFE,
                      y, 
                      cv=kfold,
                      scoring=scoring,
                      n_jobs=-1)  #AWS EC2 Instance
#%%
mse_scores= -results
rmse_scores=np.sqrt(mse_scores)
print(rmse_scores.mean())