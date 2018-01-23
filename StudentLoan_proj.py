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
df_train=pd.read_csv('train_values.csv',
               index_col='row_id')

df_label=pd.read_csv('train_labels.csv',
                index_col='row_id')

df_test=pd.read_csv('test_values.csv',
                index_col='row_id')

#%% check train data
df_train.info()
df_train.head()

#%% check train labels
df_label.info()
df_label.head()

#%%plot histogram of repayment_rate
#histogram of 'label'#
sns.color_palette(palette='deep')
sns.set_context("notebook")
sns.set_style("ticks")
sns.distplot(df_label.repayment_rate, bins=20)
plt.title('Distribution of Repayment Rate');

#%%
#==============================================================================
# Process Data
#==============================================================================
train=df_train.isnull().sum()  #try label for legend?
test=df_test.isnull().sum()
plt.hist(train, bins=20, label='Train data')
plt.hist(test, bins=20, label='Test data', alpha=0.7)
plt.title('Counts of Missing Frequency')
plt.xlabel('Counts of Missing')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show;

#%% remove features with >20% of data missing.
df_train.dropna(axis=1, thresh=6970, inplace=True)

#%% 
df_train.info()

#%%  
###############
# Categorical
###############

#%% 
df_traincat=df_train.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')

#%% #display object features to be converted to category
df_traincat.info()
df_traincat.head()

#%%
#confirm that objects been converted to category and get NA count
df_traincat.isnull().sum()

#%%
cat_cols=list(df_traincat)

#%%#list of categorical features to be removed
drop_cols=['report_year',
           'school__online_only',
           'school__religious_affiliation', 
           'school__state',
           'school__main_campus',
           'school__degrees_awarded_predominant']

#%% #drop select categorical columns
df_traincat.drop(drop_cols, 
                 axis=1,
                 inplace=True)

#%%# fill missing value in region_id with mode
df_traincat['school__region_id'].fillna(df_traincat['school__region_id'].mode()[0], inplace=True)

#%%
#generate dummy variables for remaining categorical features
df_traincat=pd.get_dummies(df_traincat, 
                         columns=['school__degrees_awarded_highest', 
                                  'school__institutional_characteristics_level', 
                                  'school__ownership',
                                  'school__region_id'],
                                  prefix=['degree',
                                          'level',
                                          'ownership',
                                          'region'], 
                                  drop_first=True)

#%%
# info and head of categorical df
df_traincat.info()
df_traincat.head()

#%%
df_traincat.isnull().sum().sum()

#%% 
df_train.drop(cat_cols, axis=1, inplace=True)
df_train.info()

#%% 
df_trainTsf=pd.concat([df_train, df_traincat], axis='columns')
df_trainTsf.to_csv('df_trainTsf.csv')
df_trainTsf.info()

#%% 

###############
# Numeric
###############


# based on redundance of features as seen in feature importances- first pass
drop_cols=['student__demographics_median_family_income']

#%%
df_trainTsf.drop(drop_cols, axis='columns', inplace=True)

#%%
X=df_trainTsf.copy()
# X=df_trainTsf[X_trainRFE.columns]  after feature selection with RFE
y=df_label.copy()

#%%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=7811)
        
#%%Plot missing values in numeric df
train=X_train.isnull().sum() #try label for legend?
test=X_test.isnull().sum()
plt.hist(train, bins=20, label='X_train')
plt.hist(test, bins=20, label='X_test', alpha=0.7)
plt.title('Counts of Missing Values for X_Train and X_test')
plt.xlabel('Counts of Missing')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show();

#%%# instantiate imputer for missing values
imputer=Imputer(strategy='most_frequent', axis=0).fit(X_train)

#%%
X_train_imp=imputer.transform(X_train)

#%%
# turn matrix into df with column names
index=X_train.index

X_train_df=pd.DataFrame(X_train_imp, columns = X_train.columns, index=index)
X_train_df.info() 
X_train_df.head()

#%% 
X_test_imp=imputer.transform(X_test)

#%%
index=X_test.index
X_test_df=pd.DataFrame(X_test_imp, columns=X_test.columns, index=index)

#%%
X_test_df.head()

#%%
# =============================================================================
# Feature Selection
# =============================================================================

##-RFE-##
#%% 
# Gradient Boosting Regressor for RFE
estimator = GradientBoostingRegressor(random_state=7811)

selector = RFECV(estimator,step=1,
                 scoring='neg_mean_squared_error',
                 cv=5,
                 n_jobs=-1)

selector=selector.fit(X_train_df, y_train.values.ravel())
print("Optimal number of features : %d" % selector.n_features_)
#Optimal number of features : 78- fillna 'most_frequent'


#%%
joblib.dump(selector, 'rfe_selector.pkl')

#%%
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()

#%%
RFEselector=joblib.load('rfe_selector.pkl')

#%%
y_pred=RFEselector.predict(X_test_df)

mse_scores= mean_squared_error(y_test, y_pred)
rmse_scores=np.sqrt(mse_scores)
print(rmse_scores.mean())

#%%
# df_trainRFE=df_trainTsf.iloc[:,selector.get_support()]
X_trainRFE=X_train_df.iloc[:,selector.get_support()]

X_testRFE=X_test_df.iloc[:,selector.get_support()]

#%%
X_trainRFE.head()
y_train.head()

#%%
X_trainRFE.to_csv('X_trainRFE.csv', index_label='row_id')
X_testRFE.to_csv('X_testRFE.csv', index_label='row_id')

#%%
X_trainRFE=pd.read_csv('X_trainRFE.csv', index_col='row_id')
X_testRFE=pd.read_csv('X_testRFE.csv', index_col='row_id')

#%%
X_trainRFE.head()
X_testRFE.head()

#%%
estimator.fit(X_trainRFE, y_train.values.ravel())

#%%
y_pred = estimator.predict(X_testRFE)

#%%
importances = pd.DataFrame({'feature':X_trainRFE.columns,'importance':np.round(estimator.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.to_csv('importantRFE_gboost')

#%%
#Have a look at top 10 features ranked by importance
importances = importances.sort_values('importance',ascending=False)

importances.head(n=10)
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