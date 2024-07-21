#*************************Written by Guy Tubul & Amir Halabi*****************************
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import shapiro
import statsmodels.api as sm
import plotly.express as px
import seaborn as sns
import datetime as dt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import seaborn as sns
sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Reading the train data
train = (pd.read_csv('/kaggle/input/kaggledatamining/train.csv', sep=','))
test = (pd.read_csv('/kaggle/input/kaggledatamining/test.csv', sep=','))
sample_submission = (pd.read_csv('/kaggle/input/kaggledatamining/sample_submission.csv', sep=','))
# Size of the data 
train.shape

train.describe(include='all')

test.describe(include='all')

# A Quick Information about the Data
train.info()
test.info()

# Checking the # of Null Values for each feature
train.isnull().sum()

# Checking the # of Null Values for each feature
test.isnull().sum()

# train deatures without country and status
train_features = train.columns.values.tolist()
train_features.remove('Country')
train_features.remove('Status')
train_features.remove('Life expectancy ')
# test deatures without country and status
test_features = test.columns.values.tolist()
test_features.remove('Country')
test_features.remove('Status')
test_features.remove('ID')

countries_train = train['Country'].unique()
countries_test  = test['Country'].unique()
#train.loc[train['Afghanistan'],:]

#filling test and train data bases null values with mean values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)

for feature in train_features:
    train[feature]=imputer.fit_transform(train[[feature]]) 
    #after_mean_val_train=pd.concat([after_mean_val_train,country_ds],axis=0)
    
    
data_1 = train[train['Life expectancy '].isnull()].index
train  = train.drop(data_1) 
for feature in test_features:
    test[feature]=imputer.fit_transform(test[[feature]])

# Checking the # of Null Values for each feature after imputing
train.isnull().sum()

# Checking the # of Null Values for each feature after replacing the Null Values with mean values of the data
test.isnull().sum()

# Looking at the first 5 samples of the data
train.head()
test.head()

IDs = test['ID']

round(train[['Status','Life expectancy ']].groupby('Status').mean(),2)

# Heat map of the data
plt.figure(figsize=(15,10))
sns.heatmap(train.corr(),annot=True)
plt.show()


# As we can see from the heatmap the is a hight correlation between thinness 1-19 yeras X thinness 5-9 yeras
# percentage expenditure X GDP and infant deaths X under-five deaths.
# So we can delete one of the features because it does not add information.


#Dropping hight correlation attributes, and year attribute which is not useful
#train = train.drop([' thinness 5-9 years','GDP', 'infant deaths'],axis=1)
#test  = test.drop([' thinness 5-9 years','GDP', 'infant deaths'],axis=1)

train_W_OL = train
test_W_OL  = test

#Histogram of the Life expectancy
figger=px.histogram(train,x='Life expectancy ')
figger.show()

# **Reomving Outlayers from the data**
# Removing Outliers in the variables using Winsorization technique.
#Winsorize Life_Expectancy

from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Life_Expectancy = train['Life expectancy ']
plt.boxplot(original_Life_Expectancy)
plt.title("original_Life_Expectancy")

plt.subplot(1,2,2)
winsorized_Life_Expectancy = winsorize(train['Life expectancy '],(0.007,0))
plt.boxplot(winsorized_Life_Expectancy)
plt.title("winsorized_Life_Expectancy")

plt.show()

# Winsorize Adult_Mortality
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Adult_Mortality = train['Adult Mortality']
plt.boxplot(original_Adult_Mortality)
plt.title("original_Adult_Mortality")

plt.subplot(1,2,2)
winsorized_Adult_Mortality = winsorize(train['Adult Mortality'],(0,0.0292))
plt.boxplot(winsorized_Adult_Mortality)
plt.title("winsorized Adult Mortality")

plt.show()

# Winsorize Infant_Deaths
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Infant_Deaths = train['infant deaths']
plt.boxplot(original_Infant_Deaths)
plt.title("original_Infant_Deaths")

plt.subplot(1,2,2)
winsorized_Infant_Deaths = winsorize(train['infant deaths'],(0,0.12))
plt.boxplot(winsorized_Infant_Deaths)
plt.title("winsorized_Infant_Deaths")

plt.show()

# Winsorize Alcohol
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Alcohol = train['Alcohol']
plt.boxplot(original_Alcohol)
plt.title("original_Alcohol")

plt.subplot(1,2,2)
winsorized_Alcohol = winsorize(train['Alcohol'],(0,0.0013))
plt.boxplot(winsorized_Alcohol)
plt.title("winsorized_Alcohol")

plt.show()

# Winsorize Percentage_Exp
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Percentage_Exp = train['percentage expenditure']
plt.boxplot(original_Percentage_Exp)
plt.title("original_Percentage_Exp")

plt.subplot(1,2,2)
winsorized_Percentage_Exp = winsorize(train['percentage expenditure'],(0,0.131))
plt.boxplot(winsorized_Percentage_Exp)
plt.title("winsorized_Percentage_Exp")

plt.show()

# Winsorize HepatitisB
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_HepatitisB = train['Hepatitis B']
plt.boxplot(original_HepatitisB)
plt.title("original_HepatitisB")

plt.subplot(1,2,2)
winsorized_HepatitisB = winsorize(train['Hepatitis B'],(0.107,0))
plt.boxplot(winsorized_HepatitisB)
plt.title("winsorized_HepatitisB")

plt.show()

# Winsorize Measles
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Measles = train['Measles ']
plt.boxplot(original_Measles)
plt.title("original_Measles")

plt.subplot(1,2,2)
winsorized_Measles = winsorize(train['Measles '],(0,0.19))
plt.boxplot(winsorized_Measles)
plt.title("winsorized_Measles")

plt.show()

# Winsorization changes 19% of the data, which may not give better results. Hence drop this column.
#train.drop('Measles ',axis=1)
#test.drop('Measles ',axis=1)

# Winsorize Under_Five_Deaths
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Under_Five_Deaths = train['under-five deaths ']
plt.boxplot(original_Under_Five_Deaths)
plt.title("original_Under_Five_Deaths")

plt.subplot(1,2,2)
winsorized_Under_Five_Deaths = winsorize(train['under-five deaths '],(0,0.139))
plt.boxplot(winsorized_Under_Five_Deaths)
plt.title("winsorized_Under_Five_Deaths")

plt.show()

# Winsorize Polio
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Polio = train['Polio']
plt.boxplot(original_Polio)
plt.title("original_Polio")

plt.subplot(1,2,2)
winsorized_Polio = winsorize(train['Polio'],(0.097,0))
plt.boxplot(winsorized_Polio)
plt.title("winsorized_Polio")

plt.show()

# Winsorize Tot_Exp
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Tot_Exp = train['Total expenditure']
plt.boxplot(original_Tot_Exp)
plt.title("original_Tot_Exp")

plt.subplot(1,2,2)
winsorized_Tot_Exp = winsorize(train['Total expenditure'],(0,0.018))
plt.boxplot(winsorized_Tot_Exp)
plt.title("winsorized_Tot_Exp")

plt.show()

# Winsorize Diphtheria
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Diphtheria = train['Diphtheria ']
plt.boxplot(original_Diphtheria)
plt.title("original_Diphtheria")

plt.subplot(1,2,2)
winsorized_Diphtheria = winsorize(train['Diphtheria '],(0.10077,0))
plt.boxplot(winsorized_Diphtheria)
plt.title("winsorized_Diphtheria")

plt.show()

# Winsorize HIV/AIDS
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_HIV = train[' HIV/AIDS']
plt.boxplot(original_HIV)
plt.title("original_HIV")

plt.subplot(1,2,2)
winsorized_HIV = winsorize(train[' HIV/AIDS'],(0,0.193))
plt.boxplot(winsorized_HIV)
plt.title("winsorized_HIV")

plt.show()

# Winsorize GDP
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_GDP = train['GDP']
plt.boxplot(original_GDP)
plt.title("original_GDP")

plt.subplot(1,2,2)
winsorized_GDP = winsorize(train['GDP'],(0,0.13))
plt.boxplot(winsorized_GDP)
plt.title("winsorized_GDP")

plt.show()

# Winsorize Population
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Population = train['Population']
plt.boxplot(original_Population)
plt.title("original_Population")

plt.subplot(1,2,2)
winsorized_Population = winsorize(train['Population'],(0,0.063))
plt.boxplot(winsorized_Population)
plt.title("winsorized_Population")

plt.show()

# Winsorize thinness_1to19_years
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_thinness_1to19_years = train[' thinness  1-19 years']
plt.boxplot(original_thinness_1to19_years)
plt.title("original_thinness_1to19_years")

plt.subplot(1,2,2)
winsorized_thinness_1to19_years = winsorize(train[' thinness  1-19 years'],(0,0.04))
plt.boxplot(winsorized_thinness_1to19_years)
plt.title("winsorized_thinness_1to19_years")

plt.show()

# Winsorize thinness_5to9_years
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_thinness_5to9_years = train[' thinness 5-9 years']
plt.boxplot(original_thinness_5to9_years)
plt.title("original_thinness_5to9_years")

plt.subplot(1,2,2)
winsorized_thinness_5to9_years = winsorize(train[' thinness 5-9 years'],(0,0.0369))
plt.boxplot(winsorized_thinness_5to9_years)
plt.title("winsorized_thinness_5to9_years")

plt.show()

# Winsorize Income_Comp_Of_Resources
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Income_Comp_Of_Resources = train['Income composition of resources']
plt.boxplot(original_Income_Comp_Of_Resources)
plt.title("original_Income_Comp_Of_Resources")

plt.subplot(1,2,2)
winsorized_Income_Comp_Of_Resources = winsorize(train['Income composition of resources'],(0.045,0))
plt.boxplot(winsorized_Income_Comp_Of_Resources)
plt.title("winsorized_Income_Comp_Of_Resources")

plt.show()

#Winsorize Schooling
from scipy.stats.mstats import winsorize
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Schooling = train['Schooling']
plt.boxplot(original_Schooling)
plt.title("original_Schooling")

plt.subplot(1,2,2)
winsorized_Schooling = winsorize(train['Schooling'],(0.019,0.004))
plt.boxplot(winsorized_Schooling)
plt.title("winsorized_Schooling")

plt.show()

# Check number of Outliers after Winsorization for each variable.
win_list = [winsorized_Life_Expectancy,winsorized_Adult_Mortality,winsorized_Infant_Deaths,winsorized_Alcohol,
            winsorized_Percentage_Exp,winsorized_HepatitisB,winsorized_Under_Five_Deaths,winsorized_Polio,winsorized_Tot_Exp,winsorized_Diphtheria,winsorized_HIV,winsorized_GDP,winsorized_Population,winsorized_thinness_1to19_years,winsorized_thinness_5to9_years,winsorized_Income_Comp_Of_Resources,winsorized_Schooling]
for variable in win_list:
    q75, q25 = np.percentile(variable, [75 ,25])
    iqr = q75 - q25

    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    
    print("Number of outliers after winsorization : {}".format(len(np.where((variable > max_val) | (variable < min_val))[0])))

# train features before removing outlayers without country and status
train_W_OL_features = train_W_OL.columns.values.tolist()
train_W_OL_features.remove('Country')
train_W_OL_features.remove('Status')

# train features after removing outlayers without country and status
train_WO_OL_features = train.columns.values.tolist()
train_WO_OL_features.remove('Country')
train_WO_OL_features.remove('Status')

# %% [markdown]
# get_dummies of the objects

# Adding winsorized variabtrains to the data frame.
train['Life expectancy '] = winsorized_Life_Expectancy
train['Adult Mortality'] = winsorized_Adult_Mortality
train['infant deaths'] = winsorized_Infant_Deaths
train['Alcohol'] = winsorized_Alcohol
train['percentage expenditure'] = winsorized_Percentage_Exp
train['Hepatitis B'] = winsorized_HepatitisB
train['under-five deaths '] = winsorized_Under_Five_Deaths
train['Polio'] = winsorized_Polio
train['Total expenditure'] = winsorized_Tot_Exp
train['Diphtheria '] = winsorized_Diphtheria
train[' HIV/AIDS'] = winsorized_HIV
train['GDP'] = winsorized_GDP
train['Population'] = winsorized_Population
train[' thinness  1-19 years'] = winsorized_thinness_1to19_years
train[' thinness 5-9 years'] = winsorized_thinness_5to9_years
train['Income composition of resources'] = winsorized_Income_Comp_Of_Resources
train['Schooling'] = winsorized_Schooling

train.info()
test['Status'].unique()
train['Status'].unique()

# Lets see number of unique values.
len(test['Country'].unique())

len(train['Country'].unique())

# missing countries values between two databases
idx1 = pd.Index(train['Country'].unique())
idx2 = pd.Index(test['Country'].unique())
missing_cols = idx1.difference(idx2).values

# making a zero matrix for the missing countries of sieze  test.shape[0] |  missing_cols
Zeros = pd.DataFrame(0, index=np.arange(test.shape[0]), columns=missing_cols)

#adding to the test
#Zeros = pd.DataFrame(0, index=np.arange(train.shape[0]), columns=missing_cols)
Country_dummy = pd.get_dummies(train['Country'])
Country_dummy_test = pd.get_dummies(test['Country'])
# Dummy variables for Country feature.

#merging the test data frame with the 0 DF of missing countries
Country_dummy_test_combined = pd.concat([Country_dummy_test, Zeros], axis=1)

Country_dummy.sort_index(axis=1)

Country_dummy_test_combined = Country_dummy_test_combined.sort_index(axis=1)

status_dummy=pd.get_dummies(train['Status'])
status_dummy_test=pd.get_dummies(test['Status'])
# Dummy variables for status feature.

#dropping the objects columns
train.drop(['Country','Status'],inplace=True,axis=1)
test.drop(['Country','Status'],inplace=True,axis=1)

def correlation(df, threashold):
    correlated_cols = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threashold:
                colname = corr_matrix.columns[i]
                correlated_cols.add(colname)
    return correlated_cols

corr_train_features = correlation(train, 0.8)
corr_train_features

#dropping hight correlation features from the train
train.drop(labels=corr_train_features, axis=1, inplace=True)

#dropping hight correlation features from the test
test.drop(labels=corr_train_features, axis=1, inplace=True)

#from statsmodels.stats.outliers_influence import variance_inflation_factor
train_WO_OL_features = train.columns.values.tolist()
train_WO_OL_features .remove('Life expectancy ')
var = train[train_WO_OL_features]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]
vif['features'] = var.columns
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#dropping hight correlation features from the train
train.drop('Year', axis=1, inplace=True)

#dropping hight correlation features from the test
test.drop('Year', axis=1, inplace=True)

#from statsmodels.stats.outliers_influence import variance_inflation_factor
train_WO_OL_features = train.columns.values.tolist()
train_WO_OL_features .remove('Life expectancy ')
var = train[train_WO_OL_features]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]
vif['features'] = var.columns
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#dropping hight correlation features from the train
train.drop('Polio', axis=1, inplace=True)

#dropping hight correlation features from the test
test.drop('Polio', axis=1, inplace=True)

#from statsmodels.stats.outliers_influence import variance_inflation_factor
train_WO_OL_features = train.columns.values.tolist()
train_WO_OL_features .remove('Life expectancy ')
var = train[train_WO_OL_features]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]
vif['features'] = var.columns
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#dropping hight correlation features from the train
train.drop('Hepatitis B', axis=1, inplace=True)

#dropping hight correlation features from the test
test.drop('Hepatitis B', axis=1, inplace=True)

#from statsmodels.stats.outliers_influence import variance_inflation_factor
train_WO_OL_features = train.columns.values.tolist()
train_WO_OL_features .remove('Life expectancy ')
var = train[train_WO_OL_features]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]
vif['features'] = var.columns
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#dropping hight correlation features from the train
train.drop('Income composition of resources', axis=1, inplace=True)

#dropping hight correlation features from the test
test.drop('Income composition of resources', axis=1, inplace=True)

#from statsmodels.stats.outliers_influence import variance_inflation_factor
train_WO_OL_features = train.columns.values.tolist()
train_WO_OL_features .remove('Life expectancy ')
var = train[train_WO_OL_features]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]
vif['features'] = var.columns
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#dropping hight correlation features from the train
train.drop('Total expenditure', axis=1, inplace=True)

#dropping hight correlation features from the test
test.drop('Total expenditure', axis=1, inplace=True)

#from statsmodels.stats.outliers_influence import variance_inflation_factor
train_WO_OL_features = train.columns.values.tolist()
train_WO_OL_features .remove('Life expectancy ')
var = train[train_WO_OL_features]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]
vif['features'] = var.columns
vif = vif.sort_values(by = "VIF", ascending = False)
vif


data_train=pd.concat([train,Country_dummy,status_dummy],axis=1)
test=pd.concat([test,Country_dummy_test_combined,status_dummy_test],axis=1)
##countries = pd.merge(X,test, how='inner')

data_train.info()
data_train.head()

#shuffiling randomly the data frame
data_train = data_train.sample(frac = 1)

from sklearn.model_selection import train_test_split
y = data_train['Life expectancy ']
data_train.drop(['Life expectancy '],inplace=True,axis=1)
X = data_train
test.drop(['ID'],inplace=True,axis=1)

# Lets set 30% for testing and 70% for training the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train, X_train_lm).fit()
lr_1.summary()

from sklearn.linear_model import LinearRegression

Linear_model= LinearRegression()

Linear_model.fit(X_train,y_train)

predictions1=Linear_model.predict(X_test)
predictions2=Linear_model.predict(test)

# mean_squared_error calculating
print(mean_squared_error(y_test,predictions1)**(0.5))

# R_square calculation
print('R_square score on the training: %.2f' % Linear_model.score(X_train,y_train))

# Bulding an output data frame including ID numbers
Result = pd.DataFrame(predictions2,columns=['Life Expectancy'])

Results = pd.concat([IDs,Result], axis=1)
# Exporting Results
Results.to_csv('/kaggle/working/Results6.csv',index=False)

from sklearn.linear_model import Lasso

lasso_model=Lasso(alpha=0.00000001)

# Alpha value here was selected after choosing 8 different combinations like 0.1,0.001,0.0001...etc.
lasso_model.fit(X_train,y_train)

predictions4=lasso_model.predict(X_test)
predictions5=lasso_model.predict(test)

Result_lasso = pd.DataFrame(predictions5,columns=['Life Expectancy'])

Results_lasso = pd.concat([IDs,Result_lasso], axis=1)
# Exporting Results
Results_lasso.to_csv('/kaggle/working/Results_lasso5.csv',index=False)

print(mean_squared_error(y_test,predictions4)**(0.5))