#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.figure_factory as ff


# In[2]:


#extracting dataset
df=pd.read_csv('LifeExpectancyData.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated()


# In[7]:


#https://elearning.dbs.ie/pluginfile.php/1540609/mod_resource/content/1/CarResale.html
#to binarize the entries of column "status"

df['Status'] = df['Status'].map({'Developed':1,'Developing':0})


# In[8]:


#https://www.geeksforgeeks.org/replacing-missing-values-using-pandas-in-python/
#https://vitalflux.com/pandas-impute-missing-values-mean-median-mode/

df['Alcohol'] = df['Alcohol'].fillna(df['Alcohol'].median())
df['HepatitisB'] = df['HepatitisB'].fillna(df['HepatitisB'].median())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['GDP'] = df['GDP'].fillna(df['GDP'].median())
df['Population'] = df['Population'].fillna(df['Population'].median())
df['Polio'] = df['Polio'].fillna(df['Polio'].median())
df['Totalexpenditure'] = df['Totalexpenditure'].fillna(df['Totalexpenditure'].median())
df['Schooling'] = df['Schooling'].fillna(df['Schooling'].median())
df['thinness(1-19 years)'] = df['thinness(1-19 years)'].fillna(df['thinness(1-19 years)'].median())
df['thinness(5-9 years)'] = df['thinness(5-9 years)'].fillna(df['thinness(5-9 years)'].median())
df['HIV/AIDS'] = df['HIV/AIDS'].fillna(df['HIV/AIDS'].median())
df['lifeexpect'] = df['lifeexpect'].fillna(df['lifeexpect'].median())
df['AdultMortality'] = df['AdultMortality'].fillna(df['AdultMortality'].median())
df['Diphtheria'] = df['Diphtheria'].fillna(df['Diphtheria'].median())
df['IncomecCmposition-of- resources'] = df['IncomecCmposition-of- resources'].fillna(df['IncomecCmposition-of- resources'].median())


# In[9]:


#check again if there is any null value
df.isnull().sum()


# In[10]:


print(df.info())
print(df.head(20))


# In[11]:


#https://elearning.dbs.ie/pluginfile.php/1540609/mod_resource/content/1/CarResale.html (class lectures)
corrs = df.corr()
corrs


# In[12]:


#https://elearning.dbs.ie/pluginfile.php/1540609/mod_resource/content/1/CarResale.html (class lectures)
plt.subplots(figsize=(15,10))
map_correlation=sns.heatmap(corrs, annot=True, linewidth=1)#, cmap='coolwarm')
map_correlation


# In[13]:


##lifeexpect is correlated to Schooling,IncomecComposition-of-resources,GDP,Diphteria,Totalexpenditure, Polio,BMI, HepatitisB,percentageexpenditure, Alcohol,Status and Year


# In[12]:


# to check from the distplots and boxplots, the presence of outliers
#https://www.kaggle.com/asimislam/tutorial-python-subplots
# https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(16,8))
plt.subplot(8,4,1)
sns.distplot(df['Schooling'])
plt.subplot(8,4,2)
sns.boxplot(df['Schooling'])
plt.subplot(8,4,3)
sns.distplot(df['IncomecCmposition-of- resources'])
plt.subplot(8,4,4)
sns.boxplot(df['IncomecCmposition-of- resources'])
plt.show()
plt.figure(figsize=(16,8))
plt.subplot(8,4,5)
sns.distplot(df['GDP'])
plt.subplot(8,4,6)
sns.boxplot(df['GDP'])
plt.subplot(8,4,7)
sns.distplot(df['IncomecCmposition-of- resources'])
plt.subplot(8,4,8)
sns.boxplot(df['IncomecCmposition-of- resources'])
plt.show()
plt.figure(figsize=(16,8))
plt.subplot(8,4,9)
sns.distplot(df['Diphtheria'])
plt.subplot(8,4,10)
sns.boxplot(df['Diphtheria'])
plt.subplot(8,4,11)
sns.distplot(df['Totalexpenditure'])
plt.subplot(8,4,12)
sns.boxplot(df['Totalexpenditure'])
plt.show()
plt.figure(figsize=(16,8))
plt.subplot(8,4,13)
sns.distplot(df['BMI'])
plt.subplot(8,4,14)
sns.boxplot(df['BMI'])
plt.subplot(8,4,15)
sns.distplot(df['Polio'])
plt.subplot(8,4,16)
sns.boxplot(df['Polio'])
plt.show()
plt.figure(figsize=(16,8))
plt.subplot(8,4,17)
sns.distplot(df['lifeexpect'])
plt.subplot(8,4,18)
sns.boxplot(df['lifeexpect'])
plt.subplot(8,4,19)
sns.distplot(df['HepatitisB'])
plt.subplot(8,4,20)
sns.boxplot(df['HepatitisB'])
plt.show()
plt.figure(figsize=(16,8))
plt.subplot(8,4,21)
sns.distplot(df['Alcohol'])
plt.subplot(8,4,22)
sns.boxplot(df['Alcohol'])
plt.subplot(8,4,23)
sns.distplot(df['Totalexpenditure'])
plt.subplot(8,4,24)
sns.boxplot(df['Totalexpenditure'])
plt.show()


# In[25]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/
#Schooling
percentile25_Schooling = df['Schooling'].quantile(0.25)
percentile75_Schooling = df['Schooling'].quantile(0.75)
iqr_Schooling=percentile75_Schooling-percentile25_Schooling
upper_limit_Schooling = percentile75_Schooling + 1.5 * iqr_Schooling
lower_limit_Schooling = percentile25_Schooling - 1.5 * iqr_Schooling
df[df['Schooling'] > upper_limit_Schooling]  #Find outliers
df[df['Schooling'] < lower_limit_Schooling]
#Trimming
new_df_Schooling = df[df['Schooling'] < upper_limit_Schooling]
new_df_Schooling.shape

#Capping
new_df_cap_Schooling = df.copy()
new_df_cap_Schooling['Schooling'] = np.where(
    new_df_cap_Schooling['Schooling'] > upper_limit_Schooling,
    upper_limit_Schooling,
    np.where(
        new_df_cap_Schooling['Schooling'] < lower_limit_Schooling,
        lower_limit_Schooling,
        new_df_cap_Schooling['Schooling']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Schooling'])
plt.subplot(2,2,2)
sns.boxplot(df['Schooling'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Schooling['Schooling'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Schooling['Schooling'])
plt.show()


# In[13]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/
#'GDP'
percentile25_GDP = df['GDP'].quantile(0.25)
percentile75_GDP = df['GDP'].quantile(0.75)
iqr_GDP=percentile75_GDP-percentile25_GDP
upper_limit_GDP = percentile75_GDP + 1.5 * iqr_GDP
lower_limit_GDP = percentile25_GDP - 1.5 * iqr_GDP
df[df['GDP'] > upper_limit_GDP]  #Find outliers
df[df['GDP'] < lower_limit_GDP]
#Trimming
new_df_GDP = df[df['GDP'] < upper_limit_GDP]
new_df_GDP.shape


#Capping
new_df_cap_GDP = df.copy()
new_df_cap_GDP['GDP'] = np.where(
    new_df_cap_GDP['GDP'] > upper_limit_GDP,
    upper_limit_GDP,
    np.where(
        new_df_cap_GDP['GDP'] < lower_limit_GDP,
        lower_limit_GDP,
        new_df_cap_GDP['GDP']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['GDP'])
plt.subplot(2,2,2)
sns.boxplot(df['GDP'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_GDP['GDP'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_GDP['GDP'])
plt.show()


# In[33]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/

#'Diphteria'
percentile25_Dip = df['Diphtheria'].quantile(0.25)
percentile75_Dip = df['Diphtheria'].quantile(0.75)
iqr_Dip=percentile75_Dip-percentile25_Dip
upper_limit_Dip = percentile75_Dip + 1.5 * iqr_Dip
lower_limit_Dip = percentile25_Dip - 1.5 * iqr_Dip
df[df['Diphtheria'] > upper_limit_Dip] #Find outliers
df[df['Diphtheria'] < lower_limit_Dip]

#Trimming
new_df_Dip = df[df['Diphtheria'] < upper_limit_Dip]
new_df_Dip.shape

#Capping
new_df_cap_Dip = df.copy()
new_df_cap_Dip['Diphtheria'] = np.where(
    new_df_cap_Dip['Diphtheria'] > upper_limit_Dip,
    upper_limit_Dip,
    np.where(
        new_df_cap_Dip['Diphtheria'] < lower_limit_Dip,
        lower_limit_Dip,
        new_df_cap_Dip['Diphtheria']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Diphtheria'])
plt.subplot(2,2,2)
sns.boxplot(df['Diphtheria'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Dip['Diphtheria'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Dip['Diphtheria'])
plt.show()


# In[14]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/

#IncomecComposition-of-resources
percentile25_Income = df['IncomecCmposition-of- resources'].quantile(0.25)
percentile75_Income = df['IncomecCmposition-of- resources'].quantile(0.75)
iqr_Income=percentile75_Income-percentile25_Income
upper_limit_Income = percentile75_Income + 1.5 * iqr_Income
lower_limit_Income= percentile25_Income - 1.5 * iqr_Income
df[df['IncomecCmposition-of- resources'] > upper_limit_Income] #Find outliers
df[df['IncomecCmposition-of- resources'] < lower_limit_Income]
#Trimming
new_df_Income = df[df['IncomecCmposition-of- resources'] < upper_limit_Income]
new_df_Income.shape

#Capping
new_df_cap_Income = df.copy()
new_df_cap_Income['IncomecCmposition-of- resources'] = np.where(
    new_df_cap_Income['IncomecCmposition-of- resources'] > upper_limit_Income,
    upper_limit_Income,
    np.where(
        new_df_cap_Income['IncomecCmposition-of- resources'] < lower_limit_Income,
        lower_limit_Income,
        new_df_cap_Income['IncomecCmposition-of- resources']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['IncomecCmposition-of- resources'])
plt.subplot(2,2,2)
sns.boxplot(df['IncomecCmposition-of- resources'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Income['IncomecCmposition-of- resources'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Income['IncomecCmposition-of- resources'])
plt.show()


# In[31]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/

#'Totalexpenditure'
percentile25_Totalexpenditure = df['Totalexpenditure'].quantile(0.25)
percentile75_Totalexpenditure = df['Totalexpenditure'].quantile(0.75)
iqr_Totalexpenditure=percentile75_Schooling-percentile25_Schooling
upper_limit_Totalexpenditure = percentile75_Totalexpenditure+ 1.5 * iqr_Totalexpenditure
lower_limit_Totalexpenditure = percentile25_Totalexpenditure- 1.5 * iqr_Totalexpenditure
df[df['Totalexpenditure'] > upper_limit_Totalexpenditure]#Find outliers
df[df['Totalexpenditure'] < lower_limit_Totalexpenditure]

#Trimming
new_df_Totalexp = df[df['Totalexpenditure'] < upper_limit_Totalexpenditure]
new_df_Totalexp.shape

#Capping
new_df_cap_Expenditure= df.copy()
new_df_cap_Expenditure['Totalexpenditure'] = np.where(
    new_df_cap_Expenditure['Totalexpenditure'] > upper_limit_Totalexpenditure,
   upper_limit_Totalexpenditure,
    np.where(
        new_df_cap_Expenditure['Totalexpenditure'] < lower_limit_Totalexpenditure,
       lower_limit_Totalexpenditure,
        new_df_cap_Expenditure['Totalexpenditure']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Totalexpenditure'])
plt.subplot(2,2,2)
sns.boxplot(df['Totalexpenditure'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Expenditure['Totalexpenditure'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Expenditure['Totalexpenditure'])
plt.show()


# In[15]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/

#'HepatitisB'
percentile25_HepatitisB  = df['HepatitisB'].quantile(0.25)
percentile75_HepatitisB  = df['HepatitisB'].quantile(0.75)
iqr_HepatitisB =percentile75_HepatitisB -percentile25_HepatitisB 
upper_limit_HepatitisB  = percentile75_HepatitisB  + 1.5 * iqr_HepatitisB 
lower_limit_HepatitisB  = percentile25_HepatitisB  - 1.5 * iqr_HepatitisB 
df[df['HepatitisB'] > upper_limit_HepatitisB ]
df[df['HepatitisB'] < lower_limit_HepatitisB ]

#Trimming
new_df_Hepatitis= df[df['HepatitisB'] < upper_limit_HepatitisB]
new_df_Hepatitis.shape

#Capping
new_df_cap_Hepatitis= df.copy()
new_df_cap_Hepatitis['HepatitisB'] = np.where(
    new_df_cap_Hepatitis['HepatitisB'] >upper_limit_HepatitisB,
   upper_limit_HepatitisB,
    np.where(
        new_df_cap_Hepatitis['HepatitisB'] < lower_limit_HepatitisB,
       lower_limit_HepatitisB,
        new_df_cap_Hepatitis['HepatitisB']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['HepatitisB'])
plt.subplot(2,2,2)
sns.boxplot(df['HepatitisB'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Hepatitis['HepatitisB'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Hepatitis['HepatitisB'])
plt.show()


# In[29]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/

#'percentageexpenditure'
percentile25_percentageExpenditure= df['percentageExpenditure'].quantile(0.25)
percentile75_percentageExpenditure = df['percentageExpenditure'].quantile(0.75)
iqr_percentageExpenditure=percentile75_Schooling-percentile25_percentageExpenditure
upper_limit_percentageExpenditure = percentile75_percentageExpenditure + 1.5 * iqr_percentageExpenditure
lower_limit_percentageExpenditure = percentile25_percentageExpenditure - 1.5 * iqr_percentageExpenditure
df[df['percentageExpenditure'] > upper_limit_percentageExpenditure]
df[df['percentageExpenditure'] < lower_limit_percentageExpenditure]

#Trimming
new_df_percentageExpenditure= df[df['percentageExpenditure'] < upper_limit_percentageExpenditure]
new_df_percentageExpenditure.shape

#Capping
new_df_cap_percentageExpenditure= df.copy()
new_df_cap_percentageExpenditure['percentageExpenditure'] = np.where(
    new_df_cap_percentageExpenditure['percentageExpenditure'] >upper_limit_percentageExpenditure,
   upper_limit_percentageExpenditure,
    np.where(
        new_df_cap_percentageExpenditure['percentageExpenditure'] < lower_limit_percentageExpenditure,
       lower_limit_percentageExpenditure,
        new_df_cap_percentageExpenditure['percentageExpenditure']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['percentageExpenditure'])
plt.subplot(2,2,2)
sns.boxplot(df['percentageExpenditure'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_percentageExpenditure['percentageExpenditure'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_percentageExpenditure['percentageExpenditure'])
plt.show()


# In[16]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/

#'Alcohol
percentile25_Alcohol= df['Alcohol'].quantile(0.25)
percentile75_Alcohol = df['Alcohol'].quantile(0.75)
iqr_Alcohol=percentile75_Alcohol-percentile25_Alcohol
upper_limit_Alcohol = percentile75_Alcohol+ 1.5 * iqr_Alcohol
lower_limit_Alcohol = percentile25_Alcohol- 1.5 * iqr_Alcohol
df[df['Alcohol'] > upper_limit_Alcohol]
df[df['Alcohol'] < lower_limit_Alcohol]

#Trimming
new_df_Alcohol= df[df['Alcohol'] < upper_limit_Alcohol]
new_df_Alcohol.shape

#Capping
new_df_cap_Alcohol= df.copy()
new_df_cap_Alcohol['Alcohol'] = np.where(
    new_df_cap_Alcohol['Alcohol'] >upper_limit_Alcohol,
   upper_limit_Alcohol,
    np.where(
        new_df_cap_Alcohol['Alcohol'] < lower_limit_Alcohol,
       lower_limit_Alcohol,
        new_df_cap_Alcohol['Alcohol']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Alcohol'])
plt.subplot(2,2,2)
sns.boxplot(df['Alcohol'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Alcohol['Alcohol'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Alcohol['Alcohol'])
plt.show()


# In[17]:


#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/
#'BMI'
percentile25_BMI = df['BMI'].quantile(0.25)
percentile75_BMI = df['BMI'].quantile(0.75)
iqr_BMI=percentile75_BMI-percentile25_BMI
upper_limit_BMI = percentile75_BMI + 1.5 * iqr_BMI
lower_limit_BMI = percentile25_BMI - 1.5 * iqr_BMI
df[df['BMI'] > upper_limit_BMI]
df[df['BMI'] < lower_limit_BMI]

#Trimming
new_df_BMI= df[df['BMI'] < upper_limit_BMI]
new_df_BMI.shape

#Capping
new_df_cap_BMI= df.copy()
new_df_cap_BMI['BMI'] = np.where(
    new_df_cap_BMI['BMI'] >upper_limit_BMI,
   upper_limit_BMI,
    np.where(
        new_df_cap_BMI['BMI'] < lower_limit_BMI,
       lower_limit_BMI,
        new_df_cap_BMI['BMI']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['BMI'])
plt.subplot(2,2,2)
sns.boxplot(df['BMI'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_BMI['BMI'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_BMI['BMI'])
plt.show()


# In[18]:


#'Polio'
percentile25_Polio = df['Polio'].quantile(0.25)
percentile75_Polio = df['Polio'].quantile(0.75)
iqr_Polio =percentile75_Polio -percentile25_Polio 
upper_limit_Polio  = percentile75_Polio  + 1.5 * iqr_Polio 
lower_limit_Polio  = percentile25_Polio  - 1.5 * iqr_Polio 
df[df['Polio'] > upper_limit_Polio ]
df[df['Polio'] < lower_limit_Polio ]

#Trimming
new_df_Polio= df[df['Polio'] < upper_limit_Polio]
new_df_Polio.shape

#Capping
new_df_cap_Polio= df.copy()
new_df_cap_Polio['Polio'] = np.where(
    new_df_cap_Polio['Polio'] >upper_limit_Polio ,
   upper_limit_Polio ,
    np.where(
        new_df_cap_Polio['Polio'] < lower_limit_Polio,
       lower_limit_Polio,
        new_df_cap_Polio['Polio']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Polio'])
plt.subplot(2,2,2)
sns.boxplot(df['Polio'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Polio['Polio'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Polio['Polio'])
plt.show()


# In[19]:


#'Lifeexpect'
percentile25_lifeexpect = df['lifeexpect'].quantile(0.25)
percentile75_lifeexpect = df['lifeexpect'].quantile(0.75)
iqr_lifeexpect =percentile75_lifeexpect -percentile25_lifeexpect
upper_limit_lifeexpect  = percentile75_lifeexpect  + 1.5 * iqr_lifeexpect 
lower_limit_lifeexpect  = percentile25_lifeexpect  - 1.5 * iqr_lifeexpect 
df[df['lifeexpect'] > upper_limit_lifeexpect ]
df[df['lifeexpect'] < lower_limit_lifeexpect ]

#Trimming
new_df_Polio= df[df['lifeexpect'] < upper_limit_lifeexpect]
new_df_Polio.shape

#Capping
new_df_cap_lifeexpect= df.copy()
new_df_cap_lifeexpect['lifeexpect'] = np.where(
    new_df_cap_lifeexpect['lifeexpect'] >upper_limit_lifeexpect ,
   upper_limit_lifeexpect ,
    np.where(
        new_df_cap_lifeexpect['lifeexpect'] < lower_limit_lifeexpect,
       lower_limit_lifeexpect,
        new_df_cap_lifeexpect['lifeexpect']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['lifeexpect'])
plt.subplot(2,2,2)
sns.boxplot(df['lifeexpect'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_lifeexpect['lifeexpect'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_lifeexpect['lifeexpect'])
plt.show()


# In[23]:


#'IncomecCmposition-of- resources'
percentile25_Income = df['IncomecCmposition-of- resources'].quantile(0.25)
percentile75_Income = df['IncomecCmposition-of- resources'].quantile(0.75)
iqr_Income =percentile75_Income-percentile25_Income
upper_limit_Income = percentile75_Income  + 1.5 * iqr_Income
lower_limit_Income = percentile25_Income - 1.5 * iqr_Income
df[df['IncomecCmposition-of- resources'] > upper_limit_Income]
df[df['IncomecCmposition-of- resources'] < lower_limit_Income ]

#Trimming
new_df_Income= df[df['IncomecCmposition-of- resources'] < upper_limit_Income]
new_df_Income.shape

#Capping
new_df_cap_Income= df.copy()
new_df_cap_Income['IncomecCmposition-of- resources'] = np.where(
    new_df_cap_Income['IncomecCmposition-of- resources'] >upper_limit_Income ,
   upper_limit_Income,
    np.where(
        new_df_cap_Income['IncomecCmposition-of- resources'] < lower_limit_Income,
       lower_limit_Income,
        new_df_cap_Income['IncomecCmposition-of- resources']
    )
)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['IncomecCmposition-of- resources'])
plt.subplot(2,2,2)
sns.boxplot(df['IncomecCmposition-of- resources'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap_Income['IncomecCmposition-of- resources'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap_Income['IncomecCmposition-of- resources'])
plt.show()


# In[34]:


#to give the new values of df new columns to the old columns
df['Polio']=new_df_cap_Polio['Polio']
df['lifeexpect']=new_df_cap_lifeexpect['lifeexpect']
df['BMI']=new_df_cap_BMI['BMI']
df['Schooling']= new_df_cap_Schooling['Schooling']
df['Alcohol']=new_df_cap_Alcohol['Alcohol']
df['percentageExpenditure']=new_df_cap_percentageExpenditure['percentageExpenditure']
df['HepatitisB']=new_df_cap_Hepatitis['HepatitisB']
df['Totalexpenditure']=new_df_cap_Expenditure['Totalexpenditure']
df['IncomecComposition-of- resources']=new_df_cap_Income['IncomecCmposition-of- resources']
df['Diphtheria']=new_df_cap_Dip['Diphtheria']
df['GDP']= new_df_cap_GDP['GDP']


# In[35]:


df.describe()


# In[36]:


from scipy.stats import norm
from scipy import stats
sns.distplot(df['lifeexpect'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['lifeexpect'], plot=plt)


# In[37]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 


# In[38]:


#https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/#h2_1
# https://colab.research.google.com/drive/1crT7PUQHjI4zN4NZAN0Zzg9W3aeRmrvZ#scrollTo=GLJeS3EmUrNN


y = df['lifeexpect']
y.head()
X=df.drop(columns=['Country','lifeexpect','AdultMortality','infantdeaths','Measles','under-five deaths','HIV/AIDS','Population','thinness(1-19 years)','thinness(5-9 years)'],axis=1)
X.head()


# In[39]:


y = df['lifeexpect']
y.head()


# In[42]:


y = df['lifeexpect']
y.head()
#print ("\nOriginal data values : \n",  X)
print(type(X))
print(type(y))
print(X.shape)
print(y.shape)


# In[43]:




X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=12345)

# Normalizing numerical features so that each feature has mean 0 and variance 1
Standardisation = StandardScaler()

x_after_Standardisation = Standardisation.fit_transform(X)

print ("\nAfter Standardisation : \n", x_after_Standardisation)


# In[44]:


# https://stackabuse.com/linear-regression-in-python-with-scikit-learn/

regressor = LinearRegression()
regressor.fit(X_train, y_train)
  
# predicting the test set results
y_pred = regressor.predict(X_test)


# In[45]:


cof=regressor.coef_
pd.DataFrame(cof, X.columns, columns=['coef'])


# In[46]:


# Linear Regression
# Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter' using Grid Search

from sklearn.model_selection import GridSearchCV

sgdr = SGDRegressor(random_state = 1, penalty = None)
grid_param = {'eta0': [.0001, .001, .01, .1, 1], 'max_iter':[10000, 20000, 30000, 40000]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(x_after_Standardisation, y)

results = pd.DataFrame.from_dict(gd_sr.cv_results_)
print("Cross-validation results:\n", results)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("Best result: ", best_result)

best_model = gd_sr.best_estimator_
print("Intercept: ", best_model.intercept_)

print(pd.DataFrame(zip(X.columns, best_model.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))


# In[39]:


# Implementing Regularization
# Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter', along with the regularization parameter alpha using Grid Search

sgdr = SGDRegressor(random_state = 1, penalty = 'elasticnet', eta0=.01, max_iter=10000)

grid_param = {'alpha': [.0001, .001, .01, .1, 1], 'l1_ratio': [0,0.25,0.5,0.75,1]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(x_after_Standardisation, y)

results = pd.DataFrame.from_dict(gd_sr.cv_results_)
print("Cross-validation results:\n", results)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("Best result: ", best_result)

best_model = gd_sr.best_estimator_
print("Intercept: ", best_model.intercept_)

print(pd.DataFrame(zip(X.columns, best_model.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))


# In[47]:


from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor(random_state = 6, penalty = None)
sgradient = sgdr.fit(x_after_Standardisation,y)
sgdr_score = gd_sr.best_score_
sgdr_score


# In[48]:


regressor.score(X_test,y_test)


# In[49]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[50]:


# https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
# https://medium.com/machine-learning-with-python/multiple-linear-regression-implementation-in-python-2de9b303fc0c
from sklearn import metrics
print('R squared: {:.2f}'.format(regressor.score(X,y)*100))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




