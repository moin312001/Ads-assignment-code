# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:56:00 2023

@author: MOIN
"""

#Importing packages
import wbgapi as wb
import pandas as pd
import sklearn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from numpy import array, exp

#Initialising the data for different economies
global_dframe=pd.read_csv(r"C:\\Mohit\Global_Economic_Dataset.csv", low_memory=False)

#Initial rows of the dataset
global_dframe.head()

#Transpose dataset format
global_dframe.T

#Assigning indicators for analysis in different economies
import warnings
with warnings.catch_warnings(record=True):
    indct_econm = ['NY.GDP.MKTP.CD','SP.POP.TOTL']
    cntries = ['PAK','BGD','BRA','JPN','ARG','JAM','CHE','LUX','AUS','BMU']
    indct_clmt=['EN.ATM.CO2E.GF.KT','EG.CFT.ACCS.RU.ZS']
    data_econm  = wb.data.DataFrame(indct_econm, cntries, mrv=6)
    data_clmt = wb.data.DataFrame(indct_clmt, cntries, mrv=6)

#NY.GDP.MKT.CD- Country's GDP
#SP.POP.TOTL- Country's total population
#EN.ATM.CO2E.GF.KT- CO2 emissions due to use of gaseous fuel 
#EG.CFT.ACCS.ZS- Access of cleaner fuels and technologies in rural places for cooking

# Economic indicators of a country
data_econm.columns = [b.replace('YR','') for b in data_econm.columns]      
data_econm=data_econm.stack().unstack(level=1)                             
data_econm.index.names = ['Country', 'Year']                           
data_econm.columns                                                     
data_econm.fillna(0)
data_econm.head()

# Climatic indicators of a country
data_clmt.columns = [b.replace('YR','') for b in data_clmt.columns]      
data_clmt=data_clmt.stack().unstack(level=1)                             
data_clmt.index.names = ['Country', 'Year']                           
data_clmt.columns                                                     
data_clmt.fillna(0)
data_clmt.head()

#Reset index and eliminate null values
a1=data_econm.reset_index()
b1=data_clmt.reset_index()
a2=a1.fillna(0)
b2=b1.fillna(0)

#Joining the 2 datasets
fnl_data = pd.merge(a2, b2)
fnl_data.head()

#Normalise the values in the dataset
scal = fnl_data.iloc[:,2:]
fnl_data.iloc[:,2:] = (scal-scal.min())/ (scal.max() - scal.min())
fnl_data.head()

#K-means Form of clustering 
fnl_data_numrc = fnl_data.drop('Country', axis = 1)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(fnl_data_numrc)

#Clustering visualisation of the countries on basis of access of cleaner fuels and technologies in rural places for cooking
sns.scatterplot(data=fnl_data, x="Country", y="EG.CFT.ACCS.RU.ZS", hue=kmeans.labels_)
plt.show()

#Scatterplot visualisation for Total population vs GDP for Jamaica
df=fnl_data[(fnl_data['Country']=='JAM')]
df1 = df.values
x, y = df1[:, 2], df1[:, 3]
plt.scatter(x, y,color="blue")
plt.ylabel('Total Population')
plt.xlabel('GDP of a Country')
plt.show()

#Applying curve_fit function for Argentina - High value of access of cleaner fuels and technologies in rural places for cooking
#Scatter visualisation - Relationship between the Total population and GDP
df2=fnl_data[(fnl_data['Country']=='ARG')]
df3 = df2.values
x, y = df3[:, 2], df3[:, 3]

def fnctin(x, a, b, c):
    return a*x**3+b*x+c
parms, covrnc = curve_fit(fnctin, x, y)
print("Covariance value equals:", covrnc)
print("Parameter value equals:", parms)


parametrs, _ = curve_fit(fnctin, x, y)
a, b, c = parms[0], parms[1], parms[2]
y_fitted = a*x**3+b*x+c

import warnings
with warnings.catch_warnings(record=True):
    plt.plot(x, y_fitted, label="y=a*x**3+b*x+c",color="blue")
    plt.plot(x, y, 'bo', label="Original Y",color="blue")
    plt.ylabel('Total Population')
    plt.xlabel('GDP of a Country')
    plt.grid(True)
    plt.legend(fancybox=True, shadow=True,loc='best')
    plt.show() 
    
#Applying curve_fit function for Bangladesh - Low value of access of cleaner fuels and technologies in rural places for cooking
#Scatter visualisation - Relationship between the Total population and GDP
df4=fnl_data[(fnl_data['Country']=='BGD')]
df5 = df4.values
x, y = df5[:, 2], df5[:, 3]

def fnctin(x, a, b, c):
    return a*x**3+b*x+c
parms, covrnc = curve_fit(fnctin, x, y)
print("Covariance value equals:", covrnc)
print("Parameter value equals:", parms)


parametrs, _ = curve_fit(fnctin, x, y)
a, b, c = parms[0], parms[1], parms[2]
y_fitted = a*x**3+b*x+c

import warnings
with warnings.catch_warnings(record=True):
    plt.plot(x, y_fitted, label="y=a*x**3+b*x+c",color="blue")
    plt.plot(x, y, 'bo', label="Original Y",color="blue")
    plt.ylabel('Total Population')
    plt.xlabel('GDP of a Country')
    plt.grid(True)
    plt.legend(fancybox=True, shadow=True,loc='best')
    plt.show() 
    
#Applying curve_fit function for Brazil - Medium value of access of cleaner fuels and technologies in rural places for cooking
#Scatter visualisation - Relationship between the Total population and GDP
df6=fnl_data[(fnl_data['Country']=='BRA')]
df7 = df6.values
x, y = df7[:, 2], df7[:, 3]

def fnctin(x, a, b, c):
    return a*x**3+b*x+c
parms, covrnc = curve_fit(fnctin, x, y)
print("Covariance value equals:", covrnc)
print("Parameter value equals:", parms)


parametrs, _ = curve_fit(fnctin, x, y)
a, b, c = parms[0], parms[1], parms[2]
y_fitted = a*x**3+b*x+c

import warnings
with warnings.catch_warnings(record=True):
    plt.plot(x, y_fitted, label="y=a*x**3+b*x+c",color="blue")
    plt.plot(x, y, 'bo', label="Original Y",color="blue")
    plt.ylabel('Total Population')
    plt.xlabel('GDP of a Country')
    plt.grid(True)
    plt.legend(fancybox=True, shadow=True,loc='best')
    plt.show() 
    
 def err_ranges(x, func, param, sigma):
    import itertools as iter
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 