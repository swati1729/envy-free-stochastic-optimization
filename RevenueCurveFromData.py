# -*- coding: utf-8 -*-
"""
Created on May 5 2021
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

Dependency: run this file before running
fig5-left.py or fig5-center.py.

This file derives and pre-processes online retail
data. The output is a list of revenue curves (one
for each item). These revenue curves are used in
experiments in "Algorithmic Challenges in Ensuring 
Fairness at the Time of Decision." 
"""


from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import datetime
from scipy import stats
import scipy
from sklearn.metrics import r2_score

# Number of items to consider
num_items = 37

# min and max prices
p_min = 0
p_max = 1


def mean_absolute_percentage_error(y_true, y_pred): 
    """MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

# read in data
df = pd.read_excel('Online-retail.xlsx')


df['Description'] = df['Description'].replace(df['Description'].value_counts().index.tolist()[:num_items],range(num_items))
df = df[pd.to_numeric(df['Description'], errors='coerce').notnull()]
# print(df.head(10))
# print(len(df["Description"].unique()))

# Remove extraneous columns
df.drop(['InvoiceNo', 'StockCode','CustomerID','Country'], axis=1, inplace=True)

# Reorder columns
df = df[['Description','UnitPrice','Quantity','InvoiceDate']]

# Save data
df.to_csv("Online-retail-several-items.csv")

def gss(h, a, b, tol=1e-5):
    """Golden Section Search: optimizes a unimodal function"""
    gr = (np.sqrt(5) + 1) / 2 
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if h(c) < h(d):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2



def RevenueFromData():
    
    # Load data and drop extraneous column
    df = pd.read_csv('Online-retail-several-items.csv')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Remove the item at price 0
    df = df[(df["UnitPrice"]!=0)]
    
    # arrays to track model coefficients/goodness of fit
    degree = 1
    d = np.zeros((degree+1,num_items-2))
    mape = np.zeros(num_items-2)
    corr = np.zeros(num_items-2)
    ls_error = np.zeros(num_items-2)
    r2 = np.zeros(num_items-2)
    optimal_x = np.zeros(num_items-2)
    
    
    for item_raw in range(num_items):
        if item_raw != 10 and item_raw != 14:
            # renumber items
            if item_raw > 14:
                item = item_raw - 2
            elif item_raw > 10:
                item = item_raw - 1
            else:
                item = item_raw
            
            print('ITEM ',str(item))
            df_temp = df[(df["Description"]==item_raw)].copy()
            
            # Scale prices to be in [0,1] 
            df_temp['UnitPrice'] = (df_temp['UnitPrice'] - df_temp['UnitPrice'].min())/(df_temp['UnitPrice'].max() - df_temp['UnitPrice'].min())
            
            # Remove returns
            num_to_remove = (df_temp['Quantity'].values < -.5).sum()
            to_remove = np.zeros((2,num_to_remove))
            index_for_removals = 0
            for index, row in df_temp.iterrows():
                if row["Quantity"] < -.5:
                    to_remove[0,index_for_removals] = row["UnitPrice"]
                    to_remove[1,index_for_removals] = row["Quantity"]
                    index_for_removals += 1
                    df_temp.drop(index, inplace=True)
            
            # count number of removed transactions
            num_removed = 0
            for i in range(num_to_remove):
                for index, row in df_temp.iterrows():
                    if (row["UnitPrice"] == to_remove[0,i]) and (row["Quantity"] == -to_remove[1,i]):
                        df_temp.drop(index, inplace=True)
                        num_removed += 1
                        break
            print(num_to_remove - num_removed, ' returns unaccounted for.')

            # normalize invoice date
            df_temp['InvoiceDate'] = np.floor(pd.to_datetime(df_temp['InvoiceDate']).astype(np.int64)/ (10**9 * 60 * 60 * 24)) - 14944 
            total_days = int(df_temp['InvoiceDate'].max()) 
            print('Total days remaining for item ', str(item), ': ',total_days)
            
            for i in range(total_days):
                if len(df_temp[(df_temp["InvoiceDate"]==i)]["UnitPrice"].unique()) > 1:
                    df_temp = df_temp[(df_temp["InvoiceDate"]!=i)]

            # revenue rate for each price
            unique_prices = df_temp["UnitPrice"].unique()
            print('Unique prices for item ', str(item), ': ',unique_prices)
            time_averages_on_unique_prices = np.zeros(len(unique_prices))
            for i in range(len(unique_prices)):
                time_averages_on_unique_prices[i] = df_temp[(df_temp["UnitPrice"]==unique_prices[i])]['Quantity'].sum()/ len(df_temp[(df_temp["UnitPrice"]==unique_prices[i])]['InvoiceDate'].unique())
            
            
            array = np.transpose(np.array([unique_prices, time_averages_on_unique_prices]))
            df2 = pd.DataFrame(array,range(1,len(df_temp["UnitPrice"].unique())+1),['UnitPrice2','Average quantity'])
            ax = df2.plot.scatter(y='Average quantity',x='UnitPrice2') # plot, if desired
            
            # linear regression
            d_temp = np.polyfit(df2['UnitPrice2'],df2['Average quantity'],degree)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df2['UnitPrice2'], df2['Average quantity'])
            print('Correlation coefficient: ',r_value)
            corr[item] = r_value
            
            # resulting demand curve
            def f(arg):
                ret = 0
                for i in range(len(d_temp)):
                    ret += d_temp[i] * arg**(degree-i)
                return ret 
            
            # resulting revenue curve
            def g(arg):
                ret = 0
                for i in range(len(d_temp)):
                    ret += d_temp[i] * arg**(degree-i)
                return -ret * arg
            
            optimal_x_temp = gss(g,p_min,p_max)
            optimal_x[item] = optimal_x_temp
            
            # statistics
            mape_temp = mean_absolute_percentage_error(df2['Average quantity'],f(df2['UnitPrice2']))
            print('MAPE: ', str(mape_temp))
            mape[item] = mape_temp
            
            lse_temp = ((df2['Average quantity'] - f(df2['UnitPrice2']))**2).mean()
            ls_error[item] = lse_temp 
            print('MSE: ', str(lse_temp))
            
            SS_tot = ((df2['Average quantity'] - df2['Average quantity'].mean())**2).sum()
            SS_res = ((df2['Average quantity'] - f(df2['UnitPrice2']))**2).sum()
            r2_temp = 1 - SS_res/SS_tot
            r2[item] = r2_temp 
            print('R2: ', str(r2_temp))
            r2_temp = r2_score(df2['Average quantity'], f(df2['UnitPrice2']))
            print('R2: ', str(r2_temp))
            
            num_rows = (df2['Average quantity'].values < 30000).sum()
            df2.insert(0,'Price',np.linspace(0,1,num_rows))
            df2.insert(0,'Linear regression',f(df2['Price']))
            
            df2.plot(x='Price', y='Linear regression',color='Red',ax=ax)
            
            d[:,item] = d_temp
            
            print('\n')
    return d,mape,corr,ls_error,r2,optimal_x

d,mape,corr,ls_error,r2,optimal_x = RevenueFromData()


file_d = open("d","wb")
np.save(file_d,d)
file_d.close

