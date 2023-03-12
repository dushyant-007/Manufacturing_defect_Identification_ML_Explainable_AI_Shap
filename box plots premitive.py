# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:28:39 2020

@author: HP LAPTOP

this is an outdated file and shouldn't be very useful anymore
the graph shows the same thing as box plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# getting the datset
train_df = pd.read_csv('Snaking_June_2020 (1).csv') ;
train_df = train_df.iloc[: , 1:8 ]
x = train_df.iloc[: , 0:4].values
y1 = train_df.iloc[: , 4].values
y2 = train_df.iloc[: , 5].values
y3 = train_df.iloc[: , 6].values

# variate the inner modulus at load of 10, outer modulus at 0.12, 0.15. 
# friction coeff at 5
plt.figure()
vary = [0.1 , 0.11 , 0.12 , 0.13 , 0.14 , 0.15]
for i in range(0,2):
    m1 = np.zeros(shape = (6,7))
    m1[: , i]= vary
    m1[: , (i+1)%2 ] = 0.10
    m1[:, 2] = 12
    m1[: , 3] = 5
   
    m1[: , 4] = XGBReg.predict(m1[: , 0:4])
    m1[: , 5] = XGBReg2.predict(m1[: , 0:4])
    m1[: , 6] = XGBReg3.predict(m1[: , 0:4])
    
    varied ='' ;
    if(i == 0): varied = 'inner is varied ' ;
    elif (i==1): varied = 'outer is varied' ;
    
    # plot the graph b/w variable and inner modulus
    #fig1 = plt.figure()
    plt.plot(m1[: , i] , m1[: , 5] , label= 'load = 12 , '+varied)

plt.legend()
plt.xlabel('modulus')
plt.ylabel('Snaking Length')