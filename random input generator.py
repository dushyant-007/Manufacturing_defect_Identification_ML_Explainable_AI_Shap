# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:00:34 2020

@author: HP LAPTOP
"""

import random 
import numpy as np

def get_an_input():
    rand = random.uniform(0,1) ;
    min_ax_ld = 5 ;
    max_ax_ld = 16 ;
    range1= 8 - min_ax_ld ;
    range2 = max_ax_ld - 8 ;
    ax_ld = 7 ;
    if (rand < 0.4):
        ax_ld = min_ax_ld + range1*random.uniform(0,1)
    else:
        ax_ld = 8+ range2*random.uniform(0,1)
    
    # selecting inner and outer modulus
    min_inn = 0.08 
    max_inn = 0.18 
    min_out = 0.05 
    max_out = 0.20 
    
    range_inn = max_inn - min_inn 
    range_out = max_out - min_out 
    
    inner_mod = min_inn + random.uniform(0,1)*range_inn
    outer_mod = min_out + random.uniform(0,1)*range_out
    
    # selecting friction factor
    rand2 = random.uniform(0,1)
    fric_coeff = 0 ;
    if(rand2 < 0.3):
        fric_coeff = random.uniform(0,1)*1 ;
    else:
        fric_coeff = 1+random.uniform(0,1)*100 ;
    
    input1=  np.array((inner_mod , outer_mod, ax_ld , fric_coeff))
    return input1

get_an_input()
