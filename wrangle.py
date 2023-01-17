#!/usr/bin/env python
# coding: utf-8

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import env
from acquire import get_zillow_data
from prepare import prep_zillow

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def wrangle_zillow():
    
    '''
    This function is used to get zillow data from sql database, renaming columns, 
    dropping nulls and duplicates.
    '''
    
    df = get_zillow_data()
    
    df = prep_zillow(df)
    
    return df

