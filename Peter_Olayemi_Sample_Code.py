#!/usr/bin/env python
# coding: utf-8

# In[1]:


#A script that automates the cleaning and exploratory data analysis (EDA) of any dataframe and also performs a linear regression machine-learning model that generalizes to a numeric variables.


# In[127]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# In[195]:


class EDA:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    
    def automated_EDA(self):
        print("\n")
        print("A dataset consisting of {} rows and {} columns \n".format(self.dataframe.shape[0], self.dataframe.shape[1]))
        print("\n")
        print(self.dataframe.head()) #displays the first 5 rows of the dataframe
        print(self.dataframe.dtypes) #displays the structure and datatype of the dataframe
        print(self.dataframe.info())
        print(self.dataframe.columns) #diplays column names of the dataframe
        print(self.dataframe.drop_duplicates(inplace=True)) #drops duplicate rows in the dataframe
        
        print("To find and sort columns with missing values:")
        percent_missing = self.dataframe.isnull().sum() / len(self.dataframe) * 100
        missing_value_df = pd.DataFrame({'column_name': self.dataframe.columns,'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing', inplace=True)
        print("\n")
        
        
        print("\nDecriptive statistics for numeric columns:")
        print(self.dataframe.describe().transpose()) #displays the basic statistics of each numeric column in a transposed fashion
        
        
        print("\nRemoving columns with only 1 unique value")
        self.dataframe1 = self.dataframe.nunique()
        columns_to_del = [i for i,v in enumerate(self.dataframe1) if v == 1]
        print(columns_to_del)
        # drop non-unique columns
        self.dataframe.drop(columns_to_del, axis=1, inplace=True)
        
    def linear_reg_model(self, column):
        self.column = column
        if self.column == float or int:
            X = self.dataframe.drop(column, axis = 1)
            y = self.dataframe[column]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
        
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            base_elastic_model = ElasticNet(tol=0.001)
        
            param_grid = {'alpha':[0.1,0.5,1,5,10,50],
              'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}
        
            grid_model = GridSearchCV(estimator=base_elastic_model,
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=10,
                          verbose=2)
        
            grid_model.fit(X_train,y_train)
            y_pred = grid_model.predict(X_test)
            MSE= mean_squared_error(y_test,y_pred)
            Root_mean_squared_error = np.sqrt(MSE)
            
            print("\nThe mean squared error is {} and the root mean squared error is {}".format(MSE, Root_mean_squared_error))
        else:
            print('Use a numeric column')
    
            
            
        
    
    


# In[196]:


df1 = pd.read_csv("C:\\Users\\peterola\\Downloads\\AMES_Final_DF.csv")


# In[197]:


Explore_data = EDA(df2)


# In[198]:


Explore_data.automated_EDA()


# In[190]:


Explore_data.linear_reg_model('SalePrice')

