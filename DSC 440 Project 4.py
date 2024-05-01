#!/usr/bin/env python
# coding: utf-8

# # Predicting Housing Prices

# The analysis below was performed on a dataset that has a variety of features describing factors about houses. There are a variety of both qualitative and quantitative features in this dataset, and both groups will be analyzed in some capacity below. The objective of the analysis is to predict the sale price of each of the houses based on different factors using different analysis methods.

# ## Imports

# In[275]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
houses=pd.read_csv(r"C:\Users\Owner\Downloads\house-prices-advanced-regression-techniques\train.csv")
houses


# The below cell checks to see which features in our dataset are either integers or floats and which are neither of these. These two groups were separated so that the titles of the columns in each group are in different lists.

# In[277]:


count=0
count1=0
col=[]
quan=[]
qual=[]
for i in houses.columns:
    col.append(i)
for i in houses.dtypes:
    if i=="int64" or i=='float64':
        print(count)
        quan.append(col[count])
    else:
        qual.append(col[count])
    count+=1


# ## Correlations Among Discrete Variables

# Below is where the list of quantitative features comes into play, as we compare all of these features to the SalePrice feature to check which of them are the most related to it using a correlation heat map.

# In[295]:


quan1=quan[1:10]+quan[-1:]
quan2=quan[10:19]+quan[-1:]
quan3=quan[19:28]+quan[-1:]
quan4=quan[28:37]+quan[-1:]
print(quan1)
print(quan2)
print(quan3)
print(quan4)


# In[296]:


smol1=houses[quan1]
smol2=houses[quan2]
smol3=houses[quan3]
smol4=houses[quan4]
corr_df=smol1.corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values,annot=True, annot_kws={'size':8})
#heat_map=plt.gcf(); heat_map.set_size_inches(10,5)
#plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.show()


# In[297]:


corr_df=smol2.corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values,annot=True, annot_kws={'size':8})


# In[298]:


corr_df=smol3.corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values,annot=True, annot_kws={'size':8})


# In[299]:


corr_df=smol4.corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values,annot=True, annot_kws={'size':8})


# Based on the above heat maps, the most features that were most correlated to "SalePrice" were put into a heat map together below.

# In[300]:


ghouse=houses[['OverallQual','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','SalePrice']]
corr_df=ghouse.corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values,annot=True, annot_kws={'size':8})


# In[302]:


mosaic=[['a','a','b','b'],
       ['c','c','d','d'],
       ['e','e','f','f']]
plt.figure()
fig, axs=plt.subplot_mosaic(mosaic,
                           layout='constrained')

axs['a'].scatter(houses['OverallQual'],houses['SalePrice'],c='b')
axs['a'].set_title('Overall Quality vs Price')
axs['b'].scatter(houses['GarageCars'],houses['SalePrice'],c='b')
axs['b'].set_title('Car Space in Garage vs Price')
axs['c'].scatter(houses['GarageArea'],houses['SalePrice'],c='b')
axs['c'].set_title('Garage Area vs Price')
axs['d'].scatter(houses['GrLivArea'],houses['SalePrice'],c='b')
axs['d'].set_title('Above Ground SF vs Price')
axs['e'].scatter(houses['TotalBsmtSF'],houses['SalePrice'],c='b')
axs['e'].set_title('Basement SF vs Price')
axs['f'].scatter(houses['1stFlrSF'],houses['SalePrice'],c='b')
axs['f'].set_title('1st Floor SF vs Price')


# Now that the features most related to the "SalePrice" variable have been found, scatterplots for each of these correlations were then created above to better visual the relationships. Based on these plots you can see that there are fairly linear relationships between all of these variables and "SalePrice".

# ## Regression Models

# In[303]:


list=['OverallQual','GarageCars','GarageArea','GrLivArea','TotalBsmtSF','1stFlrSF']
for i in list:
    X=np.array(houses[i])
    y=np.array(houses['SalePrice'])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    ridge_regression = Ridge()
    ridge_regression.fit(X_train, y_train)
    y_pred_ridge = ridge_regression.predict(X_test)

    lasso_regression = Lasso()
    lasso_regression.fit(X_train, y_train)
    y_pred_lasso = lasso_regression.predict(X_test)

    LiR = LinearRegression()
    LiR.fit(X_train, y_train)
    y_pred = LiR.predict(X_test)
    
    plt.scatter(X_test,y_pred,label='Linear Regression',alpha=0.5)
    plt.scatter(X_test,y_pred_ridge,label='Ridge',alpha=0.5)
    plt.scatter(X_test,y_pred_lasso,label='Lasso',alpha=0.5)
    plt.scatter(X,y,label='Actual Data',alpha=0.5)
    plt.xlabel(i)
    plt.ylabel("Sale Price")
    plt.legend()
    plt.show()
    print("Linear MSE:", mean_squared_error(y_test, y_pred))
    print("Ridge MSE:", mean_squared_error(y_test, y_pred_ridge))
    print("LASSO MSE:", mean_squared_error(y_test, y_pred_lasso))


# The above plots show the actual data for each of the features vs "SalePrice", along with the predicted data values when using Ridge, Lasso and Linear regression. These methods effectiveness' are measured mean squared area, and based upon these values, the linear regression model is overall the best model for predicting the target.

# In[304]:


y=houses['SalePrice']
X=houses[['OverallQual','GarageCars','GarageArea','GrLivArea','TotalBsmtSF','1stFlrSF']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

ridge_regression = Ridge()
ridge_regression.fit(X_train, y_train)
y_pred_ridge = ridge_regression.predict(X_test)

lasso_regression = Lasso()
lasso_regression.fit(X_train, y_train)
y_pred_lasso = lasso_regression.predict(X_test)

LiR = LinearRegression()
LiR.fit(X_train, y_train)
y_pred = LiR.predict(X_test)

print("Linear")
print(LiR.coef_)
print("Ridge")
print(ridge_regression.coef_)
print("LASSO")
print(lasso_regression.coef_)


# The above values represent which of features are the best parameters for predicting sale price. In this instance, the best two features for this type of prediction are the OverallQual and GarageCars variables, which make sense, as they represent a house being a higher quality, or simply having a larger garage to pair with being a larger house.

# ## Categorical Variables

# In[287]:


ohehouses = pd.get_dummies(houses, columns = ['SaleCondition'])
salehouses=ohehouses[['SaleCondition_Abnorml','SaleCondition_AdjLand','SaleCondition_Alloca','SaleCondition_Family','SaleCondition_Normal','SaleCondition_Partial']]


# The above lines convert the categorical feature of "SaleCondition" by creating new columns that have 1's if one outcome of the feature is achieved and 0's if not. This process is better shown in the next cell.

# In[288]:


print(salehouses)
print(houses['SalePrice'])


# After creating these new columns, the coefficients for each of the regression models were found to see which of these newly created columns were the best predictors of "SalePrice".

# In[290]:


y=houses['SalePrice']
X=salehouses

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

ridge_regression = Ridge()
ridge_regression.fit(X_train, y_train)
y_pred_ridge = ridge_regression.predict(X_test)

lasso_regression = Lasso()
lasso_regression.fit(X_train, y_train)
y_pred_lasso = lasso_regression.predict(X_test)

LiR = LinearRegression()
LiR.fit(X_train, y_train)
y_pred = LiR.predict(X_test)

print("Linear")
print(LiR.coef_)
print("Ridge")
print(ridge_regression.coef_)
print("LASSO")
print(lasso_regression.coef_)


# Based on the above coefficients, the "Partial" and "Normal" options for "SaleCondition" were the best two at predicting the sale price of a house.

# ## Conclusion

# Overall, the linear model was the best of the three different regression models at predicting the sale price of a house. Thus, if I had to predict the price of any given house, I would use the linear model, and the "OverallQual" and "GarageCars" variable in order to predict the most accurately.
