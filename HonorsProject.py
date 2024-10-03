# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:52:11 2024

@author: Dnyaneshwari...

"""

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Load the dataset
df = pd.read_csv('C:/Honours Data Science/housing.csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Data Cleaning
df.head(5)
df = df.dropna()
print(df.shape)
df_filltered = df.drop(columns=['latitude','longitude','ocean_proximity'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Data Preprocessing
corr_df = df_filltered.corrwith(df_filltered['median_house_value'],method='pearson')
corr_df
y = df['median_house_value']
X = df.drop(columns=['median_house_value'])
print(X)
print(y)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Exploratory Data Analysis
ax = X.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
X.describe()
dummies = pd.get_dummies(X.ocean_proximity, dtype=float)
dummies.head(5)
X = X.drop(columns=['ocean_proximity'], axis=1)
X = pd.concat([X, dummies], axis=1)
print(X)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(y_train)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Selecting Model
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1,1))
X_test = x_scaler.transform(X_test)
y_test = y_scaler.transform(np.array(y_test).reshape(-1,1))
ax = X.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
reg = LinearRegression()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Traing the model
reg.fit(X_train, y_train)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Predicting the values
y_preds = reg.predict(X_test)
y_preds = y_scaler.inverse_transform(y_preds)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Testing the data
y_test = y_scaler.inverse_transform(y_test)
print(y_preds,y_test)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Principle Component Analysis
pca = PCA(n_components=2)
pca.fit_transform(X_train)
Xt = pca.fit_transform(X_train)
plot = plt.scatter( Xt[:,0], Xt[:,1], c=y_train)
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("Percentage of variance captured: ",pca.explained_variance_ratio_*100)
#Percentage of variance captured:  [30.22935486 19.76054975]
print("Strength of each PCA components: ",pca.singular_values_)
#Strength of each PCA components:  [237.0826019  191.68351157]
plt.scatter(y_test, y_preds, edgecolor='orange', alpha=0.5, color='blue')
plt.xlabel("Actual House Price")
plt.ylabel("Predicated House Price")
plt.grid()
#from that we can say the actual price lies in between 2lac-6lac
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Mean Absolute percentage error.
# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_preds)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_preds, squared=False)

# Calculate R-squared
r2 = r2_score(y_test, y_preds)

# Display the metrics
print(f"MAPE: {mape:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
'''
MAPE: 0.2928
RMSE: 69325.2734
R-squared: 0.6448
'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~