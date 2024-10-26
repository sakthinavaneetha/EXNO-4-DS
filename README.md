# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE

       import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
![image](https://github.com/user-attachments/assets/a25056da-dfce-4eda-b833-fbf386db71fb)


data.isnull().sum()
![image](https://github.com/user-attachments/assets/b8bb4365-c898-4f76-a464-4b215e14ebd8)


missing=data[data.isnull().any(axis=1)]
missing
![image](https://github.com/user-attachments/assets/702f0647-44f1-448e-9c46-2a9f9f468439)


data2=data.dropna(axis=0)
data2
![image](https://github.com/user-attachments/assets/b61bae67-7c82-4f06-931e-4e9e1786cce1)


sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
![image](https://github.com/user-attachments/assets/487c9aa6-94e7-43f3-bad4-99110baa4109)


sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
![image](https://github.com/user-attachments/assets/4d67a974-7c05-4956-be64-a57cf79984eb)


data2
![image](https://github.com/user-attachments/assets/0169674a-4ec0-4db0-b47c-e3870b570e20)


new_data=pd.get_dummies(data2, drop_first=True)
new_data
![image](https://github.com/user-attachments/assets/d6537085-bdf3-4f62-92e3-b0ce985828bf)


columns_list=list(new_data.columns)
print(columns_list)
![image](https://github.com/user-attachments/assets/02c11de8-914c-4859-a94b-fcc6da9f3c1d)


features=list(set(columns_list)-set(['SalStat']))
print(features)
![image](https://github.com/user-attachments/assets/ac346068-477f-405a-b5f7-5791dc38a5e0)


y=new_data['SalStat'].values
print(y)
![image](https://github.com/user-attachments/assets/c4933057-1cb7-4148-ba58-d7906a6e0b90)


x=new_data[features].values
print(x)
![image](https://github.com/user-attachments/assets/debe340f-874d-40c6-806a-5c62d5c311b2)


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
![image](https://github.com/user-attachments/assets/eae06a2d-aea9-492a-9484-502b66effb65)


prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
![image](https://github.com/user-attachments/assets/bec8e77b-8d31-40a4-a899-083825699bdb)


accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
![image](https://github.com/user-attachments/assets/399813d5-f5af-4371-a0d8-96fa2e422a13)


print("Misclassified Samples : %d" % (test_y !=prediction).sum())
![image](https://github.com/user-attachments/assets/1c753f3b-afca-4d06-bd6b-939c6a9fb5c3)


data.shape
![image](https://github.com/user-attachments/assets/6efebc02-3fb9-4dea-9d53-55f8f57f4377)


import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
image

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
![image](https://github.com/user-attachments/assets/41ec6737-ab7f-47b6-bdb4-b457aae49e54)


tips.time.unique()
![image](https://github.com/user-attachments/assets/4a930174-27d7-42a5-b780-9adecd8283e8)


contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
![image](https://github.com/user-attachments/assets/63e13d7f-97f8-4a69-accb-2e9f6e9f1cfc)


chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
![image](https://github.com/user-attachments/assets/b03df278-6d4a-4a91-90e0-1d5c8b9be48d)

# RESULT:
       Thus, Feature selection and Feature scaling has been used on the given dataset.
