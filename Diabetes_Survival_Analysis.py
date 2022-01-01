#!/usr/bin/env python
# coding: utf-8

# In[1]:


from xgbse import XGBSEDebiasedBCE
import pandas as pd
import re

from xgbse.converters import convert_to_structured
from pycox.datasets import metabric
import numpy as np

from xgbse.converters import convert_to_structured
from sklearn.model_selection import train_test_split


# In[4]:


df = pd.read_excel('Diabetes_Train_Data.xlsx')
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) #칼럼명이 이상해서 코드 추가
df = df.drop('Unnamed0',1)
df


# In[5]:


df['gender'] = df['gender'].map({'M':0,'F':1}) #남자는 0, 여자는 1
df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y.%m.%d') #날짜형식으로 전환
df.loc[:, 'date_E'] = pd.to_datetime(df['date_E'], format='%Y.%m.%d')
df['delta_date'] = df['date_E'] - df['date'] #진단일 endpoint - baseline 기간
df['delta_date'] = df['delta_date'].astype(str) #string 형식으로 바꾸기
df['delta_date']  = df['delta_date'].apply(lambda x : int(re.sub('[a-z]+','',x))) #숫자(int) 형태로 바꿔서 분석하기 쉽게 저장


# In[8]:


X = df.drop(['CDMID','date','date_E'],axis = 1) #필요 없는 column 삭제
T = df['delta_date'] #시간 데이터
E = df['label'] #이벤트 데이터
y = convert_to_structured(T,E) #생존 분석의 y값은 시간 + 이벤트 : 언제 어떤 확률로 event가 발생하는 지


# In[12]:


# splitting between train, and validation 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state = 0)
TIME_BINS = np.arange(150, 4370, 200) #150부터 4370까지 200씩 증가
TIME_BINS


# In[13]:


# fitting xgbse model
xgbse_model = XGBSEDebiasedBCE()
xgbse_model.fit(X_train, y_train, time_bins=TIME_BINS)

# predicting
y_pred = xgbse_model.predict(X_test)

print(y_pred.shape)
y_pred.head()


# In[15]:


y_pred.mean().plot.line();


# In[16]:


neighbors = xgbse_model.get_neighbors(
    query_data = X_test,
    index_data = X_train,
    n_neighbors = 5
)

print(neighbors.shape)
neighbors.head(5)


# In[17]:


desired = neighbors.iloc[10]

X_test.loc[X_test.index == desired.name]


# In[18]:


X_train.loc[X_train.index.isin(desired.tolist())]


# In[19]:


# importing metrics
from xgbse.metrics import concordance_index, approx_brier_score

# running metrics
print(f"C-index: {concordance_index(y_test, y_pred)}")
print(f"Avg. Brier Score: {approx_brier_score(y_test, y_pred)}")


# In[20]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

results = cross_val_score(xgbse_model, X, y, scoring=make_scorer(approx_brier_score))
results

