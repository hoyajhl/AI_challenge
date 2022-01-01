#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from pycaret.classification import *
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
#import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')


# In[8]:


df= pd.read_csv('C:/Dacon/당뇨/new_data.csv')
display(df.head())


# In[9]:


pd.set_option('display.max_columns', None) #데이터프레임 끝까지 다 보기
pd.set_option('display.max_rows', None)


# In[10]:


df.columns


# In[11]:


df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) #칼럼명이 이상해서 코드 추가


# In[12]:


df.columns


# In[13]:


df['gender_enc'] = np.where(df['gender'] == 'M', 0, 1)


# In[14]:


df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y.%m.%d') #포맷변경하기


# In[15]:


df.loc[:, 'date_E'] = pd.to_datetime(df['date_E'], format='%Y.%m.%d')


# In[16]:


df['delta_date'] = df['date_E'] - df['date']


# In[17]:


df['delta_date'] = df['delta_date'].astype(str)
df['delta_date']  = df['delta_date'].apply(lambda x : int(re.sub('[a-z]+','',x)))


# In[18]:


df['delta_date'] 


# In[19]:


df.isnull().sum()


# In[20]:


df.min()


# In[21]:


df.dtypes


# In[24]:


df['age'] = df['age'].apply(lambda x : np.NaN if x > 130 else ( np.NaN if x < 0 else x))
df['Ht'] = df['Ht'].apply(lambda x : np.NaN if x < 0 else x)
df['Wt'] = df['Wt'].apply(lambda x : np.NaN if x < 0 else x)
df['BMI'] = df['BMI'].apply(lambda x : np.NaN if x > 50 else ( np.NaN if x < 10 else x))
df['SBP'] = df['SBP'].apply(lambda x : np.NaN if x > 250 else ( np.NaN if x < 0 else x))
df['DBP'] = df['DBP'].apply(lambda x : np.NaN if x > 200 else ( np.NaN if x < 0 else x))
df['PR'] = df['PR'].apply(lambda x : np.NaN if x > 200 else ( np.NaN if x < 20 else x))
df['Cr'] = df['Cr'].apply(lambda x : np.NaN if x < 0 else x)
df['AST'] = df['AST'].apply(lambda x : np.NaN if x > 300 else ( np.NaN if x < 0 else x))
df['ALT'] = df['ALT'].apply(lambda x : np.NaN if x > 300 else ( np.NaN if x < 0 else x))
df['GT'] = df['GGT'].apply(lambda x : np.NaN if x < 0 else x)


# In[25]:


df['ALP'] = df['ALP'].apply(lambda x : np.NaN if x < 0 else x)
df['BUN'] = df['BUN'].apply(lambda x : np.NaN if x < 0 else x)
df['Alb'] = df['Alb'].apply(lambda x : np.NaN if x < 0 else x)
df['TG'] = df['TG'].apply(lambda x : np.NaN if x < 0 else x)
df['CrCl'] = df['CrCl'].apply(lambda x : np.NaN if x < 0 else x)
df['FBG'] = df['FBG'].apply(lambda x : np.NaN if x < 0 else x)
df['HbA1c'] = df['HbA1c'].apply(lambda x : np.NaN if x > 15 else ( np.NaN if x < 0 else x))
df['LDL'] = df['LDL'].apply(lambda x : np.NaN if x < 0 else x)
df['HDL'] = df['HDL'].apply(lambda x : np.NaN if x < 0 else x)
df = df.dropna(subset=['FBG','HbA1c'],how='any')


# In[26]:


df.columns


# In[27]:


columns_bf=df.columns


# In[28]:


df.isnull().sum()


# In[29]:


# NaN 값 평균으로 채우기
NUMERIC_COLS = ['HbA1c','FBG','TG','LDL','HDL','Alb','BUN',
               'Cr','CrCl','AST','ALT','GT','ALP','TC','PR','DBP','SBP','BMI','Wt','Ht','age']
df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df.median())


# In[30]:


df.isnull().sum()


# In[31]:


df.dtypes


# In[111]:


import seaborn as sns


# In[32]:


df.columns


# In[33]:


df.dtypes


# In[34]:


x_columns_1 = ['CDMID',  'age', 'Ht', 'Wt', 'BMI', 'SBP',
       'DBP', 'PR', 'HbA1c', 'FBG', 'TC', 'TG', 'LDL', 'HDL', 'Alb', 'BUN',
       'Cr', 'CrCl', 'AST', 'ALT', 'GT', 'ALP',
       'gender_enc', 'delta_date']


# In[35]:


bohun_train_1=df[x_columns_1] 


# In[36]:


bohun_y_train_1=df['label']


# In[37]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(bohun_train_1,bohun_y_train_1, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[39]:


train_y_1


# In[40]:


train_1.head()


# In[41]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정


# In[42]:


model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[43]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[44]:


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(test_y_1, predict1)
print(cf)


# ## SMOTE 구분

# In[45]:


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot


# In[46]:


# 모델설정
sm=SMOTE(random_state=0)
# train데이터를 넣어 복제함
X_resampled_1, y_resampled_1 = sm.fit_resample(bohun_train_1,bohun_y_train_1)


# In[47]:


def count_and_plot(y): 
    counter = Counter(y)
    for k,v in counter.items():
        print('Class=%d, n=%d (%.3f%%)' % (k, v, v / len(y) * 100))
    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()
    


# In[48]:


count_and_plot(y_resampled_1)


# In[49]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(X_resampled_1,y_resampled_1, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[50]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[51]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[52]:


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(test_y_1, predict1)
print(cf)


# In[139]:


from sklearn.metrics import roc_auc_score
roc_auc_score(test_y_1, predict1)


# In[140]:


model1.feature_importances_  ## 모델링 feature importance 값들 확인 


# In[142]:


list_column = []   ##컬럼 리스트
list_fi = []   ##featureimportance 리스트
for i,j in zip(X_resampled_1.columns,model1.feature_importances_):
    list_column.append(i)
    list_fi.append(j)


# In[143]:


## feature importance 시각화 
plt.rcParams["figure.figsize"] = (5,25)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(list_fi)), list_fi, color='b', align='center')
plt.yticks(range(len(list_column)), list_column)
plt.xlabel('Relative Importance')


# In[144]:


## feature importance 상위 피쳐들 선택해서 df(데이터 프레임)으로 만드는 작업
df_importance = pd.DataFrame(list_column, columns=['list_column'])
df_importance


# In[145]:


df_importance['list_fi'] = list_fi


# In[146]:


df_importance.sort_values('list_fi',ascending=False)   ##list_fi 값대로 descending (sort_values)


# In[147]:


df_importance.sort_values('list_fi',ascending=False)   ##list_fi 값대로 descending (sort_values)


# In[150]:


sns.countplot(x="FBG", data=df)
plt.rcParams["figure.figsize"] = (1,1)
plt.title("distribution")
plt.show()


# In[202]:


plt.figure(figsize=(15, 15))
continuous_val=['FBG','age','HbA1c','delta_date']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[ ]:


#(1) 공복혈당(FBG) 126 mg/dL 이상, 또는 (2) 당화혈색소(HbA1c) 6.5% 이상


# In[185]:


np.where(df['FBG']>=120)


# In[186]:


np.where(df['HbA1c']>=6.5)


# In[174]:


is_diabetes==True


# In[189]:


df.groupby(['label'])['HbA1c'].median()


# In[190]:


df.groupby(['label'])['FBG'].median()


# In[203]:


df.groupby(['label'])['delta_date'].median()


# In[180]:


len(df[is_diabetes])


# In[181]:


len(df[fbg_126])


# ## Feature selection

# In[192]:


x_columns_2 = [  'age', 'Ht', 'Wt', 'BMI', 'SBP',
       'DBP', 'PR', 'HbA1c', 'FBG', 'TG',  'HDL', 
       'Cr', 'CrCl', 'ALT', 'GT', 'ALP',
       'gender_enc', 'delta_date']


# In[193]:


bohun_train_1=df[x_columns_2] 


# In[194]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(bohun_train_1,bohun_y_train_1, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[195]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[196]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[208]:


def HeatMap(df,x=True):
        correlations = df.corr()
        ## Create color map ranging between two colors
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5, annot=x, cbar_kws={"shrink": .75})
        fig.set_xticklabels(fig.get_xticklabels(), rotation = 90, fontsize = 10)
        fig.set_yticklabels(fig.get_yticklabels(), rotation = 0, fontsize = 10)
        plt.tight_layout()
        plt.show()


# In[210]:


HeatMap(bohun_train_1,x=True)


# In[214]:


palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'black'

fig = plt.figure(figsize=(12,8))

ax1 = sns.scatterplot(x = df['FBG'], y = df['age'], hue = "label",
                    data = df, palette = palette, edgecolor=edgecolor)

plt.annotate('N1', size=25, color='black', xy=(80, 30), xytext=(60, 35),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.plot([50, 90], [30, 30], linewidth=2, color = 'red')
plt.plot([90, 90], [20, 30], linewidth=2, color = 'red')
plt.plot([50, 90], [20, 20], linewidth=2, color = 'red')
plt.plot([50, 50], [20, 30], linewidth=2, color = 'red')
plt.title('FBG vs Age')
plt.show()


# In[220]:


palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'black'

fig = plt.figure(figsize=(12,8))

ax1 = sns.scatterplot(x = df['FBG'], y = df['HbA1c'], hue = "label",
                    data = df, palette = palette, edgecolor=edgecolor)

plt.annotate('N1', size=25, color='black', xy=(80, 30), xytext=(60, 35),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.plot([90,130], [6.4, 6.4], linewidth=2, color = 'red')
plt.plot([130, 130], [5.3, 6.4], linewidth=2, color = 'red')
plt.plot([90, 130], [5.3, 5.3], linewidth=2, color = 'red')
plt.plot([90, 90], [5.3, 6.4], linewidth=2, color = 'red')
plt.title('FBG vs HbA1c')
plt.show()


# In[235]:


palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'black'

fig = plt.figure(figsize=(12,8))

ax1 = sns.scatterplot(x = df['age'], y = df['HbA1c'], hue = "label",
                    data = df, palette = palette, edgecolor=edgecolor)

plt.annotate('N1', size=25, color='black', xy=(80, 30), xytext=(60, 35),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.title('FBG vs HbA1c')
plt.show()


# In[236]:


df['HbA1c'].value_counts()


# In[237]:


df['label'].value_counts()


# In[238]:


palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'black'

fig = plt.figure(figsize=(12,8))

ax1 = sns.scatterplot(x = df['BMI'], y = df['TG'], hue = "label",
                    data = df, palette = palette, edgecolor=edgecolor)

plt.annotate('N1', size=25, color='black', xy=(80, 30), xytext=(60, 35),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.title('FBG vs HbA1c')
plt.show()


# ## Feature Engineering 

# In[53]:


df['DMrisk_try_1']=df['FBG']+df['BMI']+df['TG']-df['HDL']   # 'DMrisk_try_1': 단순하게 FBS BMI TG는 더하고 HDL은 빼는 식으로


# In[54]:


##40-60에 해당하는 index
age_healthy=df[(df['age']<=64) &(df['age']>=40)].index
age_old=df[(df['age']<=130) &(df['age']>=65)].index
age_young=df[(df['age']<=40) &(df['age']>=0)].index


# In[315]:


df.loc[age_healthy,'DMrisk_try_2']=df.loc[age_healthy,'FBG']+df.loc[age_healthy,'BMI']+df.loc[age_healthy,'TG']-df.loc[age_healthy,'HDL']-12.066 
df.loc[age_old,'DMrisk_try_2']=df.loc[age_old,'FBG']+df.loc[age_old,'BMI']+df.loc[age_old,'TG']-df.loc[age_old,'HDL']-12.534
df.loc[age_young,'DMrisk_try_2']=df.loc[age_young,'FBG']+df.loc[age_young,'BMI']+df.loc[age_young,'TG']-df.loc[age_young,'HDL']-11.703


# In[321]:


df['DMrisk_try_3']=1.963*df['FBG']+0.023*df['BMI']+0.158*df['TG']-0.894*df['HDL'] 


# In[55]:


df.loc[age_healthy,'DMrisk_try_4']=0.1295*df.loc[age_healthy,'FBG']+0.0538*df.loc[age_healthy,'BMI']+0.0014*df.loc[age_healthy,'TG']-0.0168*df.loc[age_healthy,'HDL']-16.0846 
df.loc[age_old,'DMrisk_try_4']=0.1072*df.loc[age_old,'FBG']+0.0416*df.loc[age_old,'BMI']+0.00283*df.loc[age_old,'TG']-0.00218*df.loc[age_old,'HDL']-14.3535
df.loc[age_young,'DMrisk_try_4']=0.1609*df.loc[age_young,'FBG']+0.0532*df.loc[age_young,'BMI']+0.00386*df.loc[age_young,'TG']-0.00782*df.loc[age_young,'HDL']-19.5128
       


# In[56]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_4']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[57]:


df.groupby(['label'])['DMrisk_try_4'].median()


# In[318]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_2']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[623]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_4']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[58]:


df['DMrisk_try_4_cat'] = df['DMrisk_try_4'].apply(lambda x: 3 if x <-5 else (2 if x < -2 else (1 if x < 0 else 0)))


# In[63]:


df['DMrisk_try_4_cat_2'] = df['DMrisk_try_4'].apply(lambda x: 4 if x <-6 else (3 if x < -4 else (2 if x < -2 else(1 if x<0 else 0))))


# In[59]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_4_cat']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[64]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_4_cat_2']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[320]:


df.groupby(['label'])['DMrisk_try_2'].median()


# In[316]:


df['DMrisk_try_2']


# In[396]:


df.loc[(df['DMrisk_try_4']<100),'DMrisk_try_5']=0
df.loc[(df['DMrisk_try_4']>=100)&(df['DMrisk_try_4']<150),'DMrisk_try_5']=1
df.loc[(df['DMrisk_try_4']>=150)&(df['DMrisk_try_4']<200),'DMrisk_try_5']=2
df.loc[(df['DMrisk_try_4']>=200),'DMrisk_try_5']=3


# In[397]:


df.loc[856,'DMrisk_try_4']


# In[375]:


pip install plotly


# In[376]:


sns.catplot(x="DMrisk_try_5", y="total_bill", kind="label", data=tips)


# In[382]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_5']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[ ]:


## dyslipidemia 관련 지수 계발 


# In[109]:


df['DMrisk_try_7']=(df['TC']+df['TG']+df['LDL'])-100*df['HDL'] # 'DMrisk_try_1': 단순하게 FBS BMI TG는 더하고 HDL은 빼는 식으로ㅍㅎ


# In[124]:


df['DMrisk_try_9']=df['DBP'].apply(lambda x: 1 if x > 90 else 0)& df['SBP'].apply(lambda x: 1 if x > 140 else 0)


# In[125]:


df['DMrisk_try_9'].value_counts()


# In[131]:


plt.figure(figsize=(15, 15))
continuous_val=['HDL']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[132]:


plt.figure(figsize=(15, 15))
continuous_val=['TG']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[6]:


if 0<=lbdhdd<=40 or lbdldl>=160 or lbxtr>=200  or  lbxtc>=240 then dyslipidemia=1;


# In[110]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_7']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[111]:


plt.figure(figsize=(15, 15))
continuous_val=['LDL']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[ ]:





# In[95]:


##'HbA1c', 'FBG'
df['DMrisk_try_8']=df['HbA1c']*df['FBG']+df['BMI']


# In[96]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_8']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[97]:


df['DMrisk_try_8_cat'] = df['DMrisk_try_8'].apply(lambda x: 1 if x >490 else (0))


# In[ ]:





# In[126]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_8_cat']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[127]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_9']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[282]:


def age_strat(x):
    if 40<=x<=64: return -12.066+1.963*df['FBG']+0.023*df['BMI']-0.894*df['HDL']+0.158*df['TG']
    elif 65<=x<=120: return -12.534+1.963*df['FBG']+0.023*df['BMI']-0.894*df['HDL']+0.158*df['TG']
    elif 0<=x<40: return -11.703+1.963*df['FBG']+0.023*df['BMI']-0.894*df['HDL']+0.158*df['TG']
    else: return -12.534+1.963*df['FBG']+0.023*df['BMI']-0.894*df['HDL']+0.158*df['TG']


# In[288]:


age_strat(50)


# In[275]:


plt.figure(figsize=(15, 15))
continuous_val=['DMrisk_try_1']
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df['label'] == 0][column].hist(bins=35, color='blue', label='Diabetes = NO', alpha=0.6)
    df[df['label'] == 1][column].hist(bins=35, color='red', label='Diabetes= YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[276]:


df.groupby(['label'])['DMrisk_try_1'].median()


# In[66]:


x_columns_2 = [  'age', 'Ht', 'Wt', 'BMI', 'SBP',
       'DBP', 'PR', 'HbA1c', 'FBG', 'TG',  'HDL', 
       'Cr', 'CrCl', 'ALT', 'GT', 'ALP',
       'gender_enc', 'delta_date','DMrisk_try_4']


# In[65]:


x_columns_3 = [  'age', 'Ht', 'Wt', 'BMI', 'SBP',
       'DBP', 'PR', 'HbA1c', 'FBG', 'TG',  'HDL', 
       'Cr', 'CrCl', 'ALT', 'GT', 'ALP',
       'gender_enc', 'delta_date','DMrisk_try_4','DMrisk_try_4_cat']


# In[70]:


bohun_train_1=df[x_columns_3] 


# In[71]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(bohun_train_1,bohun_y_train_1, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[72]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[409]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[410]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[412]:


pip install xgbse


# In[425]:


get_ipython().system('pip install --user matplotlib==3.3.0')


# In[414]:


# importing model and utils from xgbse
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured


# In[436]:


# splitting to X, y format
X = bohun_train_1.drop(['delta_date'], axis=1)
y = convert_to_structured(df['delta_date'],df['label'])


# In[539]:


# fitting xgbse model
xgbse_model=XGBSEDebiasedBCE()


# In[437]:


y


# In[439]:


xgbse_model.fit(X, y)


# In[440]:


# predicting
event_probs = xgbse_model.predict(X)
event_probs.head()


# In[528]:


event_probs.tail()


# In[542]:


x_columns_survival = [  'age', 'Ht', 'Wt', 'BMI', 'SBP',
       'DBP', 'PR', 'HbA1c', 'FBG', 'TG',  'HDL', 
       'Cr', 'CrCl', 'ALT', 'GT', 'ALP',
       'gender_enc','DMrisk_try_5','delta_date']


# In[543]:


bohun_train_1=df[x_columns_survival] 


# In[544]:


X_train=bohun_train_1.drop(['delta_date'], axis=1)
y_train = convert_to_structured(bohun_train_1['delta_date'],df['label'])


# In[566]:


TIME_BINS = np.arange(150, 4340, 200) #150부터 4370까지 200씩 증가
TIME_BINS


# In[564]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(X_train,y_train, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[562]:


bohun_train_1['delta_date'].max()


# In[567]:


# fitting xgbse model
from xgbse import XGBSEDebiasedBCE
xgbse_model = XGBSEDebiasedBCE()
xgbse_model.fit(train_1, train_y_1, time_bins=TIME_BINS)


# In[571]:


# predicting
survival= xgbse_model.predict(test_1)
survival.head()


# In[ ]:


# fitting xgbse model
from xgbse import XGBSEDebiasedBCE
xgbse_model = XGBSEDebiasedBCE()
xgbse_model.fit(train_1, train_y_1, time_bins=TIME_BINS)


# In[574]:


plt.figure(figsize=(6,8))
survival.mean().plot.line()


# In[600]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,4), dpi=120)

plt.plot(
    survival.columns,
    survival.iloc[37],
    'k--',
    label='Survival'
)

plt.title('Sample of predicted survival curves - $P(T>t)$')
plt.legend()


# In[593]:


from xgbse import XGBSEKaplanNeighbors
xgbse_model =XGBSEKaplanNeighbors(n_neighbors=30)
xgbse_model.fit(train_1, train_y_1, time_bins=TIME_BINS)


# In[594]:


# predicting
survival= xgbse_model.predict(test_1)
survival.head()


# In[595]:


survival.tail()


# In[596]:


plt.figure(figsize=(4,6))
survival.mean().plot.line()


# In[601]:


from xgbse.extrapolation import extrapolate_constant_risk

# extrapolating predicted survival
survival_ext = extrapolate_constant_risk(survival, 4500, 15)
survival_ext.head()


# In[608]:


# plotting extrapolation #

plt.figure(figsize=(12,4), dpi=120)

plt.plot(
    survival.columns,
    survival.iloc[39],
    'k--',
    label='Survival'
)

plt.plot(
    survival_ext.columns,
    survival_ext.iloc[39],
    'tomato',
    alpha=0.5,
    label='Extrapolated Survival'
)

plt.title('Extrapolation of survival curves')
plt.legend()


# In[619]:


import matplotlib.pyplot as plt
plt.style.use('bmh')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# to easily plot confidence intervals
def plot_ci(mean, upper_ci, lower_ci, i=37, title='Probability of survival $P(T \geq t)$'):

    # plotting mean and confidence intervals
    plt.figure(figsize=(12, 4), dpi=120)
    plt.plot(mean.columns,mean.iloc[i])
    plt.fill_between(mean.columns, lower_ci.iloc[i], upper_ci.iloc[i], alpha=0.2)

    plt.title(title)
    plt.xlabel('Time [days]')
    plt.ylabel('Probability')
    plt.tight_layout()


# In[620]:


mean, upper_ci, lower_ci = xgbse_model.predict(test_1, return_ci=True)


# In[622]:


# plotting CIs
plot_ci(mean, upper_ci, lower_ci,i=59)

