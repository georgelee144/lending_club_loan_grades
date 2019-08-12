#!/usr/bin/env python
# coding: utf-8

# This not will attempt to classify the loan's grade. There are 2 ways we can classify this one verses all, such as grade A or not, or multi classification. This note book will just classify one verse all for all grade loans

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('data/lending_club_ml.csv')


# In[3]:


#dropping variables that are useless or foward looking
df.drop(['id','loan_status','int_rate','sub_grade'],axis=1,inplace=True)


# In[4]:


df = pd.concat([df, pd.get_dummies(df.home_ownership,
                                   prefix='home_ownership', drop_first=True)], axis=1)
df = pd.concat([df, pd.get_dummies(df.verification_status,
                                   prefix='verification_status', drop_first=True)], axis=1)
df = pd.concat(
    [df, pd.get_dummies(df.purpose, prefix='purpose', drop_first=True)], axis=1)
df = pd.concat([df, pd.get_dummies(df.verification_status_joint,
                                   prefix='verification_status_joint', drop_first=True)], axis=1)

df.drop(columns=['home_ownership', 'verification_status',
                 'purpose', 'verification_status_joint'], inplace=True)

df.disbursement_method = df.disbursement_method.apply(
    lambda disburstment: 1 if disburstment == 'Cash' else 0)

df.application_type = df.application_type.apply(
    lambda application_type: 1 if application_type == 'Joint' else 0)

df.info()


# In[5]:


df.fillna(0,inplace=True)


# In[6]:


from sklearn.model_selection import train_test_split

X = df.drop(['grade'], axis=1)
y = df.grade

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

logit.fit(X_train,y_train)


# In[ ]:


print(confusion_matrix(y_test, logit.predict(X_test)))

