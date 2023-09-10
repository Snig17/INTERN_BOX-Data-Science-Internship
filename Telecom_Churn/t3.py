#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os


# In[2]:


df = pd.read_excel(r"C:\Users\Snigdha\New folder\Studies\Intern-Box\Notes\Tasks\telecom_churn (2).xlsx")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


corr = df.corr()


# In[7]:


sns.countplot(x='Churn',data=df)


# In[8]:


df['Churn'].plot(kind='density', figsize=(14,6))


# In[9]:


fig = plt.figure(figsize=(8,8))
plt.matshow(corr, cmap='RdBu', fignum=fig.number)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
plt.yticks(range(len(corr.columns)), corr.columns);


# In[10]:


#df['data_usedd'] =  df['DataUsage'] / df['DataPlan']


# In[11]:


#df['data_usedd'].head()


# In[12]:


df.columns


# In[13]:


y = df['Churn']


# In[14]:


X = df.drop('Churn',axis=1)


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve, log_loss,roc_auc_score,confusion_matrix


# In[17]:


model = LogisticRegression()


# In[18]:


model.fit(X_train, y_train)


# In[19]:


y_pred = model.predict(X_test)


# In[20]:


print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[21]:


precision = precision_score(y_test, y_pred, average='weighted')
print('Precision:', precision)


# In[22]:


recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', recall)


# In[23]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve of Churn')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[24]:


cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[25]:


import pickle


# In[27]:


filename = 'C:\\Users\\Snigdha\\New folder\\Studies\\Intern-Box\\Notes\\Tasks\\D-2\\train.sav'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:




