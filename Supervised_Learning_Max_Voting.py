#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from numpy import  std
from numpy import mean
import math


# In[4]:


data=pd.read_csv('seeds_dataset.csv')
data.head()
# print(data.shape)


# <h3>Removing Outliers from data (mean +/- 3*sigma)</h3>

# In[5]:


means=[]
stds=[]
def remove_outliers(data1):
    for column in data1:
        means.append(mean(data1[column]))
        stds.append(std(data1[column]))
    print("means ",means)
    print("std ",stds)
    
    for i in range(len(data1)):
        count=0
        for j in range(len(data1.columns)-1):

            if(data1.iloc[i,j] >(means[j]+3*stds[j]) or data1.iloc[i,j] <(means[j]+3*stds[j])):
                count+=1
        if(count>len(data1.columns)):
            data1.drop(data1.index[i])
    return data1

data1=data.copy()
data1=remove_outliers(data1)
data1.head()
print(data1.shape)
# print(data.shape()+" "+data1.shape())


# In[6]:


data1=data1.drop(['type_of_wheat'], axis=1)
data1.head()


# <h3>Standard Scaler</h3>

# In[7]:


def standard_scaler(data1):
    for column in data1.columns:
        data1[column]=(data1[column]-mean(data1[column]))/(std(data1[column]))
    return data1

data1=standard_scaler(data1)
data1.head()


# <h3>categorical encoding</h3>

# In[8]:


def categorical_encode(data1):
    label=[]
    for i in range(len(data1.columns)):
        label.append(i)
    print("label ",label,"len ",len(label))
#     print(data1["Age"][0])
    length = len(data1)
#     print(length)
    column_names=[]
    for columns in data1:
        temp=[]
        column_names.append("cat_"+columns)
        for j in (0,length-1):
            temp.append(data1[columns][j])
        temp = np.array(temp)
        data1["cat_"+columns] = pd.cut(data1[columns].values, bins = len(label), labels = label)

    data1 = data1[column_names]
    return data1

data1=categorical_encode(data1)
data1.head()


# In[9]:


# data1[data.columns[len(data.columns)-1]] = data[data.columns[len(data.columns)-1]]
# data1.head()


# In[10]:


sp_f=0.8 ##Splitting factor

X= data1.copy()
Y= data.iloc[:,-1].values
X.head()

n_train = math.floor(sp_f * X.shape[0])
n_test = math.ceil((1-sp_f) * X.shape[0])
x_train = X[:n_train]
y_train = Y[:n_train]
x_test = X[n_train:]
y_test = Y[n_train:]
print("Total Number of rows in train:",x_train.shape[0])
print("Total Number of rows in test:",x_test.shape[0])


# <h3>SVM</h3>

# In[11]:


from sklearn import svm
from sklearn import metrics

###linear kernel
clf1 = svm.SVC(kernel='linear') 
clf1.fit(x_train, y_train)
y_pred = clf1.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# In[12]:


###Quadratic Kernel
clf2 = svm.SVC(kernel='poly') 
clf2.fit(x_train, y_train)
y_pred = clf2.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# In[13]:


###Radial Basis Function
clf3 = svm.SVC(kernel='rbf') 
clf3.fit(x_train, y_train)
y_pred = clf3.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# In[14]:


## We get best result for linear kernel


# <h3>MLP Classifier</h3>

# In[42]:


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements



# In[25]:


#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier

#Initializing the MLPClassifier

x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_test =  np.asarray(x_test).astype('float32')
y_test =  np.asarray(y_test).astype('float32')
classifier = MLPClassifier(hidden_layer_sizes=(16),learning_rate_init=0.001,batch_size=300,solver='sgd')
classifier2 = MLPClassifier(hidden_layer_sizes=(16,32),learning_rate_init=0.001,batch_size=300,solver='sgd')
classifier.fit(x_train, y_train)



#Predicting y for X_val
y_pred = classifier.predict(x_test)

#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_test)

#Printing the accuracy
print("Accuracy of MLPClassifier :",  '', accuracy(cm) *100)


# In[35]:


##MLP with 2 hidden layers
classifier2.fit(x_train, y_train)

#Predicting y for X_val
y_pred = classifier2.predict(x_test)

#Comparing the predictions against the actual observations in y_pred
cm = confusion_matrix(y_pred, y_test)

#Printing the accuracy
print("Accuracy of MLPClassifier :",  '', accuracy(cm) *100)


# In[43]:



# <h3>MLP with variable learning rates</h3>

# In[95]:


lr_list=[]
accuracy_list=[]

for lr in range(1,6):
    classifier3 = MLPClassifier(hidden_layer_sizes=(16,32),learning_rate_init=1/(10**lr),batch_size=300,solver='sgd')
    classifier3.fit(x_train, y_train)
    y_pred = classifier3.predict(x_test)
    cm = confusion_matrix(y_pred, y_test)
    print("Accuracy of MLPClassifier with lr:",  1/(10**lr)," : ", accuracy(cm) *100)
    accuracy_list.append(accuracy(cm) *100)
    lr_list.append(1/(10**lr))

print("accuradcy_list : ",accuracy_list)
print("lr_list : ",lr_list)


# <h3>  Accuracy vs Learning Rate Plot </h3>

# In[96]:


import matplotlib.pyplot as plt
import numpy as np
plt.plot(lr_list,accuracy_list)
plt.title('learning_rate vs accuracy')
plt.xlabel('learning_rate')
plt.ylabel('accuracy')
plt.show()


# <h3>Forward Feature Selection</h3>

# In[36]:


from sklearn.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(classifier2, 
                                n_features_to_select = 'auto', 
                                direction='forward') 


# In[37]:


sfs.fit(x_train, y_train)
print("Top features selected by forward sequential selection:{}"\
      .format(list(X.columns[sfs.get_support()])))


# <h3>Max-Voting</h3>

# In[38]:


from sklearn.ensemble import VotingClassifier


# In[40]:


eclf1 = VotingClassifier(estimators=[('svm_quad', clf2), ('svm_rbf', clf3), ('mlp', classifier2)], voting='hard')
eclf1 = eclf1.fit(x_train, y_train)
#Comparing the predictions against the actual observations in y_pred
eclf_pred = (eclf1.predict(x_test))
cm = confusion_matrix(eclf_pred, y_test)

print("Accuracy of MLPClassifier :",  '', accuracy(cm) *100)


# In[ ]:




