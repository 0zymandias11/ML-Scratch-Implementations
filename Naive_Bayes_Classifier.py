#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np 
import pandas as pd
from math import floor
from numpy import mean
import random
from numpy import  std
from numpy import var


# In[144]:


means=[]
stds=[]
vars=[]
def remove_outliers(data1):
    for column in data1:
        means.append(mean(data1[column]))
        stds.append(std(data1[column]))
        vars.append(var(data1[column]))
#     print("means ",means)
#     print("std ",stds)
    
    for i in range(len(data1)):
        count=0
        for j in range(len(data1.columns)-1):

            if(data1.iloc[i,j] >(means[j]+3*stds[j])):
                count+=1
        if(count>len(data1.columns)):
            data1.drop(data1.index[i])
    return data1


# In[145]:


def categorical_encode(data1):
    label=[]
    for i in range(len(data1.columns)):
        label.append(i)
#     print("label ",label,"len ",len(label))
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


# In[146]:


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior


# In[147]:


def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y


# In[148]:


def naive_bayes_categorical(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 


# In[149]:


def accuracy(y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy


# In[150]:


def kFold(data)->list:
#     print(floor(len(data)/10))
    k  = floor(len(data)/10)
    folds=[]
    for i in range(0,len(data)//k):
        folds.append(data.loc[i*k:(i+1)*k])
#     print(len(folds))
    return folds


# In[197]:


def gaussian_density(class_idx, x):     
    '''
    calculate probability from gaussian density function (normally distributed)
    we will assume that probability of specific target value given specific class is normally distributed 
    
    probability density function derived from wikipedia:
    (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
    '''
#         print("len ",len(means),len(vars))
    m = means[class_idx]
    v = vars[class_idx]
    numerator = np.exp(float((-1/2))*(max(((x-m)**2).any(),0.001)) / (2 * v))
    # numerator = np.exp(-((x-mean)**2 / (2 * var)))
    denominator = np.sqrt(2 * np.pi * v)
    prob = numerator / denominator
    return prob


# In[198]:


def calc_posterior_laplace(x,prior):
    posteriors = []
#     print("p ",prior[0])
    # calc"ulate posterior probability for each class
    for i in range(len(prior)):
#         print("prior len ",len(prior))
        priors = np.log(prior[i]) ## use the log to make it more numerically stable
        conditional = np.sum(np.log(gaussian_density(i, x))) # use the log to make it more numerically stable
        posterior = (priors + conditional+1)/(300)
        posteriors.append(posterior)
    # return class with highest posterior probability


# In[199]:


def predict_laplace(features,prior):
    # preds = self.calc_posterior(f) for f in features.to_numpy()
    preds = []
    for f in features.to_numpy():
      preds.append(calc_posterior_laplace(f,prior))


# In[200]:


def Laplace(train_vals,test_vals):
    maxx=0
    for i in range(len(train_vals)):
        prior=[]
        X_test = test_vals[i].iloc[:,:-1].values
        Y_test = test_vals[i].iloc[:,-1].values
        Y_pred = naive_bayes_categorical(train_vals[i], X=X_test, Y="Outcome")
        prior = calculate_prior(train_vals[i],"Outcome")
#         print("prior ",len(prior))
        predictions = predict_laplace(test_vals[i].iloc[:, :-1],prior)
        maxx+=accuracy(test_vals[i].iloc[:, -1], Y_pred)
    print("accuracy with Laplace: ",(maxx/len(train_vals))*100)


# In[155]:


data = pd.read_csv("Dataset_E.csv")
data1 = data.copy()
data1 = remove_outliers(data1)
data1=data1.drop(['Outcome'], axis=1)
data1 = categorical_encode(data1)

data1[data.columns[len(data.columns)-1]] = data[data.columns[len(data.columns)-1]]
# print(data1)

folds = kFold(data1)

train_vals=[]
test_vals=[]

for i in range(0,10):
    temp = []
    j=0
    
    for m in range(0,8):
        j = random.randrange(10)
        temp.append(folds[j])
        
    train_vals.append(pd.concat(temp))
    
    j = random.randrange(10)
    
    test_vals.append(folds[j])
# print(train_vals)
# print(test_vals)
    
maxx = 0.0

for i in range(len(train_vals)):
    
    X_test = test_vals[i].iloc[:,:-1].values
    Y_test = test_vals[i].iloc[:,-1].values
    Y_pred = naive_bayes_categorical(train_vals[i], X=X_test, Y="Outcome")
#     if maxx < (accuracy(test_vals[i].iloc[:, -1], Y_pred)):
    maxx+=accuracy(test_vals[i].iloc[:, -1], Y_pred)
    
print("accuracy: ",(maxx/len(train_vals))*100)


# In[201]:


##Predictions with Laplace Correction
Laplace(train_vals,test_vals)


# In[31]:


# data[data.columns[len(data.columns)-1]]
# # print(data.columns[len(data.columns)-1])


# In[ ]:




