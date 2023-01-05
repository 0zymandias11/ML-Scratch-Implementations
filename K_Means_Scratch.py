#!/usr/bin/env python
# coding: utf-8

# In[30]:


#import random libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[31]:


#reading data through txt file using pandas
data1 = pd.read_csv("seeds_dataset.txt", header=None, delimiter=r"\s+")
data1.columns =['area', 'perimeter', 'compactness', 'length_of_kernel','width_of_kernel', 'asyymetry_coeff', 'len_kernel_groove', 'type_of_wheat']


# In[32]:


data1


# In[33]:


data1.info()


# In[34]:


column_types = data1.dtypes
column_types


# In[35]:


data1['type_of_wheat'].value_counts()


# In[36]:


#standardize the data 

from sklearn.preprocessing import StandardScaler
standardize_data=StandardScaler().fit_transform(data1)
standardize_data.shape


# In[37]:


X= data1.values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)


# In[39]:


pca_95 = PCA(n_components = 0.95,random_state = 2020)
pca_95.fit(X_scaled)
X_pca_95 = pca_95.transform(X_scaled)


# In[40]:


#data after standardization
X_pca_95


# In[41]:


X_pca_95.shape


# In[42]:


plt.figure(figsize = (10,7))
plt.plot(X_pca_95)
plt.title('Standardized data by PCA with 95% variability')
plt.xlabel('Dataset')
plt.ylabel("Standardized Data")
plt.savefig('PCA graph.png')
plt.show()


# In[43]:



pca_k =PCA(n_components = 0.95)
pca_k.fit(standardize_data)
pca_k_data = pca_k.transform(standardize_data)
#pca_k_data


# In[44]:


print(pca_k.explained_variance_ratio_ *100)


# In[67]:


#plt.plot(np.cumsum(pca_k.explained_variance_ratio_))


# In[45]:


print(np.cumsum(pca_k.explained_variance_ratio_*100)[0])
print(np.cumsum(pca_k.explained_variance_ratio_*100)[1])
print(np.cumsum(pca_k.explained_variance_ratio_*100)[2])
print(np.cumsum(pca_k.explained_variance_ratio_*100)[3])


# In[15]:


#plt.figure(figsize = (10,7))
#import seaborn
#seaborn.scatterplot(x= pca_k_[:,0],y = pca_k_[:,1])


# In[46]:


#fig = plt.figure(figsize=(10,8))
#sub = fig.add_subplot(projection='3d')
#y=data1['type_of_wheat']
#sub.scatter(pca_k_data[:,0], pca_k_data[:,1], pca_k_data[:,2], c=y)
#sub.set_title("Three particular features of the PCA dataset")
#sub.set_xlabel("First principal component")
#sub.set_ylabel("First principal component")
#sub.set_zlabel("First principal component")
#plt.show()


# In[46]:


#shape after the PCA implementation 
pca_k_data.shape


# In[47]:


#copying numpy array received after PCA for further use in kmeans algo and NMI calculation algo 
X=pca_k_data
X
Y= pca_k_data 
type(Y)


# In[48]:


#appending outcome label to numpy array for NMI Calculation use
import math
Z = Y.tolist()
for i in range(70):
    Z[i].append(1)
for i in range(70,140):
    Z[i].append(2)
for i in range(140,210):
    Z[i].append(3)
Z = np.array(Z)
#Z


# In[49]:


#X = pca_k_data
#function for calculating kmeans
def kmeann(X,k):
    counts =[]
    #index = []
    #selecting k random centers
    for i in range(k):
        row_num = X.shape[0]
        rand_rows = np.random.choice(row_num, size=1, replace=False)
        counts.append(X[rand_rows])
    for t in range(100):
        arr =[[]for v in range(k)]
        for i in range(210):
            min = 0
            indexx = -1
            for j in range(k):
                new_m = math.sqrt(((X[i][0]-counts[j][0][0])**2)+((X[i][1]-counts[j][0][1])**2)+((X[i][2]-counts[j][0][2])**2)+((X[i][3]-counts[j][0][3])**2))
                if j==0:
                    min = new_m
                    indexx = 0
                else:
                    if new_m<min:
                        min = new_m
                        indexx=j
            arr[indexx].append(X[i])
        for j in range(k):
            for c in range(3):
                sum=0
                for i in range(np.asarray(arr[j]).shape[0]):
                    sum = sum+ np.asarray(arr[j])[i][c]
                sum = sum/np.asarray(arr[j]).shape[0]
                counts[j][0][c] = sum
                
    return arr,counts


# In[50]:


def countt(arrs):
    count1 =0
    count2=0
    count3=0
    for i in range(len(arrs)):
        if int(arrs[i][4]) ==1:
            count1=count1+1
        elif int(arrs[i][4]) ==2:
            count2=count2+1
        else:
            count3=count3+1
    return count1,count2,count3


# In[52]:


#calling function 
#dry run of function for k means
arrs,counts = kmeann(Z,3)
counts
#print(np.asarray(arrs[0]).shape[0])
#print(np.asarray(arrs[1]).shape)
#print(np.asarray(arrs[2]).shape)
#arrrs = arrs[0]
#arrrs[0][4]


# In[51]:


#for calculating NMI formula is NMI(Y,C) = 2*I(Y;C)/[H(Y)+H(C)]
#Calculating H(Y) First

dax = data1['type_of_wheat'].value_counts().tolist()
total_elements =0
for i in range(0, len(dax)):
    total_elements = total_elements + dax[i]
total_elements
H_y =-(dax[0]/total_elements)*math.log((dax[0]/total_elements),2)-(dax[1]/total_elements)*math.log((dax[1]/total_elements),2)-(dax[2]/total_elements)*math.log((dax[2]/total_elements),2)
H_y


# In[55]:


list_k = []
list_Hc = []
for k in range(2,9):
    arr,counts = kmeann(Z,k)
    sum =0
    class_ent =0
    for i  in range(k):
        count1,count2,count3 = countt(arr[i])
        total = count1+count2+count3
        #print(count1)
        #print(count2)
        #print(count3)
        if count1==0:
            x=0
        else:
            x = ((count1/total)*math.log((count1/total),2))
        if count2==0:
            y=0
        else:
            y =((count2/total)*math.log((count2/total),2)) 
        if count3==0:
            z=0
        else:
            z = ((count3/total)*math.log((count3/total),2))
        sum = sum + ((-1/k)*(x+y+z))
        class_ent = class_ent+((-total/210)*math.log((total/210),2))
    #print("The band for", k, "is", sum)
    list_k.append(sum)
    list_Hc.append(class_ent)
        
    


# In[56]:


#for each value of k , we are calculatin the value of NMI using formula given in comments above

for i in range(0, len(list_k)):
    list_k[i] = 2*(H_y - list_k[i])
    #print(list_k[i])
    list_k[i] = list_k[i]/(H_y+list_Hc[i])



#NMI value for each k from 2 to 8
list_k
file2 = open("NMI values for corresponding k means.txt","w")
for i in range(0, len(list_k)):
    file2.write('Value of k =' + str(i+2) + ', Value of NMI ' +str(list_k[i]))
    file2.write('\n')
file2.close()


# In[57]:


k_num = [2,3,4,5,6,7,8]


# In[58]:


plt.plot(k_num, list_k)
plt.xlabel('number of clusters')
plt.ylabel('value of NMI')
plt.title('k-means vs clusters')
  
# function to show the plot
plt.savefig('NMIgraph_vs_kmeans.png')
plt.show()


# In[59]:


# sample run to inbuilt Kmeans function to check whether our NMI output is correct 
#since inertia is min at K=3 , so , the best clustering is observed at 3 , same as in our NMI case
from sklearn.cluster import KMeans
wcss =[]
for i in range(2,9):
    kmeans_pca=KMeans(n_clusters=i,init = 'k-means++',random_state=42)
    kmeans_pca.fit(pca_k_data)
    wcss.append(kmeans_pca.inertia_)
    
plt.figure(figsize=(10,8))
plt.plot(range(2,9),wcss,marker ='o' , linestyle ='--')

plt.xlabel("Numbers of Clusters")
plt.ylabel("WCSS")

