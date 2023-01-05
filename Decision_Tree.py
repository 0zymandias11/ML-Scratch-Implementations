#!/usr/bin/env python
# coding: utf-8

# #Preliminary imports and tree-node structure

# In[832]:


#importing all suitable libraries
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import copy
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


#Tree Node structure
class Tree_Node:
  def __init__(self,Attr_index, threshold, leaf):
    self.Attribute_index = Attr_index           #or class if leaf node
    self.threshold = threshold          #or None if leaf node
    self.leaf_node = leaf                         #True for leaf node
    self.left_P = None
    self.right_P = None
    self.class_distribution = {}        #dict
  


# #Taking the input dataset

# In[833]:


#reading data from CSV file
df = pd.read_csv("Dataset_E.csv")   
df.info()


# #Sepearting Target column

# In[834]:


#Separating Target Column

df=df.sample(frac=1)      
Target_set = np.array(df.iloc[:,-1])
Train_set = np.array(df.drop([df.columns[-1]],axis=1))
for i in range(len(Target_set)):
  if Target_set[i] == 0:
    Target_set[i] = 1
  else:
    Target_set[i] = 2


# #Spliting the Data into training and testing set

# In[835]:


#Splitting dataset into Training and Testing Set
x_train, x_test, y_train, y_test = train_test_split(Train_set, Target_set, test_size = 0.2,random_state=0)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
#x_train


# In[836]:


#class of Decision Tree Classifier

class Decision_tree:
  def __init__(self,x_train,y_train,x_test,y_test, criteria = 'Information Gain', depth = None):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.depth_limit = depth
    self.root = None
    self.criteria = criteria
    if self.depth_limit is not None:
      self.depth_limit = int(self.depth_limit)
      if self.depth_limit < 0:
        self.depth_limit = -self.depth_limit
    self.possible_prunes = []


  def calculate_entropy(self, train_y):
    uni = np.unique(train_y)
    freq = [0] * len(uni)
    id = list(range(len(uni)))
    species_dict=dict(zip(uni,id))
    for i in range(len(train_y)):
      j = species_dict[train_y[i]]
      freq[j] = freq[j] + 1
    entropy = 0
    for i in freq:
      if i == 0:
        continue
      entropy = entropy - (i/len(train_y))*math.log((i/len(train_y)),2)
    return entropy
  
  

  def split_calculate(self, train_x, train_y, threshold, atrr):
    l = len(train_y)
    temp1_y = []        #classes with atrr >= threshold
    
    temp2_y = []      #classes with attr < threshold

    s1 = 0
    s2 = 0
    for i in  range(0,l):
      if train_x[i][atrr] >= threshold:
        temp1_y.append(train_y[i])
        s1 = s1 + 1
      else:
        temp2_y.append(train_y[i])
        s2 = s2 + 1
    entr1 = 0
    entr2 = 0
    entr1 = self.calculate_entropy(copy.deepcopy(temp1_y))
    entr2 = self.calculate_entropy(copy.deepcopy(temp2_y))
    
    return [[s1, entr1],[s2,entr2]]

  def candidates(self, train_x, train_y, attr):
    mp = []
    l = len(train_y)
    for i in range(l):
      mp.append([train_x[i][attr], train_y[i]])
    mp.sort(key = lambda x: x[0])

    #create boolean attributes
    candidate_thresholds=[]
    for i in range(1,l):
      if mp[i][1] != mp[i-1][1]:
        candidate_thresholds.append((mp[i][0] + mp[i-1][0])/2)
    return candidate_thresholds

    
  def impurity_measure(self, train_x, train_y):
    l = len(train_y)
    current_entropy = 0
    current_entropy = self.calculate_entropy(copy.deepcopy(train_y))
    if current_entropy == 0:                     #check for leaf node or pure node
      return [None, None, 1]
    max_gain = [-1,-1,-1]                                     #attribute index, its threshold value, gain
    no_of_attributes = len(train_x[0])
    for i in range(no_of_attributes):
      candidate_thresholds = self.candidates(copy.deepcopy(train_x),copy.deepcopy(train_y),i)
      max_gain_j = [-1,-1, -1]    #[attr_index, threshold, gain]
      for j in candidate_thresholds:
        entropies_ = self.split_calculate(copy.deepcopy(train_x), copy.deepcopy(train_y), j, i)
        gain_j = 0 
        for k in entropies_:
          gain_j = gain_j + (k[0]/l)*k[1]
        gain_j = current_entropy - gain_j

        if max_gain_j[2] < gain_j: 
          max_gain_j = [i, j, gain_j]
      if max_gain[2] < max_gain_j[2]:
        max_gain = copy.deepcopy(max_gain_j)

    return max_gain

  def compare(self, A, B, geq):
    if geq == 1:
      return A >= B
    return A < B 

  def partition(self, train_x, train_y, attr, threshold, geq):
    tempX = []
    tempY = []
    l = len(train_x)
    for i in range(l):
      if self.compare(train_x[i][attr] ,threshold ,geq):
        tempX.append(train_x[i])
        tempY.append(train_y[i])
    return [tempX, tempY]
  
  def most_probable_class(self,train_y):
    uni = np.unique(train_y)
    freq = np.array([0] * len(uni))
    id = list(range(len(uni)))
    species_dict=dict(zip(uni,id))
    for i in range(len(train_y)):
      j = species_dict[train_y[i]]
      freq[j] = freq[j] + 1
    max_freq_index = np.argmax(freq)
    class_index = 0
    for key,value in species_dict.items():
      if max_freq_index == value:
        class_index = key
        break
    return class_index

  def R_t(self, rt):
    class_error_if_pruned = 0
    if rt.class_distribution[1] >= rt.class_distribution[2]:
      class_error_if_pruned = rt.class_distribution[2]
    else:
      class_error_if_pruned = rt.class_distribution[1]
    R_t = class_error_if_pruned/len(self.x_train)
    return R_t

  def R_T(self,rt):
    if rt.leaf_node:
      classification = rt.Attribute_index         #class of leaf node
      if classification == 1:
        return rt.class_distribution[2]
      else:
        return rt.class_distribution[1]

    return self.R_T(rt.left_P) + self.R_T(rt.right_P)


  def inorder_tree(self, rt):
    if rt.leaf_node:
      return
    
    self.inorder_tree(rt.left_P)

    R_t_value = self.R_t(rt)
    R_T_value = self.R_T(rt)/len(self.x_train)
   #number of error if subtree is kept and number of error if it become leaf through pruning
    N_t = self.find_no_of_leaf_nodes(rt)
    gain =(R_t_value - R_T_value)/(N_t-1)
    self.possible_prunes.append([gain, rt])

    self.inorder_tree(rt.right_P)

  def Reduce_Error_prunning(self):
    self.possible_prunes = []
    self.inorder_tree(self.root)

    getting_pruned = 0
    l = len(self.possible_prunes)
    max_gain=1
    for i in range(l):
      if self.possible_prunes[i][0] < max_gain:
        max_gain = self.possible_prunes[i][0]
        getting_pruned = self.possible_prunes[i][1]

    #prune subtree with from leaf node
    getting_pruned.leaf_node = True
    if getting_pruned.class_distribution[1] >= getting_pruned.class_distribution[2]:
      getting_pruned.Attribute_index = 1
    else:
      getting_pruned.Attribute_index = 2
    getting_pruned.left_P = None
    getting_pruned.right_P = None
    getting_pruned.threshold = None

  def create_dict(self,train_y):
    d = {1 : 0, 2 : 0}
    l = len(train_y)
    for i in range(l):
      d[train_y[i]] = d[train_y[i]] + 1
    return d

  def train_model(self, train_x, train_y, d):
    if self.depth_limit == d:                                               #if self.depth_limit is None then, None == int then it will be false
      return Tree_Node(self.most_probable_class(copy.deepcopy(train_y)), None, True)
    max_gain = self.impurity_measure(copy.deepcopy(train_x),copy.deepcopy(train_y))       #[attr_index, threshold, gain] 
    if max_gain[2] == 1:            #leaf node
      rt = Tree_Node(self.most_probable_class(copy.deepcopy(train_y)), None, True)
      rt.class_distribution = self.create_dict(copy.deepcopy(train_y))
      return rt
    else:
      r = Tree_Node(max_gain[0],max_gain[1],False)
      split1_x, split1_y = self.partition(copy.deepcopy(train_x), copy.deepcopy(train_y), max_gain[0], max_gain[1], 1)
      split2_x, split2_y = self.partition(copy.deepcopy(train_x), copy.deepcopy(train_y), max_gain[0], max_gain[1], 0)

      r.right_P = self.train_model(copy.deepcopy(split1_x), copy.deepcopy(split1_y),d+1)
      r.left_P = self.train_model(copy.deepcopy(split2_x), copy.deepcopy(split2_y),d+1)

      r.class_distribution = {1 : r.right_P.class_distribution[1] + r.left_P.class_distribution[1], 2 : r.right_P.class_distribution[2] + r.left_P.class_distribution[2]}
      return r

  def fit(self):
    self.root = self.train_model(copy.deepcopy(self.x_train), copy.deepcopy(self.y_train),0)
  
  def predict_class(self, test_row, rt):
    if rt.leaf_node:
      return rt.Attribute_index
    else:
      if self.compare(test_row[rt.Attribute_index], rt.threshold, 1):
        return self.predict_class(test_row,rt.right_P)
      else:
        return self.predict_class(test_row, rt.left_P)

  def predict(self,test_row):
    return self.predict_class(test_row,self.root)
  
  def score(self):
    correct_prediction = 0
    for i in range(0, len(self.y_test)):
      if self.predict(self.x_test[i]) == self.y_test[i]:
        correct_prediction = correct_prediction + 1
    return correct_prediction/len(x_test)
  
  def score_on_training_data(self):
    correct_prediction = 0
    for i in range(0, len(self.y_train)):
      if self.predict(self.x_train[i]) == self.y_train[i]:
        correct_prediction = correct_prediction + 1
    return correct_prediction/len(x_train)
  
  def tree_depth(self):
    return self.find_depth(self.root)

  def find_depth(self, rt):           #not considering leaf nodes in depth
    if rt.leaf_node:
      return 0
    l = self.find_depth(rt.left_P)
    r = self.find_depth(rt.right_P)

    return max(l,r) + 1

  def find_no_of_leaf_nodes(self, rt):      #considering leaf nodes
    if rt is None:
      return 0
    elif rt.leaf_node:
      return 1
    else:
      return self.find_no_of_leaf_nodes(rt.left_P) + self.find_no_of_leaf_nodes(rt.right_P)

  def no_of_leaf_nodes(self):
    return self.find_no_of_leaf_nodes(self.root)
  
  def find_total_nodes(self, rt):
    if rt is None:
      return 0
    elif rt.leaf_node:
      return 1
    else:
      return self.find_total_nodes(rt.left_P) + self.find_total_nodes(rt.right_P) + 1
  
  def total_nodes(self):
    return self.find_total_nodes(self.root)
  
  def prune_tree_at_depth(self, rt, limit, curr_depth, train_x, train_y):
    if rt is None:
      return
    elif rt.leaf_node:
      return
    elif curr_depth < limit:
      split1_x, split1_y = self.partition(copy.deepcopy(train_x), copy.deepcopy(train_y), rt.Attribute_index, rt.threshold, 1)
      split2_x, split2_y = self.partition(copy.deepcopy(train_x), copy.deepcopy(train_y), rt.Attribute_index, rt.threshold, 0)

      self.prune_tree_at_depth(rt.right_P, limit, curr_depth + 1, copy.deepcopy(split1_x), copy.deepcopy(split1_y))
      self.prune_tree_at_depth(rt.left_P, limit, curr_depth + 1, copy.deepcopy(split2_x), copy.deepcopy(split2_y))
    else:                                                                                                           #make it to a leaf node
      rt.leaf_node = True
      rt.Attribute_index = self.most_probable_class(copy.deepcopy(train_y))
      rt.threshold = None
      rt.left_P = None
      rt.right_P = None

  def prune_at_depth(self, d):
    self.prune_tree_at_depth(self.root, d, 0,copy.deepcopy(self.x_train) ,copy.deepcopy(self.y_train))
    self.depth_limit = d
    #print('depth = ' + str(self.find_depth(self.root)))

  def rightmost_nodes(self,rt,l):
    if rt is None:
      return
    l.append(rt)
    self.rightmost_nodes(rt.right_P,l)
  def right_mostnode_true(self,rt,l):
    for i in l:
      if i == rt:
        return True
    return False

  def BFS_On_tree(self, q):
    l = []
    self.rightmost_nodes(self.root,l)
    x = 1
    while len(q) != 0:
      rt = q.pop(0)
      if rt.leaf_node:
        print('  ||  Class: ' + str(rt.Attribute_index), end = '    ')
      else:
        print('  ||  Attribute_index[' + str(rt.Attribute_index) + '] <= ' + str(rt.threshold), end = '  ||  ')
        q.append(rt.left_P)
        q.append(rt.right_P)
      if self.right_mostnode_true(rt, l):
        print(' || ')
      x = x + 1

  def print_tree(self):
    queue = []
    queue.append(self.root)
    self.BFS_On_tree(queue)



# In[ ]:





# In[837]:


#training the tree classifier we built with training data and seeing performance

model = Decision_tree(x_train, y_train, x_test, y_test,criteria = 'Information Gain')
model.fit()
print('Accuracy: ' + str(model.score()) + ', nodes in the tree: ' + str(model.total_nodes()))
file1 = open("DecisionTree_on_Training_Data.txt","w")
L = ['Accuracy: ' + str(model.score()) + ', nodes in the tree: ' + str(model.total_nodes())] 
file1.writelines(L)
file1.close()


# #Finding accuracy over 10 random splits: Q2 solution

# In[838]:


#Finding accuracy over 10 random splits: Q2 solution

score_list = []                  #accuuracy of the tree
best_model = 0
best_score = 0
best_depth =0
models = []
no_of_nodes1 = []       #nodes of the tree
depth_of_the_tree1 = []       #depth of the tree
for _ in range(10):
  x_train, x_test, y_train, y_test = train_test_split(Train_set, Target_set, test_size = 0.2)
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  x_test = np.array(x_test)
  y_test = np.array(y_test)

 
  
  modelj = Decision_tree(x_train, y_train, x_test, y_test,criteria = 'Information Gain')
  modelj.fit()
  s = modelj.score()
  nod = modelj.total_nodes()
  dept = modelj.tree_depth()
  score_list.append(s)
  no_of_nodes1.append(nod)
  depth_of_the_tree1.append(dept)
  if best_score < s:
    best_model = copy.deepcopy(modelj)
    best_score = s 
    best_depth = dept




# In[828]:


best_model_conclusion = copy.deepcopy(best_model)


# In[842]:


for i in range(len(score_list)):
  print('Depth of the tree: ' + str(depth_of_the_tree1[i]) + ', Number of nodes: ' +str(no_of_nodes1[i]) + ', Accuracy: ' + str(score_list[i]))
file2 = open("DecisionTree_on_random10splits.txt","w")
for i in range(len(score_list)):
    file2.write('Depth of the tree: ' + str(depth_of_the_tree1[i]) + ', Number of nodes: ' +str(no_of_nodes1[i]) + ', Accuracy: ' + str(score_list[i]))
    file2.write('\n')
file2.close()


# In[ ]:





# In[843]:


print('Depth of the tree: ' + str(best_depth) +  ', Accuracy: ' + str(best_score))
file1 = open("Best_Tree_Depth_and_Accuracy.txt","w")
L = ['Depth of the tree: ' + str(best_depth) +  ', Accuracy: ' + str(best_score)] 
file1.writelines(L)
file1.close()


# In[844]:


#Part3
#Using Reduced Error Pruning method to prune the best tree out of 10 random split: Q3 solution
#we will prune the tree node wise and calculate  accuracy, if accuracy of pruned tree is greater than accuracy of unpruned tree, then we go on with pruning , otherwise stop.
best_model.fit()     
s=best_model.score()
#print(s)
Accuracy_on_test= []
Accuracy_on_training_data= []
nodes = []
depthh = []
Accuracy_on_test.append(s)
#Accuracy_on_training_data.append(best_model.score_on_training_data())
nodes.append(best_model.total_nodes())
depthh.append(best_model.tree_depth())
best_model.Reduce_Error_prunning()
s1= best_model.score()
#print(s1)
while s1 >= s:
  Accuracy_on_test.append(s1)
  nodes.append(best_model.total_nodes())
  depthh.append(best_model.tree_depth())
  s=s1
  best_model.Reduce_Error_prunning()
  s1 = best_model.score()
    

#plotting graph
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Depth')
plt.plot(depthh, Accuracy_on_test, label="On Testing data")
plt.legend(loc='best')
plt.savefig('Reduce_prune_both_testing_and_training_with_depth.png')






# In[845]:


#plotting graph
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Numbers of nodes')
plt.plot(nodes, Accuracy_on_test, label="On Testing data")
plt.legend(loc='best')
plt.savefig('Reduce_prune_both_testing_and_training_with_number_of_nodes.png')


# In[819]:


best_model.print_tree()

