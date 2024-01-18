#!/usr/bin/env python
# coding: utf-8

# In[148]:


from sklearn.datasets import make_classification
import pandas as pd


# In[149]:


X,y = make_classification(n_samples = 1000,
                                       n_features = 5,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       flip_y=0.1,
                                       class_sep=1,
                                       random_state=23,
                                       weights = [0.06])


# In[150]:


import pandas as pd
 
# Create DataFrame with features as columns
dataset = pd.DataFrame(X)
# give custom names to the features
dataset.columns = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
# Now add the label as a column
dataset['y'] = y
 
dataset.head()


# In[151]:


dataset['y'].value_counts()


# In[152]:


#SMOTE - RESAMPLING
from collections import Counter
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)


# In[153]:


#plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=50, edgecolor="k"); # Single LOC to make PLOT for the whole dataset
## MAKE PLOT
from pandas import DataFrame
from matplotlib import pyplot

df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key])
    pyplot.legend(["1", "0"])
pyplot.show()


# In[154]:


import seaborn as sns 

# plotting histograms 
pyplot.hist(df['x'], 
            alpha=0.9,
            color='Red',
         label='1') 
pyplot.hist(df['y'], 
            alpha=0.5,
            color='Blue',
         label='0') 


pyplot.legend(loc='upper right') 
pyplot.title('Overlapping') 
pyplot.show()


# In[155]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


# In[156]:


len(X_train),len(X_test),len(y_train),len(y_test)


# In[157]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=23)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)


# In[158]:


# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[159]:


from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')


# In[160]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[161]:


from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# In[ ]:




