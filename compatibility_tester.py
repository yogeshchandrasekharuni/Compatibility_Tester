'''
@author: Yogesh Chandrasekharuni

Dataset: "Young People Survey" from Kaggle
Link to dataset: https://www.kaggle.com/miroslavsabo/young-people-survey/version/2#responses.csv 

'''


# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('responses.csv')
dataset = dataset.fillna(1)
X_train = dataset.iloc[:800, 0:70].values
X_test = dataset.iloc[800:, 0:70].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit(X_train)

result = kmeans.predict(X_test)

likeminded_people = [[], [], [], [], []]

for i in range(len(result)):
    if result[i] == 0:
        likeminded_people[0].append(i)
    elif result[i] == 1:
        likeminded_people[1].append(i)
    elif result[i] == 2:
        likeminded_people[2].append(i)   
    elif result[i] == 3:
        likeminded_people[3].append(i)
    else:
        likeminded_people[4].append(i)        

#2 people having same cluster number in result may be assumed be to like-minded