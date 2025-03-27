#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:22:03 2024

@author: polinagoldberg
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


'''

The program needs a user input for the .cs file from CryoSPARC. This would preferably
be done using the full file path.


All plots will display automatically when the program is run EXCEPT for covariance plots. 
Covariance plots can be created by using the covariance_plot(class1, class2) function below.


'''


# loading in data for each particle as cs
file_path = input("Please input file name or path: ")
cs = np.load(file_path) 

# extracting assignment probability column
probabilities = cs["alignments3D_multi/class_posterior"]

# finding number of classes
num_classes = probabilities.shape[1]


## NORMALIZED HISTOGRAMS

for i in range(num_classes):
    class_probabilities = probabilities[:,i]
    
    plt.hist(class_probabilities, \
             bins = 30, \
             edgecolor = 'black', \
             color = 'darkgreen', \
             density = True, \
             alpha = 0.5)
    
    plt.title(f'Class {i+1} Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.show()



## CONFUSION MATRIX

# creating a dictionary for particles assigned to each class
particle_classifications = {f'Class {i+1}': [] for i in range(num_classes)}
 

# filling dictionary with probabilities 
for i in range(len(probabilities)):
    
    # turning numpy array into list
    probabilities_list = list(probabilities[i])
    
    # finding the class to which the particle was assigned
    max_probability = max(probabilities_list)
    class_assignment = probabilities_list.index(max_probability) + 1
    
    # appendingthe probability to dictionary
    particle_classifications[f'Class {class_assignment}'].append(probabilities_list)
    


## MEANS ONLY MATRIX

# initializing matrix
mean_matrix = np.empty((num_classes,num_classes), dtype = object)


# filling matrix wih mean and std for the probability each particle in 
# class i was assigned to class j


for i in range(num_classes): #rows (i) are true assigned class
    for j in range(num_classes): #columns (j) are 'predicted' classes
        
        particles_in_class = np.array(particle_classifications[f'Class {i+1}'])
        
        if len(particles_in_class) > 0:
            
            mean_matrix[i,j] = np.mean(particles_in_class[:,j])
        
        else: 
            mean_matrix[i,j] = 0



## MEANS AND STD MATRIX

# initializing matrix
mean_and_std_matrix = np.empty((num_classes,num_classes), dtype = object)


# filling matrix wih mean and std for the probability each particle in 
# class i was assigned to class j


for i in range(num_classes): #rows (i) are true assigned class
    for j in range(num_classes): #columns (j) are 'predicted' classes
        
        particles_in_class = np.array(particle_classifications[f'Class {i+1}'])
        
        if len(particles_in_class) > 0:
            
            mean_particles_j = np.mean(particles_in_class[:,j])
            std_particles_j = np.std(particles_in_class[:,j])
            matrix_fill = f'{mean_particles_j:.2e}\n({std_particles_j:.2e})'
            mean_and_std_matrix[i,j] = matrix_fill
        
        else: 
            mean_and_std_matrix[i,j] = 'N/A'


# visualizing the matrix
fig, ax = plt.subplots()
ax.axis('off')

table_plot = ax.table(cellText=mean_and_std_matrix, \
                      loc='center', \
                      colLabels = [f'Class {i+1}' for i in range(num_classes)], \
                      rowLabels = [f'Class {i+1}' for i in range(num_classes)], \
                      cellLoc = 'center', \
                      colLoc = 'center', \
                      rowLoc = 'center')
    
# making diagonals blue and formattig
for i in range(num_classes):
    diagonal_cells = table_plot[i+1,i]
    diagonal_cells.set_facecolor('lightblue')
    col_labels = table_plot[0,i]
    col_labels.visible_edges = 'B'
    row_labels = table_plot[i+1,-1]
    row_labels.visible_edges = 'R'
    
plt.suptitle('Classification Matrix of Assigned (row) vs Predicted (columns) Class', fontsize = 12)
plt.show()


         

## SHANNON ENTROPY HISTOGRAMS

# entropy histograms for each class

for i in range(num_classes):
    
    # extracting probability distributions for each class
    particle_class_dists = particle_classifications[f'Class {i+1}']
    
    # generating a list of all of the shannon entropies for said class
    shannon_entropy_list = [stats.entropy(dist) for dist in particle_class_dists]
    
    plt.hist(shannon_entropy_list, \
             bins = 30, \
             edgecolor = 'black', \
             color = 'darkgreen', \
             density = True, \
             alpha = 0.5)
        
    plt.title(f'Shannon Entropy Distribution (Class {i+1})')
    plt.xlabel('Shannon Entropy')
    plt.ylabel('Density')
    plt.show()
    


# overall histogram
shannon_entropy_total = [stats.entropy(dist) for dist in probabilities]


plt.hist(shannon_entropy_list, \
         bins = 30, \
         edgecolor = 'black', \
         color = 'darkgreen', \
         density = True, \
         alpha = 0.5)
    
plt.title('Overall Shannon Entropy Distribution')
plt.xlabel('Shannon Entropy')
plt.ylabel('Density')
plt.show()






## SHANNNON ENTROPY AND NORAMLIZED PROBABILITY DENSITY CURVES

# normalized probability

plt.figure(figsize = (8,6))
sns.set_style('ticks')

for i in range(num_classes):
    
    class_probabilities = probabilities[:,i]
    if not all(class_probabilities) == 0:
        sns.kdeplot(class_probabilities, label = f'Class {i+1}') 

plt.title('Normalized Probability Plots by Class')
plt.legend()
plt.show()


# Shannon entropy

plt.figure(figsize = (8,6))
sns.set_style('ticks')

for i in range(num_classes):
    
    # extracting probability distributions for each class
    particle_class_dists = particle_classifications[f'Class {i+1}']
    
    # generating a list of all of the shannon entropies for said class
    shannon_entropy_list = [stats.entropy(dist) for dist in particle_class_dists]
    
    sns.kdeplot(shannon_entropy_list, label = f'Class {i+1}')
    

plt.title('Shannon Entropy Plots by Class')
plt.legend()
plt.show()

    

        
        
        
### SCATTERPLOT FUNCTION FOR COVARIANCE PLOTS

def covariance_plot(class1, class2):
    
    # class1: an integer value indicating class for x axis
    # class2: an integer value indicating class for y axis
    
    if type(class1) != int or type(class2) != int:
        return 'Arguments must be integers'
    
    
    # finding the index required for each class
    class1_index = class1-1
    class2_index = class2-1
    
    # creating a dictionary for all classes that AREN'T CLASS 1 or 2
    removed_keys = [f'Class {class1}', f'Class {class2}']
    other_particles = [value for key, value in particle_classifications.items() if key not in removed_keys]
    other_particles = [inner_list for value in other_particles for inner_list in value]
    
    # setting x and y values for each set
    class1_prob_xaxis = [lst[class1_index] for lst in particle_classifications[f'Class {class1}']]
    class1_prob_yaxis = [lst[class2_index] for lst in particle_classifications[f'Class {class1}']]
    
    class2_prob_xaxis = [lst[class1_index] for lst in particle_classifications[f'Class {class2}']]
    class2_prob_yaxis = [lst[class2_index] for lst in particle_classifications[f'Class {class2}']]
    
    other_prob_xaxis = [lst[class1_index] for lst in other_particles]
    other_prob_yaxis = [lst[class2_index] for lst in other_particles]
    
    
    plt.figure(figsize = (8,6))
    plt.scatter(class1_prob_xaxis, class1_prob_yaxis, c='green', label = f'Class {class1}', alpha = 0.3, s= 0.01)
    plt.scatter(class2_prob_xaxis, class2_prob_yaxis, c='blue', label = f'Class {class2}', alpha = 0.3, s= 0.01)
    plt.scatter(other_prob_xaxis, other_prob_yaxis, c='red', label = 'Other Classes', alpha = 0.05, s = 0.01)
    plt.xlim(0,0.5)
    plt.ylim(0,0.5)
    plt.xlabel(f'Probability of Assignment to Class {class1}')
    plt.ylabel(f'Probability of Assignment to Class {class2}')
    plt.legend(markerscale = 20)
    
    plt.show()
    
    


## K-MEANS CLUSTERING

# training the data on only nondummy particles 
# training the data on only the probabiltiies corresponding to nondummy classes (i.e. 7 features)
# predicting the class assignment with nondummies only using the probabililities corresponding to nondummy classes


from sklearn.cluster import KMeans

probabilities_df = pd.DataFrame(probabilities)
probabilities_df.columns = [f'Class {i+1}' for i in range(num_classes)]
probabilities_df.index = [i+1 for i in range(len(probabilities_df))]

# adding original assignment column
probabilities_df['Original Assignment'] = probabilities_df.idxmax(axis = 1)
probabilities_df['Original Assignment'] = probabilities_df['Original Assignment'].str.replace('Class ', '', case=False).astype(int)

#removing all particles originally assigned to dummies
probabilities_nondummies = probabilities_df[probabilities_df['Original Assignment'] > 6]

#making new probabilities numpy array for kmeans (contains no dummies)
probabilities_kmeansarray = probabilities_nondummies.drop(columns = ['Original Assignment'])
probabilities_kmeansarray = np.array(probabilities_kmeansarray)

# finding cluster centers and covariances (only being fit on nondummies and their probabilitiy assignments)
kmeans = KMeans(n_clusters = 7, init = mean_matrix[6:,6:], random_state = 0).fit(probabilities_kmeansarray[:,6:])
kmean_centers = kmeans.cluster_centers_
kmeans_cov = np.cov(probabilities[:,6:], rowvar = False)

# predicting which particles go where
kmeans_prediction = kmeans.predict(probabilities_kmeansarray[:,6:])
kmeans_prediction = kmeans_prediction + 7

# adding k means assignment column
probabilities_nondummies['K-means Assignment'] = kmeans_prediction.astype(int)

# proportion of particles that match in assigned class and kmeans class
matches = probabilities_nondummies['Original Assignment'] == probabilities_nondummies['K-means Assignment']
proportion_all = matches.mean()




### PCA VISUALIZATION FOR K MEANS

# new dataframe using sample 1000 values from the porbabilities data frame
probabilities_nondummies = probabilities_nondummies.reset_index(drop=True)
plotX = probabilities_nondummies.sample(1000, random_state=2)
plotX.columns = probabilities_nondummies.columns
plotX['K-means Assignment'] = plotX['K-means Assignment'].astype(str)
plotX['Original Assignment'] = plotX['Original Assignment'].astype(str)


from sklearn.decomposition import PCA

# PCA with 2, and 3 components (2D and 3D respectively)
pca_kmeans_2d = PCA(n_components = 2)
pca_kmeans_3d = PCA(n_components = 3)


# dataframe with just PC
pc_2d = pca_kmeans_2d.fit_transform(plotX.drop(columns = ['Original Assignment','K-means Assignment'], axis=1))
pc_3d = pca_kmeans_3d.fit_transform(plotX.drop(columns = ['Original Assignment','K-means Assignment'], axis=1))


# extracting pCA components for 2D Plot
pc1_2d = pc_2d[:,0]
pc2_2d = pc_2d[:,1]


# plotting
plt.figure(figsize=(8,6))
scatter = plt.scatter(pc1_2d, pc2_2d, c=kmeans.labels_[plotX.index]+7, cmap='viridis', alpha=0.7, edgecolors='k')

# add legend
plt.legend(*scatter.legend_elements(), title="Cluster Assignment")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters Visualized in PCA Space")
plt.show()



# extracting PCA components for 3D plot
pc1_3d = pc_3d[:, 0]
pc2_3d = pc_3d[:, 1]
pc3_3d = pc_3d[:, 2]

# plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pc1_3d, pc2_3d, pc3_3d, c=kmeans.labels_[plotX.index]+7, cmap='viridis', alpha=0.7, edgecolors='k')
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("K-Means Clusters in 3D PCA Space")
legend1 = plt.colorbar(scatter, ax=ax, pad=0.1)
legend1.set_label("Cluster Assignment")
plt.show()




from sklearn.manifold import TSNE

# run t-SNE (2D)
tsne = TSNE(n_components=2, perplexity=30, random_state=2)
tsne_results = tsne.fit_transform(plotX.drop(columns=['Original Assignment', 'K-means Assignment']))

# scatter plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans.labels_[plotX.index]+7, cmap='viridis', alpha=0.7, edgecolors='k')
plt.colorbar(label="Cluster Assignment")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of K-Means Clusters")
plt.show()




# fit PCA on original high-dimensional data
pca = PCA(n_components=7)
pca.fit(probabilities_nondummies.drop(columns=['Original Assignment', 'K-means Assignment']))

# plot explained variance
plt.figure(figsize=(8,5))
plt.plot(range(1, 8), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Principal Components')
plt.grid(True)
plt.show()




from sklearn.metrics import silhouette_score

# randomly sample 5000 points without replacement
sample_indices = np.random.choice(len(probabilities_kmeansarray), 5000, replace=False)
sampled_data = probabilities_kmeansarray[sample_indices]
sampled_labels = kmeans.labels_[sample_indices]

# compute silhouette score on sampled data
score = silhouette_score(sampled_data, sampled_labels)
print(f"Silhouette Score (Sampled 5000 Points): {score:.3f}")

  
from sklearn.metrics import davies_bouldin_score
db_index = davies_bouldin_score(sampled_data, sampled_labels)
print(f"Davies-Bouldin Index: {db_index:.3f}")






import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Assuming probabilities_df contains the original data
# Drop the non-numeric columns
data_for_pca = plotX.drop(columns=['Original Assignment', 'K-means Assignment'])

# Perform PCA (you can adjust the number of components based on explained variance)
pca = PCA(n_components=13)  
pca_result = pca.fit_transform(data_for_pca)

# Create a DataFrame for the PCA components
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# Optionally, add the cluster assignments to the DataFrame
pca_df['K-means Assignment'] = kmeans.labels_[plotX.index].astype(str)

# Create pairwise scatter plots
sns.pairplot(pca_df, hue='K-means Assignment', palette='viridis', plot_kws={'alpha':0.7})
plt.suptitle('Pairwise PCA Scatterplots', y=1.02)
plt.show()

