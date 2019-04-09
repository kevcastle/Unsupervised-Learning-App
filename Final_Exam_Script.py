# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:17:56 2019

@author: Kevin
"""

##############################################################################
#Code For Data Analysis
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans # k-means clustering

survey_df = pd.read_excel('finalExam_Mobile_App_Survey_Data_final_exam-2.xlsx')

survey_df.columns

for col in enumerate (survey_df):
    print (col)
    
    
# renaming columns for further analysis
survey_df.rename(columns={'q1':'age',
                          'q48':'education',
                          'q49':'married',                          
                          'q54' : 'race',
                          'q55' : 'latino',
                          'q56': 'income',
                          'q57' : 'gender'}, 
                 inplace=True)

########################
# Step 1: Remove demographic information
########################

survey_reduced = survey_df.iloc[ : , 2:-11 ]


    
#Reversing Questions 
survey_reduced['q12'].value_counts()


survey_reduced['rev_q12'] = -100

survey_reduced['rev_q12'][survey_reduced['q12'] == 1] = 6
survey_reduced['rev_q12'][survey_reduced['q12'] == 2] = 5
survey_reduced['rev_q12'][survey_reduced['q12'] == 3] = 4
survey_reduced['rev_q12'][survey_reduced['q12'] == 4] = 3
survey_reduced['rev_q12'][survey_reduced['q12'] == 5] = 2
survey_reduced['rev_q12'][survey_reduced['q12'] == 6] = 1

survey_reduced['rev_q12'].value_counts()

#after reversing the column, drop the old column
survey_reduced = survey_reduced.drop(columns = ['q12'],
                             axis = 1)

#seeing the new subset columns and position
for col in enumerate (survey_reduced):
    print (col)



#renmaing the q2 columns 
survey_reduced.rename(columns={survey_reduced.columns[0]: "iPhone", 
                               survey_reduced.columns[1]: "iPod touch",
                               survey_reduced.columns[2]: "Android",
                               survey_reduced.columns[3]: "Blackberry",
                               survey_reduced.columns[4]: "Nokia",
                               survey_reduced.columns[5]: "Windows",
                               survey_reduced.columns[6]: "HP/Palm",
                               survey_reduced.columns[7]: "Tablet",
                               survey_reduced.columns[8]: "Other_q2",
                               survey_reduced.columns[9]: "q2_none"}, 
                               inplace=True)

########################
# Step 2: Scale to get equal variance
########################

from sklearn.preprocessing import StandardScaler # standard scaler

scaler = StandardScaler()


scaler.fit(survey_reduced)


X_scaled_reduced = scaler.transform(survey_reduced)



########################
# Step 3: Run PCA without limiting the number of components
########################

from sklearn.decomposition import PCA # principal component analysis

pca_step3 = PCA(n_components = None,
                           random_state = 508)


pca_step3.fit(X_scaled_reduced)


X_pca_reduced = pca_step3.transform(X_scaled_reduced)



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(pca_step3.n_components_)


plt.plot(features,
         pca_step3.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()




########################
# Step 5: Run PCA again based on the desired number of components
########################

customer_pca_reduced = PCA(n_components = 5,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

fac_load_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))


factor_loadings_df = fac_load_df.set_index(survey_reduced.columns)


print(factor_loadings_df)


factor_loadings_df.to_excel('survey_final_factor_loadings.xlsx')

########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)


###############################################################################
# Combining PCA and Clustering
###############################################################################

########################
# Step 1: Take transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 6,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())




########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Simple Needs','Music',	'Non_Tech_Needs',
                            'Serious','Websites']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('survey_pca_centriods_final.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)

clst_pca_df.columns = ['cluster','Simple_Needs','Music','Non_Tech_Needs',
                            'Serious','Websites']

########################
# Step 6: Reattach demographic information
########################

age_df = pd.concat([survey_df.iloc[ : , 1],
                                clst_pca_df],
                                axis = 1)

final_pca_clust_df = pd.concat([survey_df.iloc[ : , 77:],
                                age_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))



########################
# Step 7: Analyze in more detail 
########################

# Adding a productivity step
data_df = final_pca_clust_df



########################
# Needs by Clusters
########################

# age and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'age',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('age_and_music.png')

"""
Boxplots are in similar position but the median for Age 1 is lower than the 
rest indetifying a need for music with ages under 18 
"""


# gender and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'gender',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()



# education and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'education',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()

# marriage status and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'married',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()

#race and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'race',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('race_and_music.png')

#latino and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'latino',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('latino_and_music.png')

#income and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'income',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('income_and_music.png')



########################
# Serious
########################

# age and serious
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'age',
            y = 'Serious',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('age_and_serious.png')



# education and serious
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'education',
            y = 'Serious',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('education_and_serious.png')



# married and serious
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'married',
            y = 'Serious',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('married_and_serious.png')

#race and serious
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'race',
            y = 'Serious',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('married_and_serious.png')

#latino and serious
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'latino',
            y = 'Serious',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('latino_and_serious.png')

#income and serious
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'income',
            y = 'Serious',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('income_and_serious.png')

#gender and serious
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'gender',
            y = 'Serious',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('gender_and_serious.png')

########################
# Simple Needs
########################

# age and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'age',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('age_and_simple.png')

# education and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'education',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('education_and_simple.png')

# married and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'married',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('married_and_simple.png')

# married and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'married',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('married_and_simple.png')

# race and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'race',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('race_and_simple.png')

# latino and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'latino',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('latino_and_simple.png')

# income and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'income',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('income_and_simple.png')

# gender and simple
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'gender',
            y = 'Simple_Needs',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 5)
plt.tight_layout()
plt.show()
fig.savefig('gender_and_simple.png')

"""
Checked to see if there was any cluster but will stick with the first one
identified.

"""

##############################################################################
#Model Code
##############################################################################
centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Simple Needs','Music',	'Non_Tech_Needs',
                            'Serious','Websites']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('survey_pca_centriods_final.xlsx')

########################
#Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)

clst_pca_df.columns = ['cluster','Simple_Needs','Music','Non_Tech_Needs',
                            'Serious','Websites']

########################
#Reattach demographic information
########################

age_df = pd.concat([survey_df.iloc[ : , 1],
                                clst_pca_df],
                                axis = 1)

final_pca_clust_df = pd.concat([survey_df.iloc[ : , 77:],
                                age_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))

#Boxplot used 
# age and music
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'age',
            y = 'Music',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 3)
plt.tight_layout()
plt.show()
fig.savefig('age_and_music.png')

"""
Boxplots are in similar position but the median for Age 1 is lower than the 
rest indetifying a need for music with ages under 18 
"""

