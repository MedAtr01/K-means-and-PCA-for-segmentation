#1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
#2

np.random.seed(42)
data = np.random.rand(300, 5)
df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(data.shape[1])])
#3

print(df.head())
print("Taille des données:", df.shape)
#4

kmeans_random = KMeans(n_clusters=3, init='random', n_init=10, random_state=42)
kmeans_random.fit(df)
labels_random = kmeans_random.labels_
kmeans_plus = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans_plus.fit(df)
labels_plus = kmeans_plus.labels_
#5

score_random = calinski_harabasz_score(df, labels_random)
score_plus = calinski_harabasz_score(df, labels_plus)
print(f"Calinski-Harabasz Score (Random): {score_random}")
print(f"Calinski-Harabasz Score (K-means++): {score_plus}")
#7

if score_random > score_plus:
    best_model = 'Random Initialization'
else:
    best_model = 'K-means++ Initialization'
print(f"Le meilleur modèle est basé sur: {best_model}")
#8

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_scaled)
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['Cluster_Random'] = labels_random
pca_df['Cluster_Plus'] = labels_plus
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Random', data=pca_df, palette='viridis', ax=ax[0])
ax[0].set_title('Clusters avec Initialisation Aléatoire')
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Plus', data=pca_df, palette='viridis', ax=ax[1])
ax[1].set_title('Clusters avec Initialisation K-means++')
plt.show()
#9

explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
components = pca.components_
print("Valeurs propres :", explained_variance)
print("Inertie de chaque axe :", explained_variance_ratio)
print("Somme des inerties :", np.sum(explained_variance_ratio))
print("Vecteurs propres :", components)