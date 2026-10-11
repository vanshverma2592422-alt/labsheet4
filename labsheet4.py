import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame.copy()

# Separate numerical features and target
feature_names = iris.feature_names
X = df[feature_names].copy()
y = df["target"].copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DataFrames for convenient analysis
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names
)

# ============================================================
# PROGRAM 1: Load a Dataset Using Pandas
# ============================================================

dataset = pd.DataFrame(iris.data, columns=feature_names)
print(dataset.head())


# ============================================================
# PROGRAM 2: Display First and Last Five Records
# ============================================================

print(dataset.head(5))
print(dataset.tail(5))


# ============================================================
# PROGRAM 3: Explore Information, Statistics, and Data Types
# ============================================================

print(dataset.info())
print(dataset.describe())
print(dataset.dtypes)


# ============================================================
# PROGRAM 4: Select Relevant Numerical Features
# ============================================================

numerical_features = dataset.select_dtypes(
    include=[np.number]
)

print(numerical_features.head())


# ============================================================
# PROGRAM 5: Standardize Dataset Using StandardScaler
# ============================================================

standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(
    numerical_features
)

standardized_df = pd.DataFrame(
    standardized_data,
    columns=numerical_features.columns
)

print(standardized_df.head())


# ============================================================
# PROGRAM 6: Implement the K-Means Clustering Algorithm
# ============================================================

kmeans_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans_model.fit_predict(X_scaled)


# ============================================================
# PROGRAM 7: Apply K-Means with K = 2
# ============================================================

kmeans_2 = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

labels_k2 = kmeans_2.fit_predict(X_scaled)


# ============================================================
# PROGRAM 8: Apply K-Means with K = 3
# ============================================================

kmeans_3 = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels_k3 = kmeans_3.fit_predict(X_scaled)


# ============================================================
# PROGRAM 9: Compare Different Values of K
# ============================================================

k_values = range(2, 8)
kmeans_results = {}

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    kmeans_results[k] = {
        "labels": labels,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels)
    }

comparison_df = pd.DataFrame({
    "K": list(kmeans_results.keys()),
    "Inertia": [
        result["inertia"]
        for result in kmeans_results.values()
    ],
    "Silhouette Score": [
        result["silhouette"]
        for result in kmeans_results.values()
    ]
})

print(comparison_df)


# ============================================================
# PROGRAM 10: Determine Optimal K Using the Elbow Method
# ============================================================

inertias = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 11: Visualize K-Means Clusters
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels_k3,
    cmap="viridis"
)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("K-Means Clustering")
plt.show()


# ============================================================
# PROGRAM 12: Display K-Means Cluster Centroids
# ============================================================

centroids = kmeans_3.cluster_centers_

centroids_df = pd.DataFrame(
    centroids,
    columns=feature_names
)

print(centroids_df)


# ============================================================
# PROGRAM 13: Assign Cluster Labels to Original Dataset
# ============================================================

clustered_df = dataset.copy()
clustered_df["Cluster"] = labels_k3

print(clustered_df.head())


# ============================================================
# PROGRAM 14: Compare Clustering Before and After Scaling
# ============================================================

kmeans_unscaled = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels_unscaled = kmeans_unscaled.fit_predict(X)

kmeans_scaled = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels_scaled = kmeans_scaled.fit_predict(X_scaled)

silhouette_unscaled = silhouette_score(
    X, labels_unscaled
)

silhouette_scaled = silhouette_score(
    X_scaled, labels_scaled
)

print("Unscaled Silhouette Score:", silhouette_unscaled)
print("Scaled Silhouette Score:", silhouette_scaled)


# ============================================================
# PROGRAM 15: Implement Agglomerative Hierarchical Clustering
# ============================================================

hierarchical_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_labels = hierarchical_model.fit_predict(
    X_scaled
)


# ============================================================
# PROGRAM 16: Generate a Dendrogram Using SciPy
# ============================================================

linkage_matrix = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30
)

plt.xlabel("Data Points or Groups")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()


# ============================================================
# PROGRAM 17: Hierarchical Clustering with Ward Linkage
# ============================================================

ward_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

ward_labels = ward_model.fit_predict(X_scaled)


# ============================================================
# PROGRAM 18: Hierarchical Clustering with Complete Linkage
# ============================================================

complete_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="complete"
)

complete_labels = complete_model.fit_predict(X_scaled)


# ============================================================
# PROGRAM 19: Compare Different Linkage Methods
# ============================================================

linkage_methods = ["ward", "complete", "average"]
linkage_results = {}

for method in linkage_methods:
    model = AgglomerativeClustering(
        n_clusters=3,
        linkage=method
    )

    labels = model.fit_predict(X_scaled)

    linkage_results[method] = silhouette_score(
        X_scaled, labels
    )

linkage_comparison = pd.DataFrame(
    list(linkage_results.items()),
    columns=["Linkage Method", "Silhouette Score"]
)

print(linkage_comparison)


# ============================================================
# PROGRAM 20: Visualize Hierarchical Clustering Results
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=ward_labels,
    cmap="plasma"
)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Hierarchical Clustering - Ward Linkage")
plt.show()


# ============================================================
# PROGRAM 21: Compare Hierarchical Clustering with K-Means
# ============================================================

kmeans_score = silhouette_score(
    X_scaled, labels_k3
)

hierarchical_score = silhouette_score(
    X_scaled, hierarchical_labels
)

algorithm_comparison = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical Clustering"],
    "Silhouette Score": [
        kmeans_score,
        hierarchical_score
    ]
})

print(algorithm_comparison)


# ============================================================
# PROGRAM 22: Apply PCA to Reduce Dataset Dimensions
# ============================================================

pca_model = PCA()

pca_transformed = pca_model.fit_transform(X_scaled)

print("Original Shape:", X_scaled.shape)
print("PCA Shape:", pca_transformed.shape)


# ============================================================
# PROGRAM 23: Reduce Features to Two Principal Components
# ============================================================

pca_2 = PCA(n_components=2)

X_pca = pca_2.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1",
             "Principal Component 2"]
)

print(pca_df.head())


# ============================================================
# PROGRAM 24: Visualize PCA Data in Two Dimensions
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Two-Dimensional Visualization")
plt.show()


# ============================================================
# PROGRAM 25: Display Explained Variance Ratio
# ============================================================

explained_variance = pca_2.explained_variance_ratio_

variance_df = pd.DataFrame({
    "Principal Component": [
        "PC1", "PC2"
    ],
    "Explained Variance Ratio": explained_variance
})

print(variance_df)


# ============================================================
# PROGRAM 26: Determine Cumulative Explained Variance
# ============================================================

pca_full = PCA()
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

print("Cumulative Explained Variance:")
print(cumulative_variance)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 27: Compare Dataset Before and After PCA
# ============================================================

original_dimensions = X_scaled.shape[1]
reduced_dimensions = X_pca.shape[1]

print("Original Number of Features:", original_dimensions)
print("Reduced Number of Features:", reduced_dimensions)

print("Original Dataset Shape:", X_scaled.shape)
print("Reduced Dataset Shape:", X_pca.shape)


# ============================================================
# PROGRAM 28: Perform K-Means on PCA-Transformed Data
# ============================================================

kmeans_pca = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels_pca = kmeans_pca.fit_predict(X_pca)


# ============================================================
# PROGRAM 29: Compare Clustering Before and After PCA
# ============================================================

score_original = silhouette_score(
    X_scaled, labels_scaled
)

score_pca = silhouette_score(
    X_pca, labels_pca
)

pca_comparison = pd.DataFrame({
    "Dataset": ["Original Standardized Data", "PCA Data"],
    "Silhouette Score": [
        score_original,
        score_pca
    ]
})

print(pca_comparison)


# ============================================================
# PROGRAM 30: Calculate Silhouette Score
# ============================================================

silhouette_value = silhouette_score(
    X_scaled, labels_k3
)

print("Silhouette Score:", silhouette_value)


# ============================================================
# PROGRAM 31: Compare Silhouette Scores for Different K
# ============================================================

silhouette_scores = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled, labels
    )

    silhouette_scores.append(score)

silhouette_df = pd.DataFrame({
    "Number of Clusters": range(2, 11),
    "Silhouette Score": silhouette_scores
})

print(silhouette_df)

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 32: Analyze Characteristics of Each Cluster
# ============================================================

analysis_df = dataset.copy()
analysis_df["Cluster"] = labels_k3

cluster_statistics = analysis_df.groupby(
    "Cluster"
)[feature_names].mean()

print(cluster_statistics)

cluster_descriptions = analysis_df.groupby(
    "Cluster"
)[feature_names].describe()

print(cluster_descriptions)


# ============================================================
# PROGRAM 33: Visualize Cluster Distributions
# ============================================================

plot_df = dataset.copy()
plot_df["Cluster"] = labels_k3.astype(str)

sns.pairplot(
    plot_df,
    vars=feature_names,
    hue="Cluster",
    diag_kind="hist"
)

plt.show()


# ============================================================
# PROGRAM 34: Save the Clustered Dataset as a CSV File
# ============================================================

final_dataset = dataset.copy()
final_dataset["Cluster"] = labels_k3

final_dataset.to_csv(
    "iris_clustered_dataset.csv",
    index=False
)


# ============================================================
# PROGRAM 35: Compare K-Means and Hierarchical Clustering
# ============================================================

kmeans_final = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_final_labels = kmeans_final.fit_predict(
    X_scaled
)

hierarchical_final = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_final_labels = hierarchical_final.fit_predict(
    X_scaled
)

kmeans_final_score = silhouette_score(
    X_scaled, kmeans_final_labels
)

hierarchical_final_score = silhouette_score(
    X_scaled, hierarchical_final_labels
)

final_comparison = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Hierarchical Clustering"
    ],
    "Silhouette Score": [
        kmeans_final_score,
        hierarchical_final_score
    ]
})

print(final_comparison)
