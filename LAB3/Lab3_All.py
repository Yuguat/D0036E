import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans

# Load your dataset
df_original = pd.read_csv("inc_vs_rent.csv", index_col=0)
df = df_original
String_Clean = df["region"]

Cleaned_string = []
for item in String_Clean:
    Cleaned_string.append(item.split()[0])

df["region"] = Cleaned_string

print(df)
plt.plot(df["Annual rent sqm"], df["Avg yearly inc KSEK"], "b.", linewidth=2)
plt.xlabel("$Annual rent sqm$", fontsize=10)
plt.ylabel("$Avg yearly inc KSEK$", rotation=90, fontsize=10)
plt.show()


def kmeans(data, k, max_iterations=10):
    # Convert DataFrame to a NumPy array (assuming numeric columns)
    data_array = data.to_numpy()

    # Step 1: Initialize centroids randomly
    centroids = data_array[np.random.choice(data_array.shape[0], k, replace=False)]

    for iteration in range(max_iterations):
        # Step 2: Assign each data point to the nearest centroid
        distances = np.linalg.norm(data_array[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 3: Update centroids to be the mean of the points in each cluster
        new_centroids = np.array([data_array[labels == i].mean(axis=0) for i in range(k)])

        # Check if the centroids have changed significantly
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# Function to calculate the inertia for a given k
def calculate_inertia(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data)
    return kmeans.inertia_


# Assuming you have your data in a DataFrame called df
# Specify the number of clusters (k)
k1 = 3

# Perform K-means clustering
labels, centroids = kmeans(df[['Annual rent sqm', 'Avg yearly inc KSEK']], k=k1)

# Assign cluster labels to the DataFrame
df['Cluster'] = labels

# Create a cluster plot
plt.figure(figsize=(10, 6))
for cluster_num in range(k1):
    cluster_data = df[df['Cluster'] == cluster_num]
    plt.scatter(cluster_data['Annual rent sqm'], cluster_data['Avg yearly inc KSEK'],
                label=f'Cluster {cluster_num + 1}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroids')

plt.xlabel('Annual rent sqm')
plt.ylabel('Avg yearly inc KSEK')
plt.title('K-means Clustering')
plt.legend(loc="upper center",prop = { "size": 8 })
plt.grid()

# Calculate inertia for different cluster counts
inertia_values = []

for k in range(2, 11):
    inertia = calculate_inertia(df[['Annual rent sqm', 'Avg yearly inc KSEK']], k=k)
    inertia_values.append(inertia)

    # Add the inertia value to the plot
    #plt.text(0.7, 0.2, f'Inertia: {inertia:.2f}', transform=plt.gca().transAxes, fontsize=16)

plt.text(0.7, 0.2, f'Inertia: {inertia_values[k1 - 2]:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.show()


def average_intra_cluster_distance(data, labels, cluster_label):
    cluster_data = data[labels == cluster_label]
    if len(cluster_data) == 0:
        return 0
    return np.mean(pairwise_distances(cluster_data, cluster_data))


def average_inter_cluster_distance(data, labels, cluster_label):
    cluster_data = data[labels == cluster_label]
    other_clusters = np.unique(labels[labels != cluster_label])
    min_inter_distance = float('inf')
    for other_cluster in other_clusters:
        other_cluster_data = data[labels == other_cluster]
        distance = np.mean(pairwise_distances(cluster_data, other_cluster_data))
        if distance < min_inter_distance:
            min_inter_distance = distance
    return min_inter_distance


def silhouette_coefficient(data, labels):
    s_values = []
    for i in range(len(data)):
        a_i = average_intra_cluster_distance(data, labels, labels[i])
        b_i = average_inter_cluster_distance(data, labels, labels[i])
        s_i = (b_i - a_i) / max(a_i, b_i)
        s_values.append(s_i)
    return np.mean(s_values)


def grid_search_silhouette_score(data):
    silhouette_scores = []
    for k in range(2, 11):
        labels, _ = kmeans(data, k=k)  # Assuming you have a kmeans function as defined earlier
        silhouette = silhouette_coefficient(data, labels)
        silhouette_scores.append(silhouette)
    return silhouette_scores


# Perform grid search and calculate silhouette scores
silhouette_scores = grid_search_silhouette_score(df[['Annual rent sqm', 'Avg yearly inc KSEK']])

# Graph the cluster's silhouette coefficient for each value in the grid
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Counts')
plt.grid()
plt.show()

# Determine the optimal number of clusters (e.g., by inspecting the silhouette score plot)
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we started from k=2

# Perform K-means clustering with the optimal k
optimal_labels, optimal_centroids = kmeans(df[['Annual rent sqm', 'Avg yearly inc KSEK']], k=optimal_k)

# Create a scatter plot with cluster colors
plt.figure(figsize=(10, 6))
for cluster_num in range(optimal_k):
    cluster_data = df[optimal_labels == cluster_num]
    plt.scatter(cluster_data['Annual rent sqm'], cluster_data['Avg yearly inc KSEK'],
                label=f'Cluster {cluster_num + 1}')

# Plot centroids
plt.scatter(optimal_centroids[:, 0], optimal_centroids[:, 1], marker='X', s=200, c='black', label='Centroids')

plt.xlabel('Annual rent sqm')
plt.ylabel('Avg yearly inc KSEK')
plt.title(f'K-means Clustering (Optimal k={optimal_k})')
plt.legend(loc="upper center",prop = { "size": 8 })
plt.grid()

# Add the inertia value to the plot
plt.text(0.7, 0.2, f'Inertia: {inertia_values[optimal_k - 2]:.2f}', transform=plt.gca().transAxes, fontsize=12)

plt.show()

# Assuming you have already performed K-means clustering and have optimal_labels, optimal_centroids
# Define the new data points
new_data_points = np.array([[1010, 320.12], [1258, 320], [980, 292.4]])

# Assign the new data points to clusters using your K-means model
new_labels = np.argmin(np.linalg.norm(new_data_points[:, np.newaxis] - optimal_centroids, axis=2), axis=1)

# Plot the existing clusters with centroids
plt.figure(figsize=(10, 6))
for cluster_num in range(optimal_k):
    cluster_data = df[optimal_labels == cluster_num]
    plt.scatter(cluster_data['Annual rent sqm'], cluster_data['Avg yearly inc KSEK'],
                label=f'Cluster {cluster_num + 1}')

# Plot centroids
plt.scatter(optimal_centroids[:, 0], optimal_centroids[:, 1], marker='X', s=200, c='black', label='Centroids')

# Plot the new data points with assigned cluster colors
for i, label in enumerate(new_labels):
    plt.scatter(new_data_points[i, 0], new_data_points[i, 1], marker='o', s=100,
                label=f'New Data Point {i + 1}, Cluster {label + 1}')

plt.text(0.7, 0.2, f'Inertia: {inertia_values[optimal_k - 2]:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.xlabel('Annual rent sqm')
plt.ylabel('Avg yearly inc KSEK')
plt.title(f'K-means Clustering (Optimal k={optimal_k})')
plt.legend(loc="upper center",prop = { "size": 8 })
plt.grid()
plt.show()
