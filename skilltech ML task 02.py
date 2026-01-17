import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

customers, _ = make_blobs(n_samples=250, centers=5, cluster_std=1.4, random_state=42)

df = pd.DataFrame(customers, columns=["Total_Spending", "Visit_Frequency"])

df["Total_Spending"] = (df["Total_Spending"] - df["Total_Spending"].min()) * 100
df["Visit_Frequency"] = (df["Visit_Frequency"] - df["Visit_Frequency"].min()) * 10
df["Customer_ID"] = np.arange(1, len(df) + 1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[["Total_Spending", "Visit_Frequency"]])

wcss = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker="o")
plt.title("Elbow Method (Choosing the Best K)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o")
plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

best_k = 5
final_kmeans = KMeans(n_clusters=best_k, init="k-means++", random_state=42)
df["Cluster"] = final_kmeans.fit_predict(scaled_features)

centers_scaled = final_kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers_original, columns=["Total_Spending", "Visit_Frequency"])
centers_df["Cluster"] = range(best_k)

plt.figure(figsize=(9, 6))
plt.scatter(df["Total_Spending"], df["Visit_Frequency"], c=df["Cluster"], cmap="viridis", s=60)

plt.scatter(
    centers_df["Total_Spending"],
    centers_df["Visit_Frequency"],
    c="red",
    marker="X",
    s=250,
    label="Cluster Centers"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Total Spending")
plt.ylabel("Visit Frequency")
plt.legend()
plt.grid(True)
plt.show()

cluster_summary = df.groupby("Cluster")[["Total_Spending", "Visit_Frequency"]].mean()
cluster_summary["Customers_Count"] = df["Cluster"].value_counts().sort_index()

print("\nCustomer Cluster Summary:\n")
print(cluster_summary)

df.to_csv("Customer_Clusters_Output.csv", index=False)
print("\n✅ Output saved as 'Customer_Clusters_Output.csv'")
