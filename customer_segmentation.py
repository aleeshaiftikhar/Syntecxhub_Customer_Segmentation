import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Mall_Customers.csv")

print("Dataset loaded successfully!")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nCustomer count by gender:")
print(df["Gender"].value_counts())

print("\n--- Data Cleaning ---")
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("\nMissing values after cleaning:")
print(df.isnull().sum())
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

print("\nSelected features:")
print(X.head())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled feature data (first 5 rows):")
print(X_scaled[:5])

print("\nScaled data shape:")
print(X_scaled.shape)

from sklearn.cluster import KMeans

inertia_values = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()

plt.savefig("elbow_method.png", dpi=300)
plt.show()

print("\nInertia values:")
for k, inertia in zip(k_values, inertia_values):
    print(f"K = {k}: {inertia:.2f}")


from sklearn.cluster import KMeans

optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

df.to_csv("customer_segments.csv", index=False)

print("\nClustered dataset saved successfully!")

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segments using K-Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig("customer_clusters.png", dpi=300)
plt.show()

print("\nCustomer cluster graph saved as customer_clusters.png")

cluster_summary = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean")
).round(2)

print("\n--- Customer Segment Analysis ---")
print(cluster_summary)

cluster_summary.to_csv("cluster_summary.csv")

print("\nCluster summary saved as cluster_summary.csv")
