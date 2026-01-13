from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("data/absenteeism_cleaned.csv")
print("Data loaded for clustering")

X = df.drop(
    ["absenteeism_time_in_hours", "absenteeism_risk"],
    axis=1
)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

df["employee_cluster"] = clusters

print(df["employee_cluster"].value_counts())

cluster_summary = df.groupby("employee_cluster").mean(numeric_only=True)
print("\nCluster Summary (Average Values):")
print(cluster_summary)


