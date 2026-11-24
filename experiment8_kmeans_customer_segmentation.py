import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate mall customer data around five sample segments
np.random.seed(5)
cluster_centres = [(25, 80), (55, 20), (85, 85), (55, 50), (25, 20)]
rows = []
for income, score in cluster_centres:
    for _ in range(40):
        rows.append([np.random.normal(income, 7), np.random.normal(score, 7)])
customers = pd.DataFrame(np.round(rows, 1),
                         columns=["annual_income_k", "spending_score"])
customers.insert(0, "customer_id", range(1, len(customers)+1))
print("MALL CUSTOMER SEGMENTATION")
print(customers.head().to_string(index=False))
print("Total customers:", len(customers))

features = ["annual_income_k", "spending_score"]
X = StandardScaler().fit_transform(customers[features])

# Compare k=2 through k=8 using WCSS and silhouette score
k_values = list(range(2, 9))
wcss, silhouettes = [], []
print("\nk       WCSS       Silhouette")
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    wcss.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(k, round(km.inertia_, 2), round(silhouettes[-1], 4))

best_k = k_values[int(np.argmax(silhouettes))]
print("\nBest k by silhouette score:", best_k)

final_model = KMeans(n_clusters=best_k, n_init=10, random_state=0)
customers["cluster"] = final_model.fit_predict(X)
profile = customers.groupby("cluster")[features].mean().round(1)
profile["size"] = customers["cluster"].value_counts().sort_index()

def describe(row):
    income = "High income" if row.annual_income_k > 70 else (
        "Low income" if row.annual_income_k < 40 else "Mid income")
    spending = "high spender" if row.spending_score > 65 else (
        "low spender" if row.spending_score < 35 else "average spender")
    return f"{income}, {spending}"

profile["segment"] = profile.apply(describe, axis=1)
print("\nCLUSTER PROFILES\n", profile.to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(k_values, wcss, "o-")
ax[0].set(xlabel="Number of clusters k", ylabel="WCSS (inertia)", title="Elbow Method")
ax[1].scatter(customers.annual_income_k, customers.spending_score,
              c=customers.cluster, cmap="tab10", s=20)
centres = customers.groupby("cluster")[features].mean()
ax[1].scatter(centres.annual_income_k, centres.spending_score,
              c="black", marker="X", s=150, label="Centroids")
ax[1].set(xlabel="Annual income (thousand Rs.)", ylabel="Spending score",
          title=f"Customer Segments (k={best_k})")
ax[1].legend()
plt.tight_layout()
plt.savefig("exp08_kmeans.png", dpi=150)
plt.show()
