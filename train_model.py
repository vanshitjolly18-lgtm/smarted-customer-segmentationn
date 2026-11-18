"""Train the RFM customer segmentation model and regenerate project outputs.

Dataset: synthetic academic demonstration data (not real company/customer data).
K-Means is fitted on all three scaled RFM features. PCA is used only for plotting.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "synthetic_ecommerce_transactions.csv"
OUT = ROOT


def build_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
    tx["OrderValue"] = pd.to_numeric(tx["OrderValue"], errors="coerce")
    tx = tx.dropna(subset=["CustomerID", "TransactionDate", "OrderValue"])
    snapshot_date = tx["TransactionDate"].max() + pd.Timedelta(days=1)
    rfm = tx.groupby("CustomerID").agg(
        LastPurchase=("TransactionDate", "max"),
        Frequency=("TransactionID", "nunique"),
        Monetary=("OrderValue", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    return rfm[["Recency", "Frequency", "Monetary"]].reset_index()


def main():
    tx = pd.read_csv(DATA)
    rfm = build_rfm(tx)
    features = ["Recency", "Frequency", "Monetary"]
    transformed = np.log1p(rfm[features].clip(lower=0))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)

    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    rfm["Cluster"] = model.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    rfm["PCA1"], rfm["PCA2"] = coords[:, 0], coords[:, 1]

    # Descriptive names are assigned from the resulting cluster profiles.
    profile = rfm.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
    ).reset_index()
    profile["SharePct"] = profile["Customers"] / len(rfm) * 100

    # Rule-based labels are descriptive interpretations, not ground-truth classes.
    names = {}
    for row in profile.itertuples():
        if row.Avg_Monetary >= 20000 and row.Avg_Recency < 100:
            names[row.Cluster] = "High-value loyalists"
        elif row.Avg_Recency >= 240 and row.Avg_Frequency < 5:
            names[row.Cluster] = "At-risk / inactive customers"
        elif row.Avg_Recency < 60:
            names[row.Cluster] = "Recent / developing customers"
        else:
            names[row.Cluster] = "Developing customers"
    rfm["Segment"] = rfm["Cluster"].map(names)
    profile["Segment"] = profile["Cluster"].map(names)

    rfm.to_csv(OUT / "customer_rfm_segments.csv", index=False)
    profile.to_csv(OUT / "cluster_profile.csv", index=False)
    metrics = {
        "dataset_type": "synthetic academic demonstration",
        "random_seed": 42,
        "transactions": int(len(tx)),
        "unique_customers": int(rfm["CustomerID"].nunique()),
        "clusters": 4,
        "kmeans_inertia": float(model.inertia_),
        "silhouette_sample_2000": float(silhouette_score(scaled, rfm["Cluster"], sample_size=min(2000, len(rfm)), random_state=42)),
        "davies_bouldin": float(davies_bouldin_score(scaled, rfm["Cluster"])),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, rfm["Cluster"])),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "note": "PCA is for visualisation only; K-Means is trained on all three scaled RFM features."
    }
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Model training complete.")
    print(f"Transactions: {len(tx):,} | Customers: {rfm.CustomerID.nunique():,}")
    print(profile.round(2).to_string(index=False))
    print(f"Silhouette score: {metrics['silhouette_sample_2000']:.4f}")
    print(f"Outputs saved in: {OUT}")

if __name__ == "__main__":
    main()
