"""Streamlit dashboard for the SmartED e-commerce customer segmentation project."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "synthetic_ecommerce_transactions.csv"
OUT = ROOT

st.set_page_config(page_title="E-commerce Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ E-commerce Customer Segmentation")
st.caption("RFM Analysis • K-Means Clustering • PCA Visualisation")
st.info("Academic demonstration using a self-generated synthetic dataset. It is not real SmartED or customer data.")

@st.cache_data
def load_data():
    tx = pd.read_csv(DATA, parse_dates=["TransactionDate"])
    rfm = pd.read_csv(OUT / "customer_rfm_segments.csv")
    profile = pd.read_csv(OUT / "cluster_profile.csv")
    metrics = json.loads((OUT / "model_metrics.json").read_text())
    return tx, rfm, profile, metrics

try:
    tx, rfm, profile, metrics = load_data()
except FileNotFoundError:
    st.error("Project files are missing. Keep the CSV and generated output files in the same folder as app.py.")
    st.stop()

with st.sidebar:
    st.header("Explore the data")
    categories = ["All"] + sorted(tx["ProductCategory"].dropna().unique().tolist())
    category = st.selectbox("Product category", categories)
    segment_options = ["All"] + sorted(rfm["Segment"].dropna().unique().tolist())
    selected_segment = st.selectbox("Customer segment", segment_options)
    st.markdown("---")
    st.caption("Model: K-Means (k=4), random_state=42, n_init=20")

filtered_tx = tx if category == "All" else tx[tx["ProductCategory"] == category]
filtered_rfm = rfm if selected_segment == "All" else rfm[rfm["Segment"] == selected_segment]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(filtered_tx):,}")
c2.metric("Customers", f"{len(filtered_rfm):,}")
c3.metric("Customer segments", "4")
c4.metric("Silhouette score", f"{metrics['silhouette_sample_2000']:.3f}")

st.subheader("Customer segment overview")
left, right = st.columns(2)
with left:
    fig = px.bar(profile, x="Segment", y="Customers", color="Segment", title="Customers in each segment", text_auto=True)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of customers")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.scatter(rfm, x="PCA1", y="PCA2", color="Segment", hover_data=["CustomerID", "Recency", "Frequency", "Monetary"], title="PCA view of customer clusters", opacity=.7)
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("What the segments mean")
show_cols = ["Cluster", "Segment", "Customers", "Avg_Recency", "Avg_Frequency", "Avg_Monetary", "SharePct"]
st.dataframe(profile[show_cols].rename(columns={"Avg_Recency":"Avg. recency (days)", "Avg_Frequency":"Avg. orders", "Avg_Monetary":"Avg. spend", "SharePct":"Customer share (%)"}).round(2), use_container_width=True, hide_index=True)

st.subheader("Find a customer")
customer_id = st.text_input("Enter a Customer ID (example: CUST00001)").strip().upper()
if customer_id:
    match = rfm[rfm["CustomerID"].astype(str).str.upper() == customer_id]
    if match.empty:
        st.warning("Customer ID not found. Try an ID such as CUST00001.")
    else:
        row = match.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Segment", row["Segment"])
        b.metric("Recency", f"{int(row['Recency'])} days")
        c.metric("Frequency", int(row["Frequency"]))
        d.metric("Monetary value", f"{row['Monetary']:,.2f}")

st.subheader("Model evaluation")
metric_cols = st.columns(3)
metric_cols[0].metric("Davies–Bouldin", f"{metrics['davies_bouldin']:.3f}")
metric_cols[1].metric("Calinski–Harabasz", f"{metrics['calinski_harabasz']:.1f}")
metric_cols[2].metric("K-Means inertia", f"{metrics['kmeans_inertia']:.2f}")
st.caption("The silhouette score indicates moderate cluster separation. Cluster names are descriptive interpretations, not known labels. PCA is only used to display the clusters; K-Means uses all three scaled RFM variables.")

st.subheader("Sample transactions")
st.dataframe(filtered_tx.head(20), use_container_width=True, hide_index=True)
st.download_button("Download customer segments CSV", data=rfm.to_csv(index=False).encode("utf-8"), file_name="customer_rfm_segments.csv", mime="text/csv")

