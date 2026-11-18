# E-commerce Customer Segmentation using RFM, K-Means and PCA

**Training organisation:** SmartED Innovations  
**Programme:** Data Science with Python  
**Student:** Vanshit Jolly  
**Training period:** 15 June 2026 – 15 August 2026

## Important dataset note
This is an academic demonstration built with a **self-generated synthetic dataset** containing 42,000 transactions and 5,000 customers. It is not real SmartED Innovations data and does not describe actual customers.

## What this project does
1. Reads transaction-level e-commerce data.
2. Builds customer-level RFM features: Recency, Frequency and Monetary value.
3. Applies `log1p` transformation and `StandardScaler` to the three RFM features.
4. Fits K-Means with four clusters (`random_state=42`, `n_init=20`).
5. Uses PCA with two components only to visualise the clusters. K-Means is **not** trained on PCA coordinates.
6. Shows segment profiles, model evaluation metrics, a customer lookup and sample transactions in a Streamlit dashboard.

## Project structure
```text
SmartED_Ecommerce_Customer_Segmentation/
├── app.py                         # Interactive Streamlit dashboard
├── train_model.py                 # Rebuilds RFM features and trains K-Means
├── requirements.txt               # Python dependencies
├── README.md                      # This guide
├── data/
│   └── synthetic_ecommerce_transactions.csv
└── outputs/
    ├── customer_rfm_segments.csv
    ├── cluster_profile.csv
    ├── model_metrics.json
    └── *.png                       # Report visualisations
```

## Run on macOS
Open Terminal, move into this folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`. Open it in your browser.

## Run on Windows
Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt and run `.venv\\Scripts\\activate.bat` instead.

## Expected dataset and model setup
- Transactions: 42,000
- Customers: 5,000
- Product categories: 6
- Clusters: 4
- RFM features: Recency, Frequency, Monetary
- Scaling: `log1p` followed by `StandardScaler`
- K-Means: `n_clusters=4`, `random_state=42`, `n_init=20`

The silhouette score in the supplied results is approximately 0.359, indicating moderate rather than perfect separation. Results may vary slightly across library versions or if the input data is changed.

## How to demonstrate it in a viva
1. Start the app with `streamlit run app.py`.
2. Show the transaction/customer/segment counts.
3. Explain RFM: how recently a customer purchased, how often they purchased, and how much they spent.
4. Show the PCA scatter plot and explain that it is a 2D visualisation of clusters.
5. Select a customer segment in the sidebar and explain its average behaviour.
6. Search for `CUST00001` to demonstrate individual customer lookup.
7. Explain that the cluster names are interpretations of the numeric groups, not pre-existing labels.

## Limitations
- Synthetic data cannot establish how real customers behave.
- Four clusters were selected as a project design choice; they are not proven to be globally optimal.
- The silhouette score suggests moderate cluster separation.
- Monetary values are illustrative dataset values, not real revenue.

## Technology stack
Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly and Matplotlib.
