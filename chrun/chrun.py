import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (DASHBOARD LOOK)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}
.metric-title {
    font-size: 16px;
    color: #9ca3af;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #22c55e;
}
.metric-value-red {
    font-size: 28px;
    font-weight: bold;
    color: #ef4444;
}
.section-title {
    color: #e5e7eb;
    font-size: 22px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = pd.read_csv(file_path)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

df = load_data()

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("## 📊 Telco Customer Churn Dashboard")
st.write("Logistic Regression model for predicting customer churn")

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
total_customers = df.shape[0]
stay_count = (df["Churn"] == "No").sum()
leave_count = (df["Churn"] == "Yes").sum()
churn_rate = round((leave_count / total_customers) * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Customers</div>
        <div class="metric-value">{total_customers}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Customers Staying</div>
        <div class="metric-value">{stay_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Customers Leaving</div>
        <div class="metric-value-red">{leave_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Churn Rate</div>
        <div class="metric-value-red">{churn_rate}%</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

# --------------------------------------------------
# CUSTOMER DISTRIBUTION
# --------------------------------------------------
st.markdown("### 👥 Customer Distribution")

fig1, ax1 = plt.subplots(figsize=(5,4))
sns.countplot(x=df["Churn"], palette=["#22c55e", "#ef4444"], ax=ax1)
ax1.set_xlabel("Churn Status")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.markdown("### 📈 Model Performance")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

st.markdown("### 🧮 Confusion Matrix")

fig2, ax2 = plt.subplots(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=["Stay", "Leave"],
    yticklabels=["Stay", "Leave"],
    ax=ax2
)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------
st.markdown("### 📉 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig3, ax3 = plt.subplots(figsize=(5,4))
ax3.plot(fpr, tpr, color="#22c55e", label=f"AUC = {roc_auc:.2f}")
ax3.plot([0,1], [0,1], linestyle="--", color="gray")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend()
st.pyplot(fig3)

# --------------------------------------------------
# FINAL INSIGHTS
# --------------------------------------------------
st.markdown("### 📌 Business Insights")
st.write(f"""
- **True Positives (Correctly caught churn):** {tp}  
- **True Negatives (Correctly identified stay):** {tn}  
- **False Positives (Unnecessary retention offers):** {fp}  
- **False Negatives (Missed churn customers):** {fn}  

➡️ It is **better to tolerate false positives** than miss actual churn customers.
""")
