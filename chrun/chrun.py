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
# CUSTOM DARK THEME CSS
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #0b0f19; }
.block-container { padding-top: 1.5rem; }
.card {
    background-color: #111827;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
}
.card-title {
    color: #9ca3af;
    font-size: 15px;
}
.card-value {
    font-size: 26px;
    font-weight: bold;
    color: #22c55e;
}
.card-value-red {
    font-size: 26px;
    font-weight: bold;
    color: #ef4444;
}
.section {
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
    path = os.path.join(base_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = pd.read_csv(path)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

df = load_data()

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("## 📊 Telco Customer Churn Dashboard")
st.write("Logistic Regression model to predict customer churn")

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
total = len(df)
stay = (df["Churn"] == "No").sum()
leave = (df["Churn"] == "Yes").sum()
churn_rate = round((leave / total) * 100, 2)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Total Customers</div>
        <div class="card-value">{total}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Customers Staying</div>
        <div class="card-value">{stay}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Customers Leaving</div>
        <div class="card-value-red">{leave}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Churn Rate</div>
        <div class="card-value-red">{churn_rate}%</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# SAMPLE DATA TABLE
# --------------------------------------------------
st.markdown("### 📋 Sample Customer Data")
st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# CUSTOMER DISTRIBUTION
# --------------------------------------------------
st.markdown("### 👥 Customer Distribution")

fig1, ax1 = plt.subplots(figsize=(4,3))
sns.countplot(x=df["Churn"], palette=["#22c55e", "#ef4444"], ax=ax1)
ax1.set_xlabel("Customer Status")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.markdown("### 📈 Model Performance")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# --------------------------------------------------
# CONFUSION MATRIX & ROC CURVE
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

colL, colR = st.columns(2)

with colL:
    st.markdown("#### 🧮 Confusion Matrix")
    fig2, ax2 = plt.subplots(figsize=(4,4))
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

with colR:
    st.markdown("#### 📉 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig3, ax3 = plt.subplots(figsize=(4,4))
    ax3.plot(fpr, tpr, color="#22c55e", label=f"AUC = {roc_auc:.2f}")
    ax3.plot([0,1], [0,1], linestyle="--", color="gray")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend()
    st.pyplot(fig3)

# --------------------------------------------------
# FINAL INSIGHTS
# --------------------------------------------------
st.markdown("### 📌 Model Insights")
st.write(f"""
- **True Positives (Correct churn predictions):** {tp}  
- **True Negatives (Correct stay predictions):** {tn}  
- **False Positives (Unnecessary retention actions):** {fp}  
- **False Negatives (Missed churn customers):** {fn}  

✔ Business should prioritize **reducing false negatives** to avoid customer loss.
""")
