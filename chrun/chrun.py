import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

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
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #0b0f19; }
.block-container { padding-top: 1rem; }
.card {
    background-color: #111827;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
}
.card-title {
    color: #9ca3af;
    font-size: 12px;
}
.card-value {
    font-size: 18px;
    font-weight: bold;
    color: #22c55e;
}
.card-value-red {
    font-size: 18px;
    font-weight: bold;
    color: #ef4444;
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
st.write("Compact dashboard with churn prediction results")

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
total = len(df)
stay = (df["Churn"] == "No").sum()
leave = (df["Churn"] == "Yes").sum()
churn_rate = round((leave / total) * 100, 2)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"<div class='card'><div class='card-title'>Total</div><div class='card-value'>{total}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='card'><div class='card-title'>Stay</div><div class='card-value'>{stay}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='card'><div class='card-title'>Leave</div><div class='card-value-red'>{leave}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='card'><div class='card-title'>Churn %</div><div class='card-value-red'>{churn_rate}%</div></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SAMPLE DATA (VERY SMALL)
# --------------------------------------------------
st.markdown("### 📋 Sample Data")
st.dataframe(df.head(5), height=160)

# --------------------------------------------------
# CUSTOMER DISTRIBUTION (VERY SMALL)
# --------------------------------------------------
st.markdown("### 👥 Distribution")
fig1, ax1 = plt.subplots(figsize=(2.2,1.6))
sns.countplot(x=df["Churn"], palette=["#22c55e", "#ef4444"], ax=ax1)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.tick_params(axis='both', labelsize=7)
st.pyplot(fig1, use_container_width=False)

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
st.markdown("### 📈 Accuracy")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# --------------------------------------------------
# CONFUSION MATRIX (VERY SMALL)
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

st.markdown("### 🧮 Confusion Matrix")
fig2, ax2 = plt.subplots(figsize=(2.2,2.2))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    annot_kws={"size": 7},
    xticklabels=["Stay", "Leave"],
    yticklabels=["Stay", "Leave"],
    ax=ax2
)
ax2.tick_params(labelsize=7)
st.pyplot(fig2, use_container_width=False)

# --------------------------------------------------
# ROC CURVE (VERY SMALL)
# --------------------------------------------------
st.markdown("### 📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig3, ax3 = plt.subplots(figsize=(2.2,2.2))
ax3.plot(fpr, tpr, color="#22c55e", linewidth=1.5, label=f"AUC={roc_auc:.2f}")
ax3.plot([0,1], [0,1], linestyle="--", color="gray", linewidth=1)
ax3.tick_params(labelsize=7)
ax3.legend(fontsize=7)
st.pyplot(fig3, use_container_width=False)

# --------------------------------------------------
# FINAL INSIGHTS
# --------------------------------------------------
st.markdown("### 📌 Insights")
st.write(f"""
TP: **{tp}** | TN: **{tn}**  
FP: **{fp}** | FN: **{fn}**

✔ Reducing **FN** is critical to avoid customer loss.
""")
