import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# App Title
# ---------------------------
st.title("📊 Customer Churn Prediction App")
st.write("Logistic Regression model to predict customer churn")

# ---------------------------
# Load Data
# ---------------------------
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = pd.read_csv(file_path)
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return df


df = load_data()

# ---------------------------
# Churn Counts
# ---------------------------
stay_count = (df['Churn'] == 'No').sum()
leave_count = (df['Churn'] == 'Yes').sum()

st.subheader("👥 Customer Distribution")
col1, col2 = st.columns(2)
col1.metric("Customers Staying", stay_count)
col2.metric("Customers Leaving", leave_count)

# ---------------------------
# Prepare Data
# ---------------------------
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# Predictions
# ---------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# ---------------------------
# Accuracy
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Accuracy")
st.metric("Accuracy", f"{accuracy * 100:.2f}%")

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

st.subheader("🧮 Confusion Matrix Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("True Positive (Churn Correct)", tp)
c2.metric("True Negative (Stay Correct)", tn)
c3.metric("False Positive (Stay → Churn)", fp)
c4.metric("False Negative (Missed Churn)", fn)

# ---------------------------
# Business Insight
# ---------------------------
st.subheader("📌 Business Insight")
st.write(
    """
- **True Positives** show customers correctly identified as leaving  
- **False Negatives** are dangerous because churn customers are missed  
- The model helps businesses take **early retention actions**
    """
)

