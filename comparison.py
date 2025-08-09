import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

st.title("Model Comparison Dashboard")

# Load models
rf_model = joblib.load("rf_model.pkl")
# xgb_model = joblib.load("xgb_model.pkl")
gb_model = joblib.load("gb_model.pkl")
# lgbm_model = joblib.load("lgbm_model.pkl")
log_model = joblib.load("lr_model.pkl")
le = joblib.load("label_encoder.pkl")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# Model selection
model_dict = {
    "Random Forest": rf_model,
    # "XGBoost": xgb_model,
    "Gradient Boosting": gb_model,
    # "LightGBM": lgbm_model,
    "Logistic Regression": log_model
}

model_choice = st.selectbox("Select a Model", list(model_dict.keys()))
model = model_dict[model_choice]

# Predict
y_pred = model.predict(X_test)

# Decode if needed
y_test_decoded = le.inverse_transform(y_test) if hasattr(le, 'inverse_transform') else y_test
y_pred_decoded = le.inverse_transform(y_pred) if hasattr(le, 'inverse_transform') else y_pred

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Sample predictions
st.subheader("Sample Predictions")
comparison_df = pd.DataFrame({
    "Actual": y_test_decoded,
    "Predicted": y_pred_decoded
})
st.dataframe(comparison_df.sample(20))