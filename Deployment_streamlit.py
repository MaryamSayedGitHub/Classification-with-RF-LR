import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load trained models
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Streamlit UI
st.title("Pulsar Classification App")
st.sidebar.header("User Input Features")

def user_input_features():
    Mean_Integrated_Profile = st.sidebar.number_input("Mean Integrated Profile", value=0.0)
    SD_Integrated_Profile = st.sidebar.number_input("SD Integrated Profile", value=0.0)
    Excess_Kurtosis_Integrated_Profile = st.sidebar.number_input("Excess Kurtosis Integrated Profile", value=0.0)
    Skewness_Integrated_Profile = st.sidebar.number_input("Skewness Integrated Profile", value=0.0)
    Mean_DM_SNR_Curve = st.sidebar.number_input("Mean DM SNR Curve", value=0.0)
    SD_DM_SNR_Curve = st.sidebar.number_input("SD DM SNR Curve", value=0.0)
    Excess_Kurtosis_DM_SNR_Curve = st.sidebar.number_input("Excess Kurtosis DM SNR Curve", value=0.0)
    Skewness_DM_SNR_Curve = st.sidebar.number_input("Skewness DM SNR Curve", value=0.0)
    
    data = {
        'Mean_Integrated_Profile': [Mean_Integrated_Profile],
        'SD_Integrated_Profile': [SD_Integrated_Profile],
        'Excess_Kurtosis_Integrated_Profile': [Excess_Kurtosis_Integrated_Profile],
        'Skewness_Integrated_Profile': [Skewness_Integrated_Profile],
        'Mean_DM_SNR_Curve': [Mean_DM_SNR_Curve],
        'SD_DM_SNR_Curve': [SD_DM_SNR_Curve],
        'Excess_Kurtosis_DM_SNR_Curve': [Excess_Kurtosis_DM_SNR_Curve],
        'Skewness_DM_SNR_Curve': [Skewness_DM_SNR_Curve]
    }
    return pd.DataFrame(data)

# Get user input
df = user_input_features()
st.write("### User Input Features")
st.write(df)

# Predictions
if st.button("Predict"):
    rf_pred = rf_model.predict(df)[0]
    lr_pred = lr_model.predict(df)[0]
    
    st.write(f"**Random Forest Prediction:** {'Pulsar' if rf_pred else 'Not Pulsar'}")
    st.write(f"**Logistic Regression Prediction:** {'Pulsar' if lr_pred else 'Not Pulsar'}")

# Load test dataset for ROC and Confusion Matrix
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

rf_y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
lr_y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred_proba)
rf_roc_auc = auc(rf_fpr, rf_tpr)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_pred_proba)
lr_roc_auc = auc(lr_fpr, lr_tpr)

# Plot ROC curves
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines', name=f'RF (AUC = {rf_roc_auc:.2f})'))
fig_roc.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode='lines', name=f'LR (AUC = {lr_roc_auc:.2f})'))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Classifier'))
fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig_roc)

# Confusion Matrices
rf_y_pred = rf_model.predict(X_test)
lr_y_pred = lr_model.predict(X_test)

rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
lr_conf_matrix = confusion_matrix(y_test, lr_y_pred)

fig_rf = go.Figure(data=go.Heatmap(z=rf_conf_matrix, x=['Not Pulsar', 'Pulsar'], y=['Not Pulsar', 'Pulsar'], colorscale='Viridis'))
fig_lr = go.Figure(data=go.Heatmap(z=lr_conf_matrix, x=['Not Pulsar', 'Pulsar'], y=['Not Pulsar', 'Pulsar'], colorscale='Viridis'))

fig_rf.update_layout(title_text='Confusion Matrix - RF', xaxis_title='Predicted', yaxis_title='Actual')
fig_lr.update_layout(title_text='Confusion Matrix - LR', xaxis_title='Predicted', yaxis_title='Actual')

st.plotly_chart(fig_rf)
st.plotly_chart(fig_lr)
