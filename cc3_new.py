import streamlit as st
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar

logo_path = "Images/PUlogo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.caption("Student Name: **Khine Nwe Lin**")
st.sidebar.caption("Student ID: **PIUS20230089**")
st.sidebar.markdown("---")


# Load and clean data

def load_data():
    df = pd.read_csv("CC GENERAL.csv")
    df_clean = df.drop(columns=['CUST_ID'], errors='ignore')
    df_clean.dropna(subset=['CREDIT_LIMIT'], inplace=True)
    df_clean['MINIMUM_PAYMENTS'] = df_clean['MINIMUM_PAYMENTS'].fillna(df_clean['MINIMUM_PAYMENTS'].mean())
    return df_clean


# Load pipeline model

def load_model():
    with open('kmodel.pkl', 'rb') as f:
        return pickle.load(f)


df = load_data()
model = load_model()


# Dataset Information
st.title("Dataset Information")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Customers", len(df))
col_m2.metric("Number of Features", len(df.columns))
col_m3.metric("Clusters (k)", 3)


st.markdown("---")
st.title("Customer Cluster Scatter Plot")

plot_path = "Images/cluster_plot.png"  # your saved plot image
if os.path.exists(plot_path):
    st.image(plot_path, use_container_width=True)
else:
    st.warning(f"Plot image not found")

st.markdown("---")

st.title("Cluster Interpretation")
st.markdown("""
- **Cluster 0:** Low activity / conservative users — have balance but spend less.
- **Cluster 1:** Active / high spenders — moderate balance but high purchases.
- **Cluster 2:** New or occasional users — low balance and low credit limit.
""")

# Predict Customer Segment

st.markdown("---")
st.title("Predict Customer Segment")

with st.form("predict_form"):
    st.write("Enter Customer Details")
    cols = st.columns(3)
    input_data = {}

    for i, column in enumerate(df.columns):
        with cols[i % 3]:
            if "FREQUENCY" in column or "PRC" in column:
                input_data[column] = st.slider(column, 0.0, 1.0, 0.5)
            else:
                input_data[column] = st.number_input(column, value=float(df[column].median()))
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        input_df = pd.DataFrame([input_data])

        
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        input_df = input_df.clip(lower=lower, upper=upper, axis=1)

    
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        if prediction == 0:
            st.info("Result: **Cluster 0** - Low activity / Conservative users.")
        elif prediction == 1:
            st.success("Result: **Cluster 1** - High Spenders / Active users.")
        else:
            st.warning("Result: **Cluster 2** - New or Occasional users.")