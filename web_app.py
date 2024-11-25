import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

st.title("Data Analysis and Machine Learning Application")

# Page configuration
st.set_page_config(page_title="Data Cleaning Demo", layout="wide")

# Function to load data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Sidebar for file upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file).copy()
    st.title("Uploaded Dataset")
    st.write(data.head())
# Data Cleaning
    st.subheader("Data Cleaning")
    missing_values = data.isnull().sum()
    st.write("Missing Values:")
    st.write(missing_values[missing_values > 0])
    
    # Dropping unnecessary columns (user-defined)
    drop_columns = st.sidebar.multiselect("Select columns to drop", data.columns)
    if drop_columns:
        data.drop(columns=drop_columns, inplace=True)
        st.write(f"Columns dropped: {drop_columns}")

    # Handle duplicates
    if st.sidebar.checkbox("Remove Duplicates"):
        data.drop_duplicates(inplace=True)
        st.write("Duplicates removed")

    # Convert date columns
    date_columns = st.sidebar.multiselect("Select date columns to convert", data.columns)
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        st.write(f"Converted {col} to datetime")

    # Normalizing text columns
    text_columns = st.sidebar.multiselect("Select text columns to normalize", data.columns)
    for col in text_columns:
        data[col] = data[col].str.lower().str.strip()
        st.write(f"Normalized text in {col}")

    # Data Overview Post Cleaning
    st.subheader("Data Overview Post Cleaning")
    st.write(data.describe())

    # Visualizations
    st.header("Visualizations")

    # Heatmap
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        corr = data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Histograms
    if st.sidebar.checkbox("Show Histograms"):
        st.subheader("Histograms")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure()
            sns.histplot(data[col], kde=True)
            plt.title(f"Distribution of {col}")
            st.pyplot(plt)

    # Modeling (Linear Regression Example)
    st.header("Modeling")
    target = st.sidebar.selectbox("Select target variable", data.columns)
    features = st.sidebar.multiselect("Select feature variables", data.columns)

    if target and features:
        st.subheader("Linear Regression Model")
        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        st.write("Model Coefficients:")
        st.write(model.coef_)
        st.write("R² Score:")
        st.write(model.score(X_test, y_test))

        # Visualization
        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        st.pyplot(plt)
