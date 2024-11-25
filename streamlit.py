import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# Function to load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

# Function to clean and preprocess data
def preprocess_data(data):
    # Step 1: Drop columns with 100% missing values
    data = data.dropna(axis=1, how='all')
    
    # Step 2: Drop duplicate rows
    data = data.drop_duplicates()
    
    # Step 3: Standardize column names
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    # Step 4: Drop rows with missing values
    data = data.dropna()
    
    return data

# Function to plot correlation heatmap
def plot_correlation_heatmap(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Function to perform linear regression
def linear_regression_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions and performance
    y_pred = model.predict(X_test)
    st.write("Linear Regression Results:")
    st.write("Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)
    st.write("R² Score:", model.score(X_test, y_test))
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
    plt.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
    plt.title('Linear Regression Predictions')
    plt.xlabel(features[0])
    plt.ylabel(target)
    plt.legend()
    st.pyplot(plt)

# Function for KMeans clustering
def kmeans_clustering(data, features, n_clusters):
    X = data[features]
    model = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = model.fit_predict(X)
    
    # Plot clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data[features[0]], data[features[1]], c=data['Cluster'], cmap='viridis', marker='o')
    plt.title('K-Means Clustering')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    st.pyplot(plt)

# Streamlit UI Layout
st.title("Generalized Data Analysis and Machine Learning Workflow")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    # Load and display data
    data = load_data(uploaded_file)
    st.write("Original Data Preview:")
    st.write(data.head())
    
    # Preprocess data
    data = preprocess_data(data)
    st.write("Cleaned Data Preview:")
    st.write(data.head())
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plot_correlation_heatmap(data)
    
    # Linear Regression
    st.subheader("Linear Regression")
    features = st.multiselect("Select features for Linear Regression:", data.columns)
    target = st.selectbox("Select target variable for Linear Regression:", data.columns)
    if st.button("Run Linear Regression"):
        if len(features) > 0:
            linear_regression_model(data, features, target)
        else:
            st.error("Please select at least one feature.")
    
    # K-Means Clustering
    st.subheader("K-Means Clustering")
    clustering_features = st.multiselect("Select features for Clustering:", data.columns)
    n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
    if st.button("Run K-Means Clustering"):
        if len(clustering_features) == 2:
            kmeans_clustering(data, clustering_features, n_clusters)
        else:
            st.error("Please select exactly two features for clustering.")
