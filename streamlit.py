import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Step 1: Load Dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Clean and Preprocess Dataset
def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()  # Drop missing values
    df = df.drop_duplicates()  # Remove duplicate rows
    df.columns = df.columns.str.lower().str.replace(' ', '_')  # Standardize column names
    return df

# Step 3: Visualization
def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    st.pyplot(plt)

def plot_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# Step 4: Modeling
def train_linear_regression(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write("Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)
    st.write("R² Score:", model.score(X_test, y_test))
    return model

# Step 5: Clustering
def perform_clustering(df, features, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = model.fit_predict(df[features])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features[0], y=features[1], hue='cluster', data=df, palette='viridis')
    st.pyplot(plt)

# Streamlit App Layout
st.title("Generalized Data Analysis and ML Workflow")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Preprocessing
    data = preprocess_data(data)
    st.write("Cleaned Dataset Preview:", data.head())
    
    # Visualizations
    st.sidebar.subheader("Visualizations")
    if st.sidebar.button("Show Heatmap"):
        plot_heatmap(data)
    
    column_to_plot = st.sidebar.selectbox("Select Column for Histogram", data.select_dtypes(include=[np.number]).columns)
    if st.sidebar.button("Plot Histogram"):
        plot_histogram(data, column_to_plot)
    
    # Linear Regression
    st.sidebar.subheader("Linear Regression")
    features = st.sidebar.multiselect("Select Features", data.columns)
    target = st.sidebar.selectbox("Select Target", data.columns)
    if st.sidebar.button("Train Linear Regression"):
        train_linear_regression(data, features, target)
    
    # Clustering
    st.sidebar.subheader("Clustering")
    clustering_features = st.sidebar.multiselect("Select Features for Clustering", data.columns)
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    if st.sidebar.button("Perform Clustering"):
        perform_clustering(data, clustering_features, n_clusters)
