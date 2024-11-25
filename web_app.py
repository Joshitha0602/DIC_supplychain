import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Title and file upload
st.title("Streamlit Application for Data Analysis and Machine Learning")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the Dataset:")
    st.write(data.head())
    st.write(f"Shape: {data.shape}")

    # Data cleaning options
    st.sidebar.header("Data Cleaning")
    if st.sidebar.checkbox("Check for Missing Values"):
        missing_values = data.isnull().sum()
        st.write("Missing Values in Each Column:")
        st.write(missing_values[missing_values > 0])

    if st.sidebar.checkbox("Drop Columns with High Missing Values"):
        threshold = st.sidebar.slider("Missing Value Threshold (%)", 0, 100, 50)
        cols_to_drop = [col for col in data.columns if data[col].isnull().mean() * 100 > threshold]
        data = data.drop(columns=cols_to_drop)
        st.write("Dropped columns:", cols_to_drop)
        st.write("Updated Dataset Shape:", data.shape)

    # Exploratory Data Analysis
    st.sidebar.header("EDA")
    if st.sidebar.checkbox("Descriptive Statistics"):
        st.write("Descriptive Statistics:")
        st.write(data.describe())

    if st.sidebar.checkbox("Correlation Heatmap"):
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if numeric_cols.any():
            st.write("Correlation Heatmap:")
            plt.figure(figsize=(10, 8))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
            st.pyplot()

    # Machine Learning Options
    st.sidebar.header("Machine Learning")
    ml_task = st.sidebar.selectbox("Select ML Task", ["Linear Regression", "K-Means Clustering", "Random Forest", "LinearSVC"])
    
    if ml_task == "Linear Regression":
        st.subheader("Linear Regression")
        features = st.multiselect("Select Features", options=data.columns)
        target = st.selectbox("Select Target", options=data.columns)
        if features and target:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("R² Score:", model.score(X_test, y_test))
            plt.scatter(X_test.iloc[:, 0], y_test, color='yellow')
            plt.plot(X_test.iloc[:, 0], y_pred, color='blue')
            plt.title("Linear Regression")
            st.pyplot()

    elif ml_task == "K-Means Clustering":
        st.subheader("K-Means Clustering")
        features = st.multiselect("Select Features", options=data.columns)
        clusters = st.slider("Select Number of Clusters", 2, 10, 3)
        if features:
            X = data[features]
            model = KMeans(n_clusters=clusters, random_state=42)
            data['Cluster'] = model.fit_predict(X)
            st.write("Cluster Assignments:")
            st.write(data['Cluster'])
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=data['Cluster'], cmap='viridis')
            plt.title("K-Means Clustering")
            st.pyplot()

    elif ml_task == "Random Forest":
        st.subheader("Random Forest")
        features = st.multiselect("Select Features", options=data.columns)
        target = st.selectbox("Select Target", options=data.columns)
        if features and target:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

    elif ml_task == "LinearSVC":
        st.subheader("Linear SVC")
        features = st.multiselect("Select Features", options=data.columns)
        target = st.selectbox("Select Target", options=data.columns)
        if features and target:
            X = data[features]
            y = data[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model = LinearSVC(dual=False, max_iter=5000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='Reds', edgecolor='k', s=100)
            plt.title("LinearSVC Classification")
            st.pyplot()

# Footer
st.write("App developed for generalized data analysis and machine learning tasks.")
