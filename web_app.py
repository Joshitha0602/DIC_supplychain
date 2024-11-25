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

# Sidebar for uploading the file
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Raw Data")
    st.write(data.head())

    # Step 1: Data Cleaning
    st.write("## Data Cleaning")
    
    # Dropping unnecessary columns
    drop_cols = st.sidebar.multiselect(
        "Select columns to drop", data.columns, default=[]
    )
    data.drop(columns=drop_cols, inplace=True)

    # Handling missing values
    if st.sidebar.checkbox("Drop rows with missing values"):
        data.dropna(inplace=True)

    # Normalizing column names
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    st.write("Cleaned Data", data.head())

    # Step 2: Exploratory Data Analysis
    st.write("## Exploratory Data Analysis")

    if st.sidebar.checkbox("Display descriptive statistics"):
        st.write(data.describe())

    if st.sidebar.checkbox("Show correlation heatmap"):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)

    if st.sidebar.checkbox("Show histograms"):
        data.hist(bins=20, figsize=(15, 10))
        st.pyplot(plt)

    # Step 3: Machine Learning Models
    st.write("## Machine Learning Models")

    model_option = st.sidebar.selectbox(
        "Choose a model",
        [
            "Linear Regression",
            "KMeans Clustering",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine (SVM)",
        ],
    )

    if model_option == "Linear Regression":
        st.write("### Linear Regression")
        features = st.sidebar.multiselect("Select features", data.columns)
        target = st.sidebar.selectbox("Select target", data.columns)
        if features and target:
            X = data[features]
            Y = data[target]
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )
            model = LinearRegression()
            model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            st.write(f"R² Score: {model.score(X_test, Y_test)}")

    if model_option == "KMeans Clustering":
        st.write("### KMeans Clustering")
        cluster_features = st.sidebar.multiselect("Select clustering features", data.columns)
        if cluster_features:
            X = data[cluster_features]
            num_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
            model = KMeans(n_clusters=num_clusters)
            data["Cluster"] = model.fit_predict(X)
            st.write("Clustered Data", data.head())

    if model_option == "Random Forest":
        st.write("### Random Forest Classifier")
        features = st.sidebar.multiselect("Select features", data.columns)
        target = st.sidebar.selectbox("Select target", data.columns)
        if features and target:
            X = data[features]
            Y = data[target]
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )
            model = RandomForestClassifier()
            model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            st.write("Classification Report")
            st.text(classification_report(Y_test, predictions))
            st.write(f"Accuracy: {model.score(X_test, Y_test):.2f}")

    if model_option == "Support Vector Machine (SVM)":
        st.write("### Support Vector Machine")
        features = st.sidebar.multiselect("Select features", data.columns)
        target = st.sidebar.selectbox("Select target", data.columns)
        if features and target:
            X = data[features]
            Y = data[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_scaled, Y, test_size=0.2, random_state=42
            )
            model = LinearSVC()
            model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            st.write(f"Accuracy: {model.score(X_test, Y_test):.2f}")

    # General visualization
    st.write("### General Visualization")
    x_col = st.sidebar.selectbox("Select X-axis column", data.columns)
    y_col = st.sidebar.selectbox("Select Y-axis column", data.columns)
    if x_col and y_col:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_col, y=y_col)
        st.pyplot(plt)
