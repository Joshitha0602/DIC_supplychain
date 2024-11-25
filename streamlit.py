import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Streamlit App
st.title("Generalized Data Analysis App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Cleaning Step 1: Handling Missing Values
    st.subheader("Cleaning: Handling Missing Values")
    st.write("Missing values in each column:")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    drop_cols = st.multiselect("Select columns to drop (if any)", df.columns)
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.dropna()  # Dropping rows with missing values
    st.write("Cleaned Dataset Preview:", df.head())

    # Cleaning Step 2: Removing Duplicates
    st.subheader("Cleaning: Removing Duplicates")
    st.write(f"Duplicate rows: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    st.write(f"Remaining rows after removing duplicates: {len(df)}")

    # Cleaning Step 3: Standardizing Column Names
    st.subheader("Cleaning: Standardizing Column Names")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    st.write("Standardized Column Names:", df.columns.tolist())

    # EDA Step 1: Descriptive Statistics
    st.subheader("EDA: Descriptive Statistics")
    st.write(df.describe())

    # EDA Step 2: Correlation Heatmap
    st.subheader("EDA: Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())

    # EDA Step 3: Histogram of Numerical Features
    st.subheader("EDA: Histograms of Numerical Features")
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=20, color='blue', alpha=0.7)
        plt.title(f"Histogram for {col}")
        st.pyplot(plt.gcf())

    # Model Training: Linear Regression
    st.subheader("Modeling: Linear Regression")
    features = st.multiselect("Select Features for Linear Regression", numeric_cols)
    target = st.selectbox("Select Target for Linear Regression", numeric_cols)
    if features and target:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)

        st.write("Model Coefficients:", lr_model.coef_)
        st.write("Model Intercept:", lr_model.intercept_)

        plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
        plt.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
        plt.title("Linear Regression: Actual vs Predicted")
        plt.legend()
        st.pyplot(plt.gcf())

    # Clustering: KMeans
    st.subheader("Clustering: K-Means")
    cluster_features = st.multiselect("Select Features for K-Means Clustering", numeric_cols)
    if cluster_features:
        X_cluster = df[cluster_features]
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_cluster)

        st.write("Cluster Labels:", df['cluster'].value_counts())

        plt.scatter(X_cluster.iloc[:, 0], X_cluster.iloc[:, 1], c=df['cluster'], cmap='viridis')
        plt.title("K-Means Clustering")
        plt.xlabel(cluster_features[0])
        plt.ylabel(cluster_features[1])
        st.pyplot(plt.gcf())

    # Classification: K-Nearest Neighbors (KNN)
    st.subheader("Classification: K-Nearest Neighbors")
    knn_features = st.multiselect("Select Features for KNN", numeric_cols)
    knn_target = st.selectbox("Select Target for KNN", numeric_cols)
    if knn_features and knn_target:
        X_knn = df[knn_features]
        y_knn = df[knn_target]
        X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', marker='o')
        plt.title("KNN Predictions")
        st.pyplot(plt.gcf())
