import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Streamlit app title
st.title("Data Analysis and ML App")

# File uploader
uploaded_file = st.file_uploader("/Users/joshithasaiuppalapati/Downloads/joshitha_ckethine_phase_2", type="csv")
if uploaded_file is not None:
    data_info = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data_info.head())

    # Data cleaning options
    st.header("Data Cleaning")
    if st.button("Check Missing Values"):
        missing_values = data_info.isnull().sum()
        st.write("Missing Values:")
        st.write(missing_values)
    if st.button("Drop Missing Values"):
        data_info = data_info.dropna()
        st.write("Missing values dropped. Remaining data:")
        st.dataframe(data_info.head())

    # Exploratory Data Analysis
    st.header("EDA")
    if st.button("Show Descriptive Statistics"):
        st.write(data_info.describe())

    if st.button("Correlation Heatmap"):
        numeric_cols = data_info.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = data_info[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.button("Show Histograms"):
        numeric_cols = data_info.select_dtypes(include=['float64', 'int64']).columns
        fig, ax = plt.subplots(figsize=(16, 12))
        data_info[numeric_cols].hist(bins=20, ax=ax)
        st.pyplot(fig)

    # Machine Learning Models
    st.header("Machine Learning")

    # Linear Regression
    if st.checkbox("Linear Regression"):
        X = data_info[['sales_per_customer', 'days_for_shipping_(real)']]
        Y = data_info['benefit_per_order']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        st.write("Linear Regression Model Trained")
        st.write("R² Score:", model.score(X_test, Y_test))

        fig, ax = plt.subplots()
        ax.scatter(X_test['sales_per_customer'], Y_test, color='yellow', label='Actual')
        ax.plot(X_test['sales_per_customer'], Y_pred, color='blue', label='Predicted')
        ax.set_title("Sales per Customer vs Benefit per Order")
        ax.set_xlabel("Sales per Customer")
        ax.set_ylabel("Benefit per Order")
        ax.legend()
        st.pyplot(fig)

    # Clustering (KMeans)
    if st.checkbox("K-Means Clustering"):
        X = data_info[['sales_per_customer', 'order_item_total']]
        model_kmeans = KMeans(n_clusters=3, random_state=42)
        data_info['cluster'] = model_kmeans.fit_predict(X)
        st.write("K-Means Clustering Completed")

        fig, ax = plt.subplots()
        ax.scatter(X['sales_per_customer'], X['order_item_total'], c=data_info['cluster'], cmap='viridis')
        ax.set_title("K-Means Clustering")
        ax.set_xlabel("Sales per Customer")
        ax.set_ylabel("Order Item Total")
        st.pyplot(fig)

    # KNN
    if st.checkbox("K-Nearest Neighbors"):
        X = data_info[['sales_per_customer', 'order_item_total']]
        Y = data_info['late_delivery_risk']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(Y_test, Y_pred))
        st.write("Classification Report:")
        st.write(classification_report(Y_test, Y_pred))

        fig, ax = plt.subplots()
        scatter = ax.scatter(X_test['sales_per_customer'], X_test['order_item_total'], c=Y_pred, cmap='viridis')
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
        ax.set_title("KNN Predictions")
        st.pyplot(fig)
