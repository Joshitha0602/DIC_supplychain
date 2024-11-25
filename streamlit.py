import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

# 1. Set up Streamlit App
st.title("Supply Chain Data Analysis")
st.sidebar.header("Options")

# 2. File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # 3. Data Cleaning Summary
    st.header("Data Cleaning")
    st.write(f"Initial data shape: {data.shape}")
    data.dropna(inplace=True)
    st.write(f"Data shape after dropping missing values: {data.shape}")

    # 4. Exploratory Data Analysis
    st.header("Exploratory Data Analysis (EDA)")

    # Select feature for visualization
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    feature = st.selectbox("Select feature for visualization", options=numeric_columns)

    if feature:
        fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, ax=ax)
        st.pyplot(fig)

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # 5. Machine Learning
    st.header("Machine Learning")

    # User chooses an ML model
    model_choice = st.selectbox("Choose a Model", ["Linear Regression", "K-Means Clustering"])

    if model_choice == "Linear Regression":
        # Select features and target
        target = st.selectbox("Select Target Variable", options=numeric_columns)
        features = st.multiselect("Select Feature Variables", options=numeric_columns)

        if target and features:
            X = data[features]
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Model Coefficients:", model.coef_)
            st.write("Intercept:", model.intercept_)
            st.write("R² Score:", model.score(X_test, y_test))

            # Visualization
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Linear Regression Results")
            st.pyplot(fig)

    elif model_choice == "K-Means Clustering":
        # Select clustering features
        features = st.multiselect("Select Features for Clustering", options=numeric_columns)
        if features:
            X = data[features]
            kmeans = KMeans(n_clusters=3, random_state=42)
            data["Cluster"] = kmeans.fit_predict(X)

            # Show Cluster Visualization
            st.write("Cluster Centers:", kmeans.cluster_centers_)
            fig, ax = plt.subplots()
            scatter = ax.scatter(data[features[0]], data[features[1]], c=data["Cluster"], cmap="viridis")
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)

# Streamlit app instructions
else:
    st.write("Upload a dataset to begin.")
