import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Title and File Upload
st.title("Generalized Data Analysis and Visualization App")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Data Cleaning
    st.header("Data Cleaning")
    if st.checkbox("Display Missing Values"):
        missing_values = data.isnull().sum()
        st.write(missing_values)

    if st.checkbox("Drop Columns"):
        cols_to_drop = st.multiselect("Select columns to drop", data.columns)
        if cols_to_drop:
            data.drop(columns=cols_to_drop, inplace=True)
            st.write("Updated Data Preview:")
            st.dataframe(data.head())

    # Descriptive Stats
    st.header("Descriptive Statistics")
    if st.checkbox("Show Descriptive Statistics"):
        st.write(data.describe())

    # Visualizations
    st.header("Visualizations")
    if st.checkbox("Correlation Heatmap"):
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns
        if not numeric_cols.empty:
            corr = data[numeric_cols].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            st.pyplot(plt)
        else:
            st.write("No numeric columns available for correlation heatmap.")

    if st.checkbox("Histograms for Numeric Columns"):
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns
        if not numeric_cols.empty:
            data[numeric_cols].hist(bins=20, figsize=(10, 6))
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.write("No numeric columns available for histograms.")

    # Machine Learning
    st.header("Machine Learning")
    if st.checkbox("Linear Regression"):
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns
        if len(numeric_cols) >= 2:
            features = st.multiselect("Select features for X", numeric_cols)
            target = st.selectbox("Select target for Y", numeric_cols)
            if features and target:
                X = data[features].dropna()
                y = data[target].dropna()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                plt.scatter(X_test[features[0]], y_test, color='yellow', label='Actual')
                plt.scatter(X_test[features[0]], predictions, color='blue', label='Predicted')
                plt.title(f'Linear Regression: {features[0]} vs {target}')
                plt.legend()
                st.pyplot(plt)
                st.write("Model Performance:")
                st.write(f"R² Score: {model.score(X_test, y_test)}")
        else:
            st.write("Insufficient numeric columns for regression.")

    if st.checkbox("K-Means Clustering"):
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns
        if len(numeric_cols) >= 2:
            features = st.multiselect("Select features for clustering", numeric_cols)
            if features:
                k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=k, random_state=42)
                data['Cluster'] = kmeans.fit_predict(data[features].dropna())

                plt.scatter(data[features[0]], data[features[1]], c=data['Cluster'], cmap='viridis')
                plt.title("K-Means Clustering")
                st.pyplot(plt)
                st.write(f"Inertia (WCSS): {kmeans.inertia_}")
        else:
            st.write("Insufficient numeric columns for clustering.")

    if st.checkbox("Decision Tree Classifier"):
        categorical_cols = data.select_dtypes(include=['object']).columns
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns
        if numeric_cols.any() and categorical_cols.any():
            features = st.multiselect("Select numeric features for classification", numeric_cols)
            target = st.selectbox("Select target (categorical) for classification", categorical_cols)
            if features and target:
                X = data[features].dropna()
                y = data[target].dropna()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                dt = DecisionTreeClassifier(max_depth=3, random_state=42)
                dt.fit(X_train, y_train)

                plt.figure(figsize=(12, 8))
                plot_tree(dt, feature_names=features, class_names=dt.classes_, filled=True)
                st.pyplot(plt)
                st.write(f"Accuracy: {dt.score(X_test, y_test)}")
        else:
            st.write("Insufficient columns for classification.")

else:
    st.info("Please upload a CSV file to proceed.")
