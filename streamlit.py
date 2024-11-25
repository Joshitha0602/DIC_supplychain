# Importing Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Function to clean data
def clean_data(data):
    # Dropping duplicates
    data = data.drop_duplicates()
    # Handling missing values by filling with mean (numeric) or mode (categorical)
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ["int64", "float64"]:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
    return data

# Function for visualizations
def visualize_data(data, viz_type):
    if viz_type == "Correlation Heatmap":
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    elif viz_type == "Histograms":
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        numeric_data.hist(bins=20, figsize=(15, 10))
        st.pyplot(plt)

    elif viz_type == "Scatter Plot":
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        col_x = st.sidebar.selectbox("Select X-axis", numeric_cols)
        col_y = st.sidebar.selectbox("Select Y-axis", numeric_cols)
        plt.scatter(data[col_x], data[col_y])
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        st.pyplot(plt)

# Function for modeling
def linear_regression_model(data, target_col, feature_cols):
    X = data[feature_cols]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)

    # Display Results
    st.write(f"Model Coefficients: {model.coef_}")
    st.write(f"Intercept: {model.intercept_}")
    st.write(f"R² Score: {r2_score}")

    # Scatter Plot
    predictions = model.predict(X_test)
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    st.pyplot(plt)

# Streamlit App Layout
st.title("Generalized Data Analysis and Modeling")
uploaded_file = st.file_uploader("Upload your CSV File", type="csv")

if uploaded_file is not None:
    # Load Dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.write(df.head())

    # Data Cleaning
    st.write("### Cleaning Data")
    df_cleaned = clean_data(df)
    st.write("Data cleaned successfully!")

    # Data Visualization
    st.sidebar.title("Visualization Options")
    viz_type = st.sidebar.selectbox("Choose a Visualization", ["None", "Correlation Heatmap", "Histograms", "Scatter Plot"])
    if viz_type != "None":
        st.write(f"### {viz_type}")
        visualize_data(df_cleaned, viz_type)

    # Modeling
    st.sidebar.title("Modeling Options")
    model_type = st.sidebar.selectbox("Choose a Model", ["None", "Linear Regression"])
    if model_type == "Linear Regression":
        st.write("### Linear Regression Model")
        target_col = st.selectbox("Select Target Variable", df_cleaned.select_dtypes(include=["float64", "int64"]).columns)
        feature_cols = st.multiselect("Select Feature Columns", df_cleaned.select_dtypes(include=["float64", "int64"]).columns)

        if st.button("Run Model"):
            if target_col and feature_cols:
                linear_regression_model(df_cleaned, target_col, feature_cols)
            else:
                st.warning("Please select a target variable and at least one feature.")
else:
    st.write("Please upload a CSV file to proceed.")
