import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="Data Analysis and Modeling", layout="wide")

# File Uploader
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file)

    # Display Dataset
    st.header("Dataset Overview")
    st.write(data.head())

    # Data Cleaning
    st.sidebar.subheader("Data Cleaning")
    st.sidebar.write("Automatically handle missing values, drop unnecessary columns, etc.")

    if st.sidebar.button("Clean Data"):
        # Drop columns with all null values
        data.dropna(axis=1, how='all', inplace=True)

        # Drop duplicates
        data.drop_duplicates(inplace=True)

        # Convert column names to lowercase and replace spaces with underscores
        data.columns = data.columns.str.lower().str.replace(" ", "_")

        # Handling missing values
        data.fillna(data.mean(numeric_only=True), inplace=True)

        st.success("Data cleaned successfully!")
        st.write(data.head())

    # Data Visualization
    st.header("Data Visualization")
    st.write("Choose visualizations from the sidebar.")

    viz_option = st.sidebar.selectbox(
        "Choose a Visualization",
        ["Correlation Heatmap", "Histograms", "Bar Plots", "Scatter Plots"]
    )

    if viz_option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr = data.select_dtypes(include=np.number).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt)

    elif viz_option == "Histograms":
        st.subheader("Histograms")
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols].hist(bins=20, figsize=(16, 12))
        st.pyplot(plt)

    elif viz_option == "Bar Plots":
        st.subheader("Bar Plots")
        categorical_cols = data.select_dtypes(include="object").columns
        selected_col = st.selectbox("Choose a Categorical Column", categorical_cols)
        if selected_col:
            avg = data.groupby(selected_col).mean()
            avg.plot(kind="bar", figsize=(10, 6))
            st.pyplot(plt)

    elif viz_option == "Scatter Plots":
        st.subheader("Scatter Plots")
        numeric_cols = data.select_dtypes(include=np.number).columns
        x_axis = st.selectbox("X-Axis", numeric_cols)
        y_axis = st.selectbox("Y-Axis", numeric_cols)
        if x_axis and y_axis:
            plt.scatter(data[x_axis], data[y_axis])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            st.pyplot(plt)

    # Linear Regression Example
    st.header("Linear Regression")
    st.write("Build a simple linear regression model.")

    target_var = st.selectbox("Choose Target Variable", data.select_dtypes(include=np.number).columns)
    features = st.multiselect("Choose Features", data.select_dtypes(include=np.number).columns)

    if st.button("Train Linear Regression Model"):
        if target_var and features:
            X = data[features]
            y = data[target_var]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            st.write("Model Coefficients:", model.coef_)
            st.write("Model Intercept:", model.intercept_)
            st.write("Model R² Score:", model.score(X_test, y_test))

            # Scatter Plot of Predictions
            predictions = model.predict(X_test)
            plt.scatter(y_test, predictions)
            plt.xlabel("True Values")
            plt.ylabel("Predictions")
            st.pyplot(plt)

    # Footer
    st.sidebar.info("Streamlit app for data analysis and machine learning.")
else:
    st.write("Please upload a CSV file to proceed.")
