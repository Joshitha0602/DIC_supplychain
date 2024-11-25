import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Data Cleaning Demo", layout="wide")

# Function to load data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Sidebar for file upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file).copy()
    st.title("Uploaded Dataset")
    st.write(data.head())

    # Data Overview
    st.header("Dataset Overview")
    st.write("Shape of dataset:", data.shape)
    st.write("Data types and missing values:")
    st.write(data.info())
    st.write("Summary statistics:")
    st.write(data.describe())

    # Missing Values
    st.subheader("Missing Value Analysis")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        "Missing Values": missing_values,
        "Percentage": missing_percentage
    }).sort_values(by="Missing Values", ascending=False)
    st.write(missing_df)

    # Drop columns with all missing values
    if st.sidebar.checkbox("Drop Columns with All Missing Values"):
        data = data.dropna(axis=1, how="all")
        st.write("Updated dataset after dropping columns with all missing values:")
        st.write(data.head())

    # Drop rows with missing values
    if st.sidebar.checkbox("Drop Rows with Missing Values"):
        data = data.dropna()
        st.write("Updated dataset after dropping rows with missing values:")
        st.write(data.head())

    # Data Cleaning Options
    st.sidebar.header("Data Cleaning")
    columns_to_drop = st.sidebar.multiselect("Select Columns to Drop", data.columns)
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
        st.write(f"Dropped Columns: {columns_to_drop}")
        st.write("Updated Dataset:")
        st.write(data.head())

    # Normalize Text Columns
    st.sidebar.header("Text Normalization")
    text_columns = st.sidebar.multiselect("Select Text Columns to Normalize", data.columns)
    if text_columns:
        for col in text_columns:
            data[col] = data[col].str.lower().str.strip()
        st.write("Text normalization applied to columns:")
        st.write(text_columns)

    # Visualizations
    st.header("Visualizations")

    # Correlation Heatmap
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        corr = data.select_dtypes(include=["float", "int"]).corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

    # Histograms for Numeric Columns
    if st.sidebar.checkbox("Show Histograms"):
        st.subheader("Histograms")
        numeric_cols = data.select_dtypes(include=["float", "int"]).columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(data[col], kde=True)
            plt.title(f"Histogram for {col}")
            st.pyplot(plt)
