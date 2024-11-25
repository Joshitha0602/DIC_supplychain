import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Page configuration
st.set_page_config(page_title="Data Cleaning Demo", layout="wide")

st.title("Data Analysis and Machine Learning App")

# Function to load data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Sidebar for file upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
        data = load_data(uploaded_file).copy()
        st.title("Uploaded Dataset")
        st.write(data.head())
        
        # Proceed only if `data` is not empty
        if not data.empty:
            # Data Cleaning
            st.header("Data Cleaning")
            if st.button("Check Missing Values"):
                missing_values = data.isnull().sum()
                st.write("Missing Values:")
                st.write(missing_values)

            if st.button("Drop Missing Values"):
                data = data.dropna()
                st.write("Missing values dropped. Remaining data:")
                st.dataframe(data.head())

            # Exploratory Data Analysis (EDA)
            st.header("Exploratory Data Analysis (EDA)")
            if st.button("Show Descriptive Statistics"):
                st.write(data.describe())

            if st.button("Correlation Heatmap"):
                numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                correlation_matrix = data[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            if st.button("Show Histograms"):
                numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                fig, ax = plt.subplots(figsize=(16, 12))
                data[numeric_cols].hist(ax=ax, bins=20)
                plt.suptitle("Histograms for Numerical Columns")
                st.pyplot(fig)

            # Machine Learning Models
            st.header("Machine Learning Models")

            # Feature Selection
            st.sidebar.header("Feature Selection")
            features = st.sidebar.multiselect("Select Features for Analysis", data.columns)
            target = st.sidebar.selectbox("Select Target Variable (if applicable)", data.columns)

            if features and target:
                st.write(f"Selected Features: {features}")
                st.write(f"Selected Target: {target}")

                # Linear Regression
                if st.checkbox("Linear Regression"):
                    if len(features) >= 2:
                        X = data[features]
                        Y = data[target]
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

                        model = LinearRegression()
                        model.fit(X_train, Y_train)
                        Y_pred = model.predict(X_test)

                        st.write("Linear Regression Model Trained")
                        st.write("R² Score:", model.score(X_test, Y_test))

                        fig, ax = plt.subplots()
                        ax.scatter(X_test.iloc[:, 0], Y_test, color="yellow", label="Actual")
                        ax.plot(X_test.iloc[:, 0], Y_pred, color="blue", label="Predicted")
                        ax.set_title("Feature 1 vs Target")
                        ax.set_xlabel("Feature 1")
                        ax.set_ylabel("Target")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.error("Please select at least 2 features for Linear Regression.")

                # K-Means Clustering
                if st.checkbox("K-Means Clustering"):
                    if len(features) >= 2:
                        X = data[features]
                        model_kmeans = KMeans(n_clusters=3, random_state=42)
                        data["cluster"] = model_kmeans.fit_predict(X)

                        st.write("K-Means Clustering Completed")
                        fig, ax = plt.subplots()
                        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=data["cluster"], cmap="viridis")
                        ax.set_title("K-Means Clustering")
                        ax.set_xlabel("Feature 1")
                        ax.set_ylabel("Feature 2")
                        st.pyplot(fig)
                    else:
                        st.error("Please select at least 2 features for K-Means Clustering.")

                # K-Nearest Neighbors
                if st.checkbox("K-Nearest Neighbors"):
                    if len(features) >= 2:
                        X = data[features]
                        Y = data[target]
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

                        knn = KNeighborsClassifier(n_neighbors=5)
                        knn.fit(X_train, Y_train)
                        Y_pred = knn.predict(X_test)

                        st.write("Confusion Matrix:")
                        st.write(confusion_matrix(Y_test, Y_pred))
                        st.write("Classification Report:")
                        st.write(classification_report(Y_test, Y_pred))

                        fig, ax = plt.subplots()
                        scatter = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=Y_pred, cmap="viridis")
                        legend = ax.legend(*scatter.legend_elements(), title="Classes")
                        ax.add_artist(legend)
                        ax.set_title("KNN Predictions")
                        ax.set_xlabel("Feature 1")
                        ax.set_ylabel("Feature 2")
                        st.pyplot(fig)
                    else:
                        st.error("Please select at least 2 features for K-Nearest Neighbors.")
        else:
            st.error("Uploaded dataset is empty. Please check your file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a dataset to proceed.")
