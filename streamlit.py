import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit title
st.title("Machine Learning Algorithm Explorer")

# File upload
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # Specify encoding
    st.write("Data Preview:")
    st.write(data.head())

    # Select features and target column
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect("Select Features", options=data.columns)
    target = st.sidebar.selectbox("Select Target Column", options=data.columns)

    if features and target:
        # Splitting dataset
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sidebar options for algorithms
        st.sidebar.header("Choose Algorithm")
        algorithm = st.sidebar.selectbox(
            "Select Algorithm",
            ["Linear Regression", "Logistic Regression", "KMeans Clustering", "KNN Classifier",
             "Decision Tree", "Random Forest", "LinearSVC"]
        )

        if algorithm == "Linear Regression":
            st.subheader("Linear Regression")
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("R² Score:", model.score(X_test, y_test))
            st.write("Coefficients:", model.coef_)
            st.write("Intercept:", model.intercept_)

            # Visualization
            plt.scatter(X_test.iloc[:, 0], y_test, color='yellow', label="Actual")
            plt.plot(X_test.iloc[:, 0], y_pred, color='blue', linewidth=2, label="Predicted")
            plt.title("Linear Regression: Predictions vs Actual")
            plt.legend()
            st.pyplot(plt)

        elif algorithm == "Logistic Regression":
            st.subheader("Logistic Regression")
            model = LogisticRegression(class_weight='balanced', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        elif algorithm == "KMeans Clustering":
            st.subheader("KMeans Clustering")
            num_clusters = st.slider("Number of Clusters", 2, 10, value=3)
            model = KMeans(n_clusters=num_clusters, random_state=42)
            model.fit(X)
            clusters = model.predict(X)

            st.write("Inertia (WCSS):", model.inertia_)

            # Visualization
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
            plt.title("KMeans Clustering")
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            st.pyplot(plt)

        elif algorithm == "KNN Classifier":
            st.subheader("KNN Classifier")
            k = st.slider("Number of Neighbors (k)", 1, 20, value=5)
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Confusion Matrix:")
            st.text(confusion_matrix(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        elif algorithm == "Decision Tree":
            st.subheader("Decision Tree")
            max_depth = st.slider("Max Depth", 1, 10, value=3)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Accuracy:", accuracy_score(y_test, y_pred))

            # Visualization
            plt.figure(figsize=(12, 8))
            plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
            st.pyplot(plt)

        elif algorithm == "Random Forest":
            st.subheader("Random Forest")
            n_estimators = st.slider("Number of Trees", 10, 200, value=100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        elif algorithm == "LinearSVC":
            st.subheader("Linear SVC")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = LinearSVC(dual=False, max_iter=5000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Accuracy:", accuracy_score(y_test, y_pred))

        else:
            st.write("Algorithm not implemented yet!")
