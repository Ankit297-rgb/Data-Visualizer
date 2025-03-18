import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from io import BytesIO

st.title("📊 Interactive Machine Learning with Feature Selection")

uploaded_file = st.file_uploader("Upload CSV File 📂", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Data uploaded successfully!")
    st.write(df)

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_columns:
        st.error("❌ No numeric columns found in the dataset. Please upload a valid dataset.")
    else:
        selected_features = st.multiselect("Select Features for Machine Learning (X)", numeric_columns)

        if not selected_features:
            st.warning("⚠️ Please select at least one feature for training.")
        else:
            target = st.selectbox("Select Target Column (Y)", numeric_columns)

            if target in selected_features:
                st.error("❌ Target column cannot be part of features. Please select different features.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[target], test_size=0.2, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                unique_values = df[target].nunique()
                is_classification = unique_values <= 10

                if is_classification:
                    st.subheader("🔍 Detected: **Classification Task**")
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    st.subheader("🔍 Detected: **Regression Task**")
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                if is_classification:
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    cm = confusion_matrix(y_test, y_pred)

                    st.subheader("📈 Model Performance Metrics")
                    st.success(f"✅ **Accuracy**: {accuracy:.2f}%")

                    st.subheader("🔍 Confusion Matrix Heatmap")
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    ax_cm.set_title("Confusion Matrix")
                    st.pyplot(fig_cm)

                    buf_cm = BytesIO()
                    fig_cm.savefig(buf_cm, format="png")
                    st.download_button("Download Confusion Matrix Heatmap", data=buf_cm.getvalue(), file_name="confusion_matrix.png", mime="image/png")

                    st.subheader("📊 Feature Correlation Heatmap")
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
                    ax_corr.set_title("Feature Correlation Matrix")
                    st.pyplot(fig_corr)

                    buf_corr = BytesIO()
                    fig_corr.savefig(buf_corr, format="png")
                    st.download_button("Download Correlation Heatmap", data=buf_corr.getvalue(), file_name="correlation_heatmap.png", mime="image/png")

                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    accuracy = 100 - (np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

                    st.subheader("📈 Model Performance Metrics")
                    st.success(f"✅ **Regression Accuracy**: {accuracy:.2f}%")
                    st.info(f"🔹 **R² Score**: {r2:.4f}")
                    st.info(f"🔹 **Mean Squared Error (MSE)**: {mse:.4f}")
                    st.info(f"🔹 **Mean Absolute Error (MAE)**: {mae:.4f}")

                    fig_pred, ax_pred = plt.subplots(figsize=(6, 4))
                    sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
                    ax_pred.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
                    ax_pred.set_xlabel("Actual Values")
                    ax_pred.set_ylabel("Predicted Values")
                    ax_pred.set_title("Actual vs Predicted Values")
                    st.pyplot(fig_pred)

                    buf_pred = BytesIO()
                    fig_pred.savefig(buf_pred, format="png")
                    st.download_button("Download Prediction Graph", data=buf_pred.getvalue(), file_name="prediction_graph.png", mime="image/png")

                    st.subheader("📊 Feature Correlation Heatmap")
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
                    ax_corr.set_title("Feature Correlation Matrix")
                    st.pyplot(fig_corr)

                    buf_corr = BytesIO()
                    fig_corr.savefig(buf_corr, format="png")
                    st.download_button("Download Correlation Heatmap", data=buf_corr.getvalue(), file_name="correlation_heatmap.png", mime="image/png")

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")
