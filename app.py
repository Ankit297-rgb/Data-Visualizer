import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import bcrypt
from io import BytesIO


saved_data_dir = "saved_file"


if not os.path.exists(saved_data_dir):
    os.makedirs(saved_data_dir)


def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


def signup():
    st.subheader("🔹 Create an Account")
    new_username = st.text_input("Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_password != confirm_password:
            st.error("❌ Passwords do not match!")
            return

        user_file = os.path.join(saved_data_dir, f"{new_email}_data.json")
        if os.path.exists(user_file):
            st.error("❌ Email already exists! Try logging in.")
            return

        hashed_password = hash_password(new_password)
        
        user_data = {
            "username": new_username,
            "email": new_email,
            "password": hashed_password.decode('utf-8')
        }
        with open(user_file, "w") as f:
            json.dump(user_data, f)
        
        st.success("✅ Account created! Please log in.")


def login():
    st.subheader("🔑 Login to Your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user_file = os.path.join(saved_data_dir, f"{email}_data.json")
        if os.path.exists(user_file):
            with open(user_file, "r") as f:
                user_data = json.load(f)
                if check_password(password, user_data["password"].encode('utf-8')):
                    st.session_state.logged_in = True
                    st.session_state.username = user_data["username"]
                    st.success(f"✅ Welcome back, {user_data['username']}!")
                    st.rerun()  
                else:
                    st.error("❌ Incorrect password!")
        else:
            st.error("❌ No account found with this email!")


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""


st.title("📊 Interactive Data Visualization App with Local File Storage")

if not st.session_state.logged_in:
    st.sidebar.title("🔐 Authentication")
    auth_option = st.sidebar.radio("Choose", ["Login", "Sign Up"])

    if auth_option == "Login":
        login()
    else:
        signup()
else:
    st.sidebar.title(f"👤 Welcome, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    
    uploaded_file = st.file_uploader("Upload CSV or Excel 📂", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        
        user_data_file = os.path.join(saved_data_dir, f"{st.session_state.username}_data.json")
        uploaded_data = df.to_dict(orient="records")
        

        with open(user_data_file, "w") as f:
            json.dump({"username": st.session_state.username, "data": uploaded_data}, f)

        st.write("✅ Data uploaded successfully!")
        st.write(df)

        
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(df)
        st.download_button(label="Download CSV", data=csv_data, file_name="uploaded_data.csv", mime="text/csv")

        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        if numeric_columns:
            x_axis = st.selectbox("Choose X-axis", numeric_columns)
            y_axis = st.selectbox("Choose Y-axis", numeric_columns)
            graph_type = st.selectbox("Choose Graph Type", ["Line Plot", "Bar Chart", "Scatter Plot", "Heatmap"])

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.set_style("whitegrid")

            if graph_type == "Line Plot":
                sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, marker="o", color="#FF5733")
            elif graph_type == "Bar Chart":
                sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax, palette="viridis")
            elif graph_type == "Scatter Plot":
                sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, color="blue", s=100)
            elif graph_type == "Heatmap":
                df_numeric = df.select_dtypes(include=['number'])
                if not df_numeric.empty:
                    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
                else:
                    st.error("❌ No numeric data available for heatmap.")
                    ax.remove()

            ax.set_title(f"{graph_type} of {y_axis} vs {x_axis}", fontsize=14, color="darkred")
            st.pyplot(fig)

            
            graph_filename = os.path.join(saved_data_dir, f"{st.session_state.username}_graph.png")
            fig.savefig(graph_filename)

           
            with open(graph_filename, "rb") as img_file:
                st.download_button(
                    label="Download Graph Image",
                    data=img_file,
                    file_name="graph_image.png",
                    mime="image/png"
                )

        else:
            st.warning("⚠ No numeric columns found for visualization.")

    
    st.sidebar.markdown("### 📂 View Your Uploaded Data")
    if st.sidebar.button("Load My Data"):
        user_data_file = os.path.join(saved_data_dir, f"{st.session_state.username}_data.json")
        if os.path.exists(user_data_file):
            with open(user_data_file, "r") as f:
                user_data = json.load(f)
                df = pd.DataFrame(user_data["data"])
                st.write(df)
        else:
            st.warning("No data found for your account.")

    
    if st.sidebar.button("Delete My Data"):
        user_data_file = os.path.join(saved_data_dir, f"{st.session_state.username}_data.json")
        if os.path.exists(user_data_file):
            os.remove(user_data_file)
            st.success("🗑️ All your uploaded data has been deleted.")
        else:
            st.warning("No data found to delete.")
