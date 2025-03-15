import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import re


st.set_page_config(page_title="Interactive Data Visualization Tool", layout="wide")


st.markdown("""
    <style>
        .reportview-container {
            background-color: #f4f7f6;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
        }
        .stButton button {
            background-color: #FF5733;
            color: white;
        }
        .stTextInput input {
            background-color: #f2f2f2;
        }
    </style>
""", unsafe_allow_html=True)


st.sidebar.title("Data Upload & Settings")


st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Interactive Data Visualization Tool</h1>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a CSV or Excel file 📂", type=["csv", "xlsx"])


if uploaded_file:
    
    try:
        # Properly read CSV or Excel file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)  
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl", dtype=str)  # Read Excel properly
        
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])  
            except:
                pass  

        st.markdown("<h2 style='color: #3498db;'>📂 Preview of Uploaded Data</h2>", unsafe_allow_html=True)
        st.write(df.head())  

        
        st.markdown("<h2 style='color: #8E44AD;'>🔍 Data Summary</h2>", unsafe_allow_html=True)
        st.write(f"**Number of Rows:** {df.shape[0]}")
        st.write(f"**Number of Columns:** {df.shape[1]}")
        st.write(f"**Column Names:** {', '.join(df.columns)}")
        
        st.write("**Data Types:**")
        st.write(df.dtypes)

        
        st.markdown("<h2 style='color: #E67E22;'>🎛️ Filter Data</h2>", unsafe_allow_html=True)
        filter_column = st.selectbox("Select Column to Filter By", df.columns.tolist())
        unique_values = df[filter_column].unique()
        selected_value = st.selectbox(f"Select Value for {filter_column}", unique_values)


        filtered_df = df[df[filter_column] == selected_value]
        st.write(filtered_df)

        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        if numeric_columns:
            st.markdown("<h2 style='color: #FF5733;'>📈 Choose Columns for Visualization</h2>", unsafe_allow_html=True)

            
            x_axis = st.sidebar.selectbox("Choose X-axis", options=numeric_columns)
            y_axis = st.sidebar.selectbox("Choose Y-axis", options=numeric_columns)

            
            graph_type = st.sidebar.selectbox("Choose Graph Type", ["Line Plot", "Bar Chart", "Scatter Plot", "Pie Chart", "Heatmap"], help="Select the type of graph to display.")

        
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.set_style("whitegrid")

        
            if graph_type == "Line Plot":
                sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, marker="o", color="#FF5733")
            elif graph_type == "Bar Chart":
                sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax, palette="viridis")
            elif graph_type == "Scatter Plot":
                sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, color="blue", s=100)
            elif graph_type == "Pie Chart":
                pie_data = df.groupby(x_axis)[y_axis].sum()
                pie_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette("Set3", len(pie_data)))
            elif graph_type == "Heatmap":
                df_numeric = df.select_dtypes(include=['number'])
                if not df_numeric.empty:
                    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
                else:
                    st.error("❌ No numeric data available for heatmap.")
                    ax.remove()  

            
            ax.set_title(f"{graph_type} of {y_axis} vs {x_axis}", fontsize=14, color="darkred")


            st.pyplot(fig)

            
            
            st.sidebar.markdown("### 📥 Download Processed Data")
            st.sidebar.download_button(label="Download CSV", data=df.to_csv(index=False), file_name="processed_data.csv", mime="text/csv")

        else:
            st.warning("⚠ No numeric columns found for visualization. Please upload a dataset with numeric values.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("📤 Please upload a file to proceed.")
