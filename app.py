import streamlit as st
import json
import pandas as pd
import google.generativeai as genai
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set your API key directly (for testing purposes)
api_key_google = ""  # Your actual API key here

if api_key_google is None:
    raise ValueError("Google API key is not set. Please set your API key.")

# Configure the Google Generative AI with the provided API key
genai.configure(api_key=api_key_google)

# Set up the page configuration
st.set_page_config(page_title="App with Navigation", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
selected = st.sidebar.radio("Go to", ["Home", "Analytics Dashboard", "Chatbot"])

# Home Page
if selected == "Home":
    st.title("Welcome to the App")
    st.markdown("""
    Use the navigation bar to explore the app.
    1. *Home* - Welcome page.
    2. *Analytics Dashboard* - Visualize and analyze your dataset.
    3. *Chatbot* - Ask questions based on the dataset uploaded.
    """)
    st.image("https://www.example.com/your_image.jpg", width=700)  # Optional image for homepage

# Analytics Dashboard
elif selected == "Analytics Dashboard":
    st.title("🔍 Analytics Dashboard")

    # File uploader to accept both JSON and CSV files
    uploaded_file = st.file_uploader("Upload a JSON or CSV file", type=["json", "csv"])

    if uploaded_file:
        file_type = uploaded_file.type
        with st.spinner('Processing your file...'):
            try:
                if file_type == "application/json":
                    # If JSON file, process it with json.load
                    data = json.load(uploaded_file)
                    st.write("Dataset Preview:", data)

                    # Convert JSON to DataFrame
                    df = pd.json_normalize(data)

                elif file_type == "text/csv":
                    # If CSV file, process it with pandas
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # Changed encoding
                    st.write("Dataset Preview:", df)

                # Data Summary Statistics
                st.subheader("🔢 Data Summary Statistics")
                st.write(df.describe())

                # AI-generated analysis
                analysis_prompt = f"Analyze this data: {df.head().to_dict()}. Provide insights about trends and patterns."
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content(analysis_prompt)
                st.write("AI Generated Analytics:", response.text)

                # Interactive Data Filtering
                st.subheader("🔎 Data Filtering")
                filter_column = st.selectbox("Select a column to filter by", df.columns)
                if df[filter_column].dtype in [np.number, float, int]:
                    min_val, max_val = st.slider("Select range for " + filter_column,
                                                 min_value=float(df[filter_column].min()),
                                                 max_value=float(df[filter_column].max()),
                                                 value=(float(df[filter_column].min()), float(df[filter_column].max())))
                    filtered_df = df[(df[filter_column] >= min_val) & (df[filter_column] <= max_val)]
                else:
                    unique_values = df[filter_column].unique()
                    selected_values = st.multiselect(f"Select values for {filter_column}", unique_values)
                    filtered_df = df[df[filter_column].isin(selected_values)]

                st.write("Filtered Data Preview:", filtered_df)

                # Advanced Visualization
                st.subheader("📊 Choose Graph Type and Columns")

                graph_type = st.selectbox("Choose Graph Type", ["Bar", "Pie", "Scatter", "Line", "Heatmap"])

                if graph_type == "Bar":
                    category_column = st.selectbox("Select Category Column for Bar Graph", df.select_dtypes(include="object").columns)
                    category_counts = filtered_df[category_column].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
                    ax.set_title(f'{category_column} Distribution (Bar Chart)')
                    st.pyplot(fig)

                elif graph_type == "Pie":
                    value_column = st.selectbox("Select Value Column for Pie Chart", df.select_dtypes(include=np.number).columns)
                    value_counts = filtered_df[value_column].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    value_counts.plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
                    ax.set_title(f'{value_column} Distribution (Pie Chart)')
                    st.pyplot(fig)

                elif graph_type == "Scatter":
                    x_col = st.selectbox("Select X Column for Scatter Plot", df.select_dtypes(include=np.number).columns)
                    y_col = st.selectbox("Select Y Column for Scatter Plot", df.select_dtypes(include=np.number).columns)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
                    st.pyplot(fig)

                elif graph_type == "Line":
                    line_column = st.selectbox("Select Column for Line Graph", df.select_dtypes(include=np.number).columns)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.lineplot(data=filtered_df, x=filtered_df.index, y=line_column, ax=ax)
                    ax.set_title(f'{line_column} Line Graph')
                    st.pyplot(fig)

                elif graph_type == "Heatmap":
                    heatmap_columns = st.multiselect("Select Columns for Heatmap", df.select_dtypes(include=np.number).columns)
                    if len(heatmap_columns) > 1:
                        corr = filtered_df[heatmap_columns].corr()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                        ax.set_title("Correlation Heatmap")
                        st.pyplot(fig)
                    else:
                        st.warning("Please select more than one column for the heatmap.")

                # Provide option to download the processed data
                st.markdown("""
                ### 📥 Download Processed Data
                You can download the cleaned dataset after processing.
                """)
                download_button = st.download_button(
                    label="Download Processed Data",
                    data=filtered_df.to_csv(index=False),
                    file_name="processed_data.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error: {e}. Please try uploading a different file.")

# Chatbot
elif selected == "Chatbot":
    st.title("💬 Chatbot with Google Gemini")

    uploaded_file = st.file_uploader("Upload a JSON or CSV file for Q&A", type=["json", "csv"])

    if uploaded_file:
        file_type = uploaded_file.type

        try:
            if file_type == "application/json":
                data = json.load(uploaded_file)
                df = pd.json_normalize(data)
            elif file_type == "text/csv":
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

            question = st.text_input("Ask a question about the data:")

            if question:
                # AI-based response generation
                model = genai.GenerativeModel("gemini-1.5-flash")  
                try:
                    prompt = f"Given the dataset: {df.head().to_dict()}, answer the following question: {question}"
                    response = model.generate_content(prompt)
                    st.write("Chatbot response:", response.text)

                except Exception as e:
                    st.write(f"Error: {e}")
                    st.write("Failed to get a valid response from the Gemini chatbot.")

        except Exception as e:
            st.error(f"Error: {e}. Please try uploading a valid JSON or CSV file.")

    else:
        question = st.text_input("Ask a general question:")

        if question:
            # AI-based general response
            prompt = f"Answer this general question: {question}"
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
            st.write("Chatbot response:", response.text)
