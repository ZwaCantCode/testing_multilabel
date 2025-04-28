from utils.data_loader import load_data
import streamlit as st


st.set_page_config(
    page_title="Multi-label Text Classification",
    layout="wide"
)


if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'label_columns' not in st.session_state:
    st.session_state.label_columns = None


if 'df' not in st.session_state:
    st.session_state.df = load_data()

st.title("Automotive Reviews Multi-label Text Classification")
st.markdown("Multi-label classification for automotive reviews across different aspects: fuel, machine, and parts.")

st.write("""
## Welcome to the Multi-label Text Classification App
         
This application demonstrates text classification that can predict multiple labels simultaneously.

### Available Pages:

1. **Dataset Explorer** - Explore and understand the dataset
2. **Model Training** - Train and evaluate multi-label classification models
3. **Prediction** - Make predictions on new text inputs

Use the sidebar to navigate between pages.
""")

st.subheader("Dataset Overview")
df = st.session_state.df
st.write(f"Number of samples: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")
st.dataframe(df.head(5))
