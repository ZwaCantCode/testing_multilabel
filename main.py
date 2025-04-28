from utils.data_loader import load_data
from utils.feature_engineering import feature_engineering  # --- perubahan: Import feature engineering
from utils.model import train_model, predict  # --- perubahan: Import training dan prediksi
import streamlit as st

# --- perubahan: Konfigurasi halaman
st.set_page_config(
    page_title="Multi-label Text Classification",
    layout="wide"
)

# --- perubahan: Session state untuk model, vectorizer, dan data
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
if 'X' not in st.session_state:  # --- perubahan: Session state untuk fitur hasil feature engineering
    st.session_state.X = None

# --- perubahan: Sidebar navigasi
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Loading", "Feature Engineering", "Model Training", "Prediction"])

# --- perubahan: Halaman Data Loading
if page == "Data Loading":
    st.title("Data Loading")
    st.write("### Dataset Overview")
    df = st.session_state.df
    st.write(f"Number of samples: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")
    st.dataframe(df.head())

# --- perubahan: Halaman Feature Engineering
# elif page == "Feature Engineering":
#     st.title("Feature Engineering")
#     st.write("### Processing text data into features...")
#     df = st.session_state.df
#     X, vectorizer = feature_engineering(df['review'])  # Asumsikan kolom teks namanya 'review'
#     st.session_state.X = X
#     st.session_state.vectorizer = vectorizer
#     st.success("Feature engineering completed!")

# --- perubahan: Halaman Model Training
elif page == "Model Training":
    st.title("Model Training")
    if st.session_state.X is None:
        st.warning("Please complete feature engineering first.")
    else:
        model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "XGBoost"])
        if st.button("Train Model"):
            model, label_columns = train_model(st.session_state.X, st.session_state.df, model_choice)
            st.session_state.trained_model = model
            st.session_state.model_name = model_choice
            st.session_state.label_columns = label_columns
            st.success(f"Model {model_choice} trained successfully!")

# --- perubahan: Halaman Prediction
elif page == "Prediction":
    st.title("Prediction")
    if st.session_state.trained_model is None:
        st.warning("Please train a model first.")
    else:
        input_text = st.text_area("Enter a review text to predict:")
        if st.button("Predict"):
            features = st.session_state.vectorizer.transform([input_text])
            preds = predict(st.session_state.trained_model, features, st.session_state.label_columns)
            st.write("### Prediction Results:")
            st.json(preds)
