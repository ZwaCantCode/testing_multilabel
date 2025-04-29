import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.feature_engineering import feature_engineering
from utils.data_loader import load_data

if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'X' not in st.session_state:
    st.session_state.X = None

st.header("Feature Engineering")
df = st.session_state.df

# Pilih kolom teks
text_columns = df.columns.tolist()
selected_column = st.selectbox(
    "Pilih kolom:", 
    text_columns, 
    index=text_columns.index("sentence") if "sentence" in text_columns else 0
)

if st.button("Run"):
    X, vectorizer = feature_engineering(df[selected_column])
    st.session_state.X = X
    st.session_state.vectorizer = vectorizer
    st.success(f"Feature engineering selesai untuk kolom '{selected_column}'!")

    st.subheader("Hasil")
    vocab = vectorizer.get_feature_names_out()
    st.write(f"Jumlah (kata unik): {len(vocab)}")

    import pandas as pd
    import numpy as np

    try:
        X_array = X.toarray()
    except:
        X_array = X 

    df_features = pd.DataFrame(X_array, columns=vocab)
    df_features.insert(0, "sentence", df[selected_column].values)

    st.dataframe(df_features.head(10))
