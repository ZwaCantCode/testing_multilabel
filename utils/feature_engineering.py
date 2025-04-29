from sklearn.feature_extraction.text import TfidfVectorizer

def feature_engineering(text_series):
    
    vectorizer = TfidfVectorizer(
        min_df=0.0,
        max_df=1.0,
        norm='l2',
        use_idf=True,
        smooth_idf=True
    )
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer