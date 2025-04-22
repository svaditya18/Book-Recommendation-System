
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim import corpora, models
import time
import os


# Replace your NLTK setup section with this:
# Initialize NLTK resources
def initialize_nltk():
    try:
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)

        nltk.data.path = [nltk_data_dir] + nltk.data.path

        # Download necessary resources
        required_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in required_resources:
            nltk.download(resource, download_dir=nltk_data_dir)

        # Verify that 'punkt' was successfully downloaded
        nltk.data.find('tokenizers/punkt')

    except Exception as e:
        st.error(f"Error initializing NLTK: {e}")
        st.stop()


# Initialize NLTK at the start
initialize_nltk()


# Set page config
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)




 # Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
    
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)
        try:
            tokens = nltk.word_tokenize(text.lower())  # Use nltk.word_tokenize
        except Exception as e:
            st.error(f"Tokenization error: {e}")
            return ""  # Or handle the error as appropriate
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    return ""


def get_lda_features(text, dictionary, lda_model):
    tokens = preprocess_text(text).split()
    bow = dictionary.doc2bow(tokens)
    vec = np.zeros(lda_model.num_topics)
    for topic_id, prob in lda_model[bow]:
        vec[topic_id] = prob
    return vec

@st.cache_resource
def load_data_and_models():
    """Load data and train models (cached for performance)"""
    # Load data with encoding fallback
    try:
        csv_url = "https://raw.githubusercontent.com/svaditya18/Book-Recommendation-System/main/data/book_details.csv"
        df = pd.read_csv(csv_url, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_url, encoding='latin1', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")
            st.stop()

    # Continue with data processing
    df = df[["title", "description"]].dropna().reset_index(drop=True)
    df = df[:5000]  # Limit dataset size

    # Preprocess descriptions
    df['processed_description'] = df['description'].apply(preprocess_text)

    # Drop empty processed rows
    original_count = len(df)
    df = df[df['processed_description'].str.strip() != ""]
    removed = original_count - len(df)
    if removed > 0:
        st.warning(f"{removed} books removed due to empty processed descriptions.")

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_description'])

    # Load Universal Sentence Encoder
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = model(df['description'].tolist()).numpy()

    # Topic Modeling (LDA)
    processed_descriptions_for_lda = [text.split() for text in df['processed_description']]
    dictionary = corpora.Dictionary(processed_descriptions_for_lda)
    corpus = [dictionary.doc2bow(text) for text in processed_descriptions_for_lda]
    lda_model = models.LdaModel(corpus, num_topics=20, random_state=42, id2word=dictionary)

    # Generate LDA features
    df['lda_features'] = df['description'].apply(lambda x: get_lda_features(x, dictionary, lda_model))
    lda_features_array = np.vstack(df['lda_features'].values)

    # Combine features and train Nearest Neighbors
    combined_features = np.hstack([embeddings, tfidf_matrix.toarray(), lda_features_array])
    nn = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn.fit(combined_features)

    return df, model, tfidf_vectorizer, nn, dictionary, lda_model, embeddings, tfidf_matrix


def get_recommendations(query, method, df, model, tfidf_vectorizer, nn, dictionary, lda_model, embeddings, tfidf_matrix):
    """Get book recommendations based on the query and method"""
    processed_text = preprocess_text(query)
    text_embedding = model([query]).numpy()
    text_tfidf = tfidf_vectorizer.transform([processed_text]).toarray()
    lda_features = get_lda_features(query, dictionary, lda_model).reshape(1, -1)
    
    if method == 'Combined Features':
        combined_text_features = np.hstack([text_embedding, text_tfidf, lda_features])
        neighbors = nn.kneighbors(combined_text_features, return_distance=False)[0]
        recommendations = df.iloc[neighbors]
    elif method == 'Embeddings Only':
        similarity_scores = cosine_similarity(text_embedding, embeddings)[0]
        sorted_indices = np.argsort(similarity_scores)[::-1][1:11]
        recommendations = df.iloc[sorted_indices]
    else:  # TF-IDF Only
        nn_tfidf = NearestNeighbors(n_neighbors=10, metric='cosine')
        nn_tfidf.fit(tfidf_matrix)
        neighbors = nn_tfidf.kneighbors(text_tfidf, return_distance=False)[0]
        recommendations = df.iloc[neighbors]
    
    return recommendations

# Main App
def main():
    st.title("📚 Book Recommendation System")
    st.write("Discover books tailored to your interests!")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        method = st.radio(
            "Recommendation Method:",
            ("Combined Features", "Embeddings Only", "TF-IDF Only"),
            index=0
        )
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("""
        This app recommends books based on:
        - **Combined Features**: Uses embeddings, TF-IDF, and topic modeling (most accurate)
        - **Embeddings Only**: Uses Universal Sentence Encoder
        - **TF-IDF Only**: Traditional keyword-based approach
        """)
    
    # Main content
    query = st.text_input("What kind of books are you interested in?", 
                         placeholder="e.g., science fiction, romance, mystery")
    
    if query:
        with st.spinner('Loading models and data...'):
            df, model, tfidf_vectorizer, nn, dictionary, lda_model, embeddings, tfidf_matrix = load_data_and_models()
        
        with st.spinner('Finding recommendations...'):
            recommendations = get_recommendations(
                query, method, df, model, tfidf_vectorizer, nn, dictionary, lda_model, embeddings, tfidf_matrix
            )
            time.sleep(1)  # For dramatic effect
        
        st.subheader(f"Recommended Books ({method})")
        
        # Display recommendations
        for idx, row in recommendations.iterrows():
            with st.expander(f"**{row['title']}**"):
                st.write(row['description'])
                st.caption(f"Similarity score: {idx}")  # Placeholder for actual score

if __name__ == '__main__':
    main()
