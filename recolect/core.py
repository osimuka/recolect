import re
import typing
import pandas as pd
import numpy as np
import nltk
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models.fasttext import FastText as FT_gensim

nltk.download('punkt')  # Download data for tokenizer
nltk.download('stopwords')  # Download stopwords list
nltk.download('wordnet')  # Download data for lemmatizer


__all__ = ['get_recommendations', 'load_and_preprocess_data', 'train']


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a csv file"""
    return pd.read_csv(filepath)


# Precompiled patterns
NON_WORD_PATTERN = re.compile(r'\W')
SINGLE_CHAR_PATTERN = re.compile(r'\s+[a-z]\s+')

# response columns
RESPONSE_COLUMNS = ['title', 'listed_in', 'similarity']


def preprocess_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()

    # Remove all non-word characters & all single characters
    text = NON_WORD_PATTERN.sub(' ', SINGLE_CHAR_PATTERN.sub(' ', text))

    # Tokenization
    tokens = word_tokenize(text)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in ENGLISH_STOP_WORDS]

    return ' '.join(stemmed_tokens)


def load_and_preprocess_data(filepath: str, col: str = None) -> pd.DataFrame:
    data = load_data(filepath)
    data['processed_text'] = data[col or 'description'].apply(lambda x: preprocess_text(x))
    return data


def train(data: pd.DataFrame, col: str = None) -> pd.DataFrame:
    data['processed_text'] = data[col or 'description'].apply(lambda x: preprocess_text(x))
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    model = FT_gensim(vector_size=100, window=5, min_count=5)
    model.build_vocab(train['processed_text'])
    model.train(train['processed_text'], total_examples=len(train['processed_text']), epochs=10)
    return model, test


def get_avg_vector(words, model) -> np.ndarray:
    """Get average vector for a list of words"""
    # Get the vectors for each word in the list
    vectors = [model.wv[word] for word in words if word in model.wv.index_to_key]

    # If no words are in the model, return a vector of zeros
    if not vectors:
        return np.zeros(model.vector_size)

    # Return the average vector
    return np.mean(vectors, axis=0)


def k_means_clustering(model: typing.Any, data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Perform KMeans clustering on the average vectors"""

    # Calculate average feature vectors for the words in each row of the DataFrame
    data['avg_vector'] = data[column_name].apply(lambda x: get_avg_vector(x, model))

    # Create a matrix of the average vectors
    avg_vectors_matrix = np.vstack(data['avg_vector'].to_numpy())

    # Fit the KMeans clustering
    kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto").fit(avg_vectors_matrix)

    # Predict clusters for those average vectors
    data['cluster_id'] = kmeans.predict(avg_vectors_matrix)

    return data


def _get_recommendations_k_means_clustering(title: str, model: typing.Any, data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get recommendations for a title"""

    title_row = data[data["title"] == title].copy()

    # Check if title_row is empty (title not found)
    if title_row.empty:
        return pd.DataFrame(columns=RESPONSE_COLUMNS, data=[])

    # Assign clusters to each row in the data
    data = k_means_clustering(model, data, "processed_text")

    title_row = data[data["title"] == title].copy()

    # Get the cluster of the given title
    title_cluster = title_row["cluster_id"].iloc[0]

    # Filter the DataFrame to only include rows from the same cluster as the given title
    result_df = data[data["cluster_id"] == title_cluster].copy()

    # Exclude the given title itself from the recommendations
    result_df = result_df.drop(result_df[result_df["title"] == title].index)

    # Calculate word2vec similarities between the given title's processed_text and the processed_text of each row in result_df

    result_df["similarity"] = result_df["processed_text"].apply(lambda x: model.wv.n_similarity(title_row["processed_text"].iloc[0], x))  

    # Sort the DataFrame based on similarity values in descending order
    result_df.sort_values(by=["similarity"], ascending=False, inplace=True)

    return result_df[RESPONSE_COLUMNS].head(n)


def _get_cosine_similarity(title: str, model: typing.Any, data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get recommendations for a title"""

    title_row = data[data["title"] == title].copy()

    # Check if title_row is empty (title not found)
    if title_row.empty:
        return pd.DataFrame(columns=RESPONSE_COLUMNS, data=[])

    # Calculate average feature vectors for the words in each row of the DataFrame
    data['avg_vector'] = data["processed_text"].apply(lambda x: get_avg_vector(x, model))

    # Create a matrix of the average vectors
    avg_vectors_matrix = np.vstack(data['avg_vector'].to_numpy())

    # Find the vector for the given title
    title_vector = data[data['title'] == title]['avg_vector'].iloc[0].reshape(1, -1)

    # Compute the cosine similarity between the title vector and all other vectors
    cosine_sim = cosine_similarity(title_vector, avg_vectors_matrix)

    # Add the similarity scores to the dataframe
    data['similarity'] = cosine_sim[0]

    # Sort by similarity
    result_df = data.sort_values(by="similarity", ascending=False)

    # Exclude the movie itself and get the top n
    result_df = result_df[result_df["title"] != title]

    return result_df[RESPONSE_COLUMNS].head(n)


def get_recommendations(title: str, model: typing.Any, data: pd.DataFrame, n: int = 10, method: str = "k_means_clustering") -> pd.DataFrame:
    """Get recommendations for a title"""
    METHODS = {
        "k_means_clustering": _get_recommendations_k_means_clustering,
        "cosine_similarity": _get_cosine_similarity
    }
    return METHODS[method](title, model, data, n)
