import re
import typing
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from gensim.models.fasttext import FastText as FT_gensim


__all__ = ['get_recommendations', 'load_and_preprocess_data', 'train']


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a csv file"""
    return pd.read_csv(filepath)


def preprocess_text(text: str) -> str:
    text = text.lower()  # Convert to lowercase
    # Remove all non-word characters & all single characters
    text = re.sub(r'\W', ' ', re.sub(r'\s+[a-z]\s+', ' ', text))
    text = ' '.join([PorterStemmer().stem(word) for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text


def load_and_preprocess_data(filepath: str, col: str = None) -> pd.DataFrame:
    data = load_data(filepath)
    data['processed_text'] = data[col or 'description'].apply(lambda x: preprocess_text(x))
    return data


def k_means_clustering(model: typing.Any, data: pd.DataFrame) -> pd.DataFrame:
    """Train k-means clustering with feature vector Add cluster_id on dataframe"""
    ft_vectors = [model.wv[word] for word in model.wv.vocab]
    ft_vectors = np.array(ft_vectors)
    kmeans = KMeans(n_clusters=50, random_state=42).fit(ft_vectors)
    data['cluster_id'] = kmeans.predict(ft_vectors)
    return data


def train(data: pd.DataFrame, col: str = None) -> pd.DataFrame:
    data['processed_text'] = data[col or 'description'].apply(lambda x: preprocess_text(x))
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    model = FT_gensim(vector_size=100, window=5, min_count=5)
    model.build_vocab(train['processed_text'])
    model.train(train['processed_text'], total_examples=len(train['processed_text']), epochs=10)
    return model, test


def _get_recommendations_k_means_clustering(title: str, model: typing.Any, data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get recommendations for a title"""

    data = k_means_clustering(model, data)
    title_row = data[data["title"] == title].copy()
    result_df = data[data["cluster_id"].isin(title_row["cluster_id"])].copy()
    result_df = result_df.drop(result_df[result_df["title"] == title].index)
    result_df["similarity"] = result_df.apply(lambda x: model.wv.n_similarity(title_row["processed_text"], x["processed_text"]), axis=1)
    result_df.sort_values(by=["similarity"], ascending=False, inplace=True)
    return result_df[['title', 'similarity']].head(n)


def _get_cosine_similarity(title: str, model: typing.Any, data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get recommendations for a title"""

    title_row = data[data["title"] == title].copy()

    # Check if title_row is empty (title not found)
    if title_row.empty:
        return pd.DataFrame(columns=['title', 'similarity'], data=[])

    result_df = data.copy()
    cv = CountVectorizer()

    converted_metrix = cv.fit_transform(result_df["processed_text"])
    cosine_sim = cosine_similarity(converted_metrix)
    result_df["similarity"] = cosine_sim[result_df.index, title_row.index]
    result_df.sort_values(by=["similarity"], ascending=False, inplace=True)
    # ignore the first score because it will give us a 100% score because it's the same
    return result_df[['title', 'similarity']].head(n)[1:]


def get_recommendations(title: str, model: typing.Any, data: pd.DataFrame, n: int = 10, method: str = "k_means_clustering") -> pd.DataFrame:
    """Get recommendations for a title"""
    METHODS = {
        "k_means_clustering": _get_recommendations_k_means_clustering,
        "cosine_similarity": _get_cosine_similarity
    }
    return METHODS[method](title, model, data, n)
