import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from clean import clean_dataframe, RU_STOPWORDS
import os

###############################################################################
# Конфигурация
###############################################################################

CONFIG = {
    "kmeans": {
        "n_clusters": 12,
        "tfidf_max_features": 5000,
    },
    "hdbscan": {
        "min_cluster_size": 8, # меньше шума, но не дробит мелкие темы
        "umap_n_neighbors": 15,
        "umap_n_components": 8,
    },
    "embedding": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2", 
        "batch_size": 64,
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
def get_embedder():
    return SentenceTransformer(CONFIG["embedding"]["model"])

###############################################################################
# создали векторайзер
def get_tfidf():
    return TfidfVectorizer(max_features=CONFIG["kmeans"]["tfidf_max_features"], ngram_range=(1, 2), min_df=3, max_df=0.95, stop_words=list(RU_STOPWORDS))

###############################################################################
# Топ-N слов для каждого центроида (для интерпретации кластеров)
def get_top_words(centroids, feature_names, topn=8):
    top_words = []
    for cent in centroids:
        # индексы самых больших весов
        indices = np.argsort(-cent)[:topn]
        top_words.append([feature_names[i] for i in indices])
    return top_words

###############################################################################
def cluster_kmeans(df, text_col="text_clean"):
    vectorizer = get_tfidf()
    X = vectorizer.fit_transform(df[text_col].tolist())
    km = KMeans(n_clusters=CONFIG["kmeans"]["n_clusters"], random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    logger.info("KMeans: готово")
    
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    top_words_list = get_top_words(km.cluster_centers_, feature_names)
    
    for i, words in enumerate(top_words_list):
        topics[f"cluster_{i}"] = words
    
    return pd.Series(labels, name="cluster"), topics

###############################################################################
def cluster_hdbscan(df, text_col="text_clean"):
    embedder = get_embedder()
    embeddings = embedder.encode(df[text_col].tolist(), show_progress_bar=True, batch_size=CONFIG["embedding"]["batch_size"])
    embeddings = normalize(embeddings)
    
    proj = umap.UMAP(
        n_neighbors=CONFIG["hdbscan"]["umap_n_neighbors"],
        n_components=CONFIG["hdbscan"]["umap_n_components"],
        min_dist=0.0,
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CONFIG["hdbscan"]["min_cluster_size"],
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(proj)
    
    # для каждого кластера (кроме шума) - топ-слова через TF-IDF
    vectorizer = get_tfidf()
    X_tfidf = vectorizer.fit_transform(df[text_col].tolist())
    feature_names = vectorizer.get_feature_names_out()
    
    topics = {}
    unique_labels = set(labels)
    
    for lbl in unique_labels:
        if lbl == -1:
            topics["noise"] = []
            continue
        
        mask = labels == lbl
        # средний TF-IDF вектор по кластеру
        cluster_center = X_tfidf[mask].mean(axis=0).A1  # .A1 превращает матрицу в плоский массив
        top_indices = np.argsort(-cluster_center)[:8]
        topics[f"cluster_{lbl}"] = [feature_names[i] for i in top_indices]    
    return pd.Series(labels, name="cluster"), topics

###############################################################################
def save_results(df, labels, topics, method_name, output_dir):
    out_csv = output_dir / f"clusters_{method_name}.csv"
    pd.DataFrame({"id": df["id"], "cluster": labels.values}).to_csv(out_csv, index=False)
    logger.info(f"Сохранено: {out_csv}")
    
    out_json = output_dir / f"topics_{method_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    logger.info(f"Сохранено: {out_json}")

###############################################################################
def run_all(input_parquet, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_raw = pd.read_parquet(input_parquet)
    df = clean_dataframe(df_raw)
    clean_path = output_dir / "posts_clean.parquet"
    df[["id", "date", "text_clean", "tickers"]].to_parquet(clean_path, index=False)

    # KMeans
    labels_kmeans, topics_kmeans = cluster_kmeans(df)
    save_results(df, labels_kmeans, topics_kmeans, "kmeans", output_dir)
    
    # HDBSCAN
    labels_hdb, topics_hdb = cluster_hdbscan(df)
    save_results(df, labels_hdb, topics_hdb, "hdbscan", output_dir)

###############################################################################
if __name__ == "__main__":
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    run_all(data_dir / "posts.parquet", out_dir)