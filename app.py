from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_CSV = os.path.join(BASE_DIR, "movies_soup.csv")
EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")

HF_URL = "https://huggingface.co/datasets/gigilamorani/movie-recomender/resolve/main/embeddings.npy"

movies = None
embeddings = None
knn = None

def load_resources():
    print("📂 Current directory:", os.getcwd())
    print("📄 CSV path:", MOVIES_CSV)
    print("✅ Exists:", os.path.exists(MOVIES_CSV))

    if os.path.exists(MOVIES_CSV):
        print("File size:", os.path.getsize(MOVIES_CSV), "bytes")
        print(pd.read_csv(MOVIES_CSV, nrows=3).head())
    global movies, embeddings, knn
    print("Looking for:", MOVIES_CSV, "=> Exists:", os.path.exists(MOVIES_CSV))

    if os.path.exists(MOVIES_CSV):
        movies = pd.read_csv(MOVIES_CSV)
    else:
        movies = pd.DataFrame(columns=["movieId", "title"])

    if not os.path.exists(EMB_PATH):
        print("⬇️ Téléchargement des embeddings depuis Hugging Face…")
        r = requests.get(HF_URL, stream=True)
        r.raise_for_status()
        with open(EMB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Embeddings téléchargés avec succès.")

    if os.path.exists(EMB_PATH):
        embeddings = np.load(EMB_PATH)


    if embeddings is not None:
        knn = NearestNeighbors(metric="cosine", algorithm="auto").fit(embeddings)

def find_movie_index_by_title(q):
    if movies is None or movies.empty:
        return None
    ql = q.strip().lower()
    # try exact match
    exact = movies[movies['title'].str.lower() == ql]
    if not exact.empty:
        return int(exact.index[0])
    # try startswith
    starts = movies[movies['title'].str.lower().str.startswith(ql)]
    if not starts.empty:
        return int(starts.index[0])
    # fallback to contains
    contains = movies[movies['title'].str.lower().str.contains(ql)]
    if not contains.empty:
        return int(contains.index[0])
    return None

@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    print("Received request:", request.method, request.args if request.method == "GET" else request.get_json())
    data = request.get_json(silent=True) if request.method == "POST" else request.args
    title = data.get("title") if isinstance(data, dict) else data.get("title")
    if not title:
        return jsonify({"error": "missing 'title' parameter"}), 400
    try:
        n = int(data.get("n", 5))
    except:
        n = 5

    # 1) If we have embeddings + knn, use vector similarity
    if knn is not None and embeddings is not None:
        idx = find_movie_index_by_title(title)
        if idx is None:
            return jsonify({"error": "movie not found"}), 404
        neigh = min(n + 1, embeddings.shape[0])
        dists, inds = knn.kneighbors(embeddings[[idx]], n_neighbors=neigh)
        dists = dists[0]
        inds = inds[0]
        results = []
        for dist, i in zip(dists, inds):
            if int(i) == int(idx):
                continue  # skip the query movie itself

            sim = float(max(0.0, 1.0 - dist))  # cosine similarity approx
            row = movies.iloc[int(i)].to_dict() if not movies.empty else {"index": int(i)}
            row = {k: (None if pd.isna(v) else v) for k, v in row.items()}

            poster_path = row.get("poster_url", "")
            row.update({
                "score": sim,
                "poster_url": poster_path
            })

            results.append(row)
            if len(results) >= n:
                break
        return jsonify({"query": title, "results": results}), 200

    return jsonify({"error": "no recommender available (place movies.csv + embeddings.npy or model.pkl next to app.py)"}), 500

if __name__ == "__main__":
    load_resources()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)