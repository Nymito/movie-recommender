from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MOVIES_CSV = os.path.join(BASE_DIR, "movies_soup.csv")
EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")


movies = None
embeddings = None
knn = None

def load_resources():
    global movies, embeddings, knn
    if os.path.exists(MOVIES_CSV):
        movies = pd.read_csv(MOVIES_CSV)
    else:
        movies = pd.DataFrame(columns=["movieId", "title"])

    if os.path.exists(EMB_PATH):
        embeddings = np.load(EMB_PATH)
        # lazy import to keep startup slight
        knn = NearestNeighbors(metric="cosine", algorithm="auto")
        knn.fit(embeddings)


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