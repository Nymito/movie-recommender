# 🎬 Movie Recommender — ML Proof of Concept

A lightweight **movie recommendation web app** built with **Flask** and **scikit-learn**.  
It uses **content-based filtering** powered by **TF-IDF embeddings** and **cosine similarity** to suggest movies similar to a given title.

This project was designed as a **machine learning proof of concept (POC)** — not a production system — to demonstrate how to:
- preprocess and vectorize text data (from TMDB / Kaggle),
- compute movie embeddings,
- perform similarity search with Nearest Neighbors,
- and serve results via a simple API and frontend.

---

## 🚀 Demo
If deployed, the app is accessible at:
https://movie-recommender-1-24x5.onrender.com/

## 🧠 How It Works

1. The dataset (`movies_soup.csv`) contains combined metadata such as:
   - cast, crew, genres, and keywords.
2. A **TF-IDF vectorizer** converts the text into embeddings.
3. A **KNN (cosine)** model finds the most similar movies.
4. Flask exposes an API endpoint `/recommend` to return JSON results.
5. The `/ui` route serves a clean HTML frontend with poster previews.

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **Flask** — backend API
- **pandas**, **scikit-learn** — data & ML pipeline
- **HTML/CSS/JS** — simple responsive frontend
- **Render** or **Hugging Face Spaces** — for deployment

---
