# 🚀 Spaceship Titanic — Streamlit Deployment

## Project Structure

```
spaceship_titanic_app/
├── app.py               ← Streamlit application
├── requirements.txt     ← Python dependencies
├── best_model.pkl       ← (you generate this — see Step 1)
├── scaler.pkl           ← (you generate this — see Step 1)
└── feature_names.pkl    ← (you generate this — see Step 1)
```

---

## Step 1 — Export the model from your notebook

Run **Cell 21** in `Spaceship_Titanic.ipynb`. It saves three files:

```
best_model.pkl
scaler.pkl
feature_names.pkl
```

Copy those three files into the same folder as `app.py`.

---

## Step 2 — Install dependencies locally

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run locally

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Step 4 — Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **GitHub repo** (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repo, branch, and set the main file to `app.py`.
4. Click **Deploy**. Your app will be live at a public URL in ~2 minutes.

> ⚠️ Make sure the `.pkl` files are committed to the repo — Streamlit Cloud needs them.

---

## Step 5 — Optional: Deploy to Hugging Face Spaces

1. Create a new **Space** at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK.
3. Upload `app.py`, `requirements.txt`, and the three `.pkl` files.
4. The app starts automatically.

---

## Features

- **Passenger form** — enter all relevant features from the dataset
- **Prediction** — instant Transported / Not Transported result
- **Confidence bar** — shows model probability for each class
- **Passenger summary** — expandable review of inputs
- **Auto-detects model type** — works whether Random Forest or SVM won
