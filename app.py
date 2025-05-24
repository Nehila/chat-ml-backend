from pathlib import Path
from functools import lru_cache
from flask_cors import CORS
from flask import Flask, request, jsonify, abort
import joblib
import nltk
import torch
from transformers import pipeline
import re, string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent / "models"

def parse_name(raw_name: str):
    name = raw_name.lower()
    if name.startswith("all_data"):
        dataset = "All_Data"
        suffix = name[len("all_data_"):]
    else:
        dataset, suffix = name.split("_", 1)
        dataset = dataset.upper()
    family = suffix.split("_")[0]
    return dataset, family

def map_binary(idx: int) -> str:          # 0/1 → Negative/Positive
    return "Positive" if idx == 1 else "Negative"

def map_yelp(idx: int) -> str:            # 0/1/2 → Neg/Pos/Neu
    return ("Negative", "Positive", "Neutral")[idx]

def label_from_hf(hf_label: str, dataset: str) -> str:
    """Convert HF 'LABEL_0/1/2' to human‐readable."""
    num = int(hf_label.split("_")[-1])
    if dataset.lower() == "yelp":
        return map_yelp(num)
    return map_binary(num)

@lru_cache(maxsize=8)
def load_model(raw_name: str):
    dataset, family = parse_name(raw_name)

    if family == "bert":
        model_dir = BASE_DIR / dataset / f"{dataset}_bert_model"
        if not model_dir.exists():
            abort(404, description=f"BERT model folder not found: {model_dir}")

        device = 0 if torch.cuda.is_available() else -1

        pipe = pipeline(
            task="text-classification",
            model=str(model_dir),
            top_k=1,
            device=device
        )

        def _predict(texts):
            outs = pipe(texts if isinstance(texts, list) else [texts])
            flat = [o[0] if isinstance(o, list) else o for o in outs]

            results = [
                {
                    "result": label_from_hf(d["label"], dataset),
                    "score": float(d["score"])
                }
                for d in flat
            ]
            return results

        return pipe, _predict
    elif family in {"logistic", "naive"}:
        suffix_model = "LogisticRegression.pkl" if family == "logistic" else "naive_bayes.pkl"
        model_path = BASE_DIR / dataset / f"{dataset}_{suffix_model}"

        vect_candidates = [
            BASE_DIR / dataset / f"vectorizer_{'logreg' if family == 'logistic' else 'bayes'}_{dataset.lower()}.pkl",
            BASE_DIR / dataset / f"vectorizer_{'logreg' if family == 'logistic' else 'bayes'}_merged.pkl",
            BASE_DIR / dataset / f"{dataset}_vectorizer.pkl",  # fallback « ancien nom »
        ]
        try:
            vect_path = next(p for p in vect_candidates if p.exists())
        except StopIteration:
            abort(404, description=f"Vectorizer not found: tried {', '.join(str(p.name) for p in vect_candidates)}")

        if not model_path.exists():
            abort(404, description=f"Model file not found: {model_path.name}")

        clf = joblib.load(model_path)
        vectorizer = joblib.load(vect_path)

        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        def clean(txt: str) -> str:
            txt = BeautifulSoup(txt, "html.parser").get_text()
            txt = re.sub(r"http\S+|www\.\S+", " ", txt)
            txt = re.sub(rf"[{re.escape(string.punctuation)}]", " ", txt)
            txt = txt.lower()
            txt = " ".join(w for w in txt.split() if w not in stop_words)
            txt = " ".join(lemmatizer.lemmatize(w) for w in txt.split())
            return txt

        def _predict(inputs):
            if not isinstance(inputs, list):
                inputs = [inputs]

            cleaned = [clean(t) for t in inputs]
            X = vectorizer.transform(cleaned)

            preds = clf.predict(X).tolist()

            try:
                probas = clf.predict_proba(X)
                best = probas.max(axis=1)
            except AttributeError:
                best = [None] * len(preds)

            # Choose the right mapping automatically
            if len(clf.classes_) == 3:  # Yelp style 0/1/2
                mapper = map_yelp
            else:  # binary 0/1
                mapper = map_binary

            results = [
                {"result": mapper(p), "score": (float(s) if s is not None else None)}
                for p, s in zip(preds, best)
            ]
            return results

        return clf, _predict
    else:
        abort(400, description=f"Model family not supported: {family}")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=False)
    model_name = data.get("model")
    text       = data.get("text")

    if not model_name or text is None:
        abort(400, description="'model' and 'text' are required")

    _, predict_fn = load_model(model_name)
    result_dict   = predict_fn(text)[0]   # we sent a single text

    return jsonify({
        "model":  model_name,
        "input":  text,
        "result": result_dict["result"],
        "score":  result_dict["score"]
    })


@app.route("/models", methods=["GET"])
def list_models():
    out = []
    for dataset_dir in BASE_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        if (dataset_dir / f"{dataset}_bert_model").exists():
            out.append(f"{dataset.lower()}_bert")

        if (dataset_dir / f"{dataset}_LogisticRegression.pkl").exists():
            out.append(f"{dataset.lower()}_logistic")

        if (dataset_dir / f"{dataset}_naive_bayes.pkl").exists():
            out.append(f"{dataset.lower()}_naive")
    return jsonify(sorted(out))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)