import re
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

ds = load_dataset("Intel/polite-guard", split="train")

polite_texts   = [r["text"] for r in ds if r["label"] in ("polite", "somewhat polite")]
nonpolite_texts = [r["text"] for r in ds if r["label"] in ("neutral", "impolite")]

all_texts = polite_texts + nonpolite_texts
all_labels = [1] * len(polite_texts) + [0] * len(nonpolite_texts)
y = np.array(all_labels)

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    all_texts, y, test_size=0.2, stratify=y, random_state=42
)

pipe_tfidf_lr = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        strip_accents="unicode",
        token_pattern=r"\b\w+\b",
        smooth_idf=True,
        sublinear_tf=True
    )),
    ("lr", LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=1000
    ))
])

pipe_tfidf_lr.fit(X_train_texts, y_train)

y_pred = pipe_tfidf_lr.predict(X_test_texts)
y_proba = pipe_tfidf_lr.predict_proba(X_test_texts)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

cv = CountVectorizer(ngram_range=(1,3), min_df=5, strip_accents="unicode",
                     token_pattern=r"\b\w+\b", stop_words=None)

Xp = cv.fit_transform(polite_texts) 
Xn = cv.transform(nonpolite_texts)      
vocab = np.array(cv.get_feature_names_out())

alpha = 1
p = Xp.sum(axis=0).A1 + alpha  
n = Xn.sum(axis=0).A1 + alpha     
pp, nn = p.sum(), n.sum()
print(pp)

log_odds = np.log(p/pp) - np.log(n/nn)

top_k = 250
best = vocab[np.argsort(-log_odds)[:top_k]]
print(best)

tokens_sorted = sorted(best, key=lambda s: len(s.split()), reverse=True)
escaped = [re.escape(t) for t in tokens_sorted]
pattern = r"\b(?:{})\b".format("|".join(escaped))
regex = re.compile(pattern, flags=re.I)

joblib.dump(
    {
        "pipeline_tfidf_lr": pipe_tfidf_lr,
        "count_vectorizer": cv,
        "log_odds": log_odds.astype(np.float32),
        "regex_pattern": pattern
    },
    "politeness_model.joblib",
    compress=3
)