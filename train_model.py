import re
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

ds = load_dataset("Intel/polite-guard", split="train")

polite_texts   = [r["text"] for r in ds if r["label"] in ("polite", "somewhat polite")]
nonpolite_texts = [r["text"] for r in ds if r["label"] in ("neutral", "impolite")]

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
        "vectorizer": cv,
        "log_odds": log_odds.astype(np.float32), 
        "regex_pat": pattern,
    },
    "politeness_model.joblib",
    compress=3, 
)