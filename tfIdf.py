import joblib
from strip_polite import strip_polite

bundle = joblib.load("politeness_model.joblib")
pipe   = bundle["pipeline_tfidf_lr"]
THRESHOLD = 0.3

def tfIdf(sentence : str) -> str:
    texts = [sentence]
    probas = pipe.predict_proba(texts)[:,1]

    if probas[0] >= THRESHOLD:
        cleaned = strip_polite(sentence)
        return cleaned , probas
    
    return sentence , probas[0]