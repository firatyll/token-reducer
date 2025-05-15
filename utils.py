import pickle
import spacy

POLITE_FEATURES_DIC = None
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def load_resources():
    with open('model/polite_features.pkl', 'rb') as f:
        polite_features_dict = pickle.load(f)
    nlp = NLP
    
    return polite_features_dict, nlp

def lemmatize_text(text):
    doc = NLP(text)
    return " ".join([token.lemma_.lower() for token in doc])