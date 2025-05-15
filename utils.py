import pickle
import spacy

POLITE_FEATURES_DIC = None
NLP = None

def load_resources():
    with open('model/polite_features.pkl', 'rb') as f:
        polite_features_dict = pickle.load(f)
    NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    return polite_features_dict, NLP

def lemmatize_text(text):
    doc = NLP(text)
    return " ".join([token.lemma_.lower() for token in doc])