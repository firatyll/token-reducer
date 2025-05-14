import pickle
import os
import spacy

def load_resources():
    if not os.path.exists('model/polite_features.pkl'):
        print("Error: Model files not found. Please run main.py first to train the model.")
        
    with open('model/polite_features.pkl', 'rb') as f:
        polite_features_dict = pickle.load(f)
    
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Error: spaCy model 'en_core_web_sm' not found. Please install it with:")
    
    return polite_features_dict, nlp

def lemmatize_text(text, nlp):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc])

def remove_polite_features(text, polite_features_dict, nlp, threshold):
    doc = nlp(text)
    
    tokens = []
    lemmas = []
    for token in doc:
        tokens.append(token.text)
        lemmas.append(token.lemma_.lower())
    
    token_flags = [False] * len(tokens)
    removed_features = []
    
    for i, lemma in enumerate(lemmas):
        if lemma in polite_features_dict and polite_features_dict[lemma] > threshold:
            token_flags[i] = True
            removed_features.append(tokens[i])
    
    for n in range(2, 4): 
        for i in range(len(lemmas) - n + 1):
            lemma_ngram = ' '.join(lemmas[i:i+n])
            
            if lemma_ngram in polite_features_dict and polite_features_dict[lemma_ngram] > threshold:
                for j in range(i, min(i+n, len(token_flags))):
                    token_flags[j] = True
                
                original_ngram = ' '.join(tokens[i:i+n])
                removed_features.append(original_ngram)
    
    for feature, score in polite_features_dict.items():
        if score <= threshold:
            continue
            
        if ' ' not in feature:
            for i, lemma in enumerate(lemmas):
                if (feature in lemma or lemma in feature) and lemma != feature and not token_flags[i]:
                    token_flags[i] = True
                    removed_features.append(tokens[i])
        
        elif ' ' in feature:
            feature_parts = feature.split()
            for i in range(len(lemmas) - len(feature_parts) + 1):
                match_count = 0
                for j, part in enumerate(feature_parts):
                    if i+j < len(lemmas) and (part in lemmas[i+j] or lemmas[i+j] in part):
                        match_count += 1
                
                if match_count >= len(feature_parts) * 0.7:
                    for j in range(len(feature_parts)):
                        if i+j < len(tokens):
                            token_flags[i+j] = True
                    
                    matched_text = ' '.join(tokens[i:i+len(feature_parts)])
                    removed_features.append(matched_text)
    
    result_tokens = [tokens[i] for i in range(len(tokens)) if not token_flags[i]]
    result_text = ' '.join(result_tokens)
    
    return result_text, list(set(removed_features))