from datasets import load_dataset
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import os

ds = load_dataset("Intel/polite-guard", split="train")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

polite_texts = [r["text"] for r in ds if r["label"] in ("polite", "somewhat polite")]
nonpolite_texts = [r["text"] for r in ds if r["label"] in ("impolite", "neutral")]

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc])

lemmatized_polite = [lemmatize_text(text) for text in polite_texts]
lemmatized_nonpolite = [lemmatize_text(text) for text in nonpolite_texts]

X = lemmatized_polite + lemmatized_nonpolite
y = [1] * len(lemmatized_polite) + [0] * len(lemmatized_nonpolite)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

word_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=5, 
        max_df=0.7,
        stop_words='english',
        sublinear_tf=True
    )),
    ('classifier', MultinomialNB(alpha=0.1)) 
])

word_pipeline.fit(X_train, y_train)

y_pred = word_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["non-polite", "polite"]))

print("\nTrying different n-gram ranges for comparison:")

unigram_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 1),
        min_df=5, 
        max_df=0.7,
        stop_words='english'
    )),
    ('classifier', MultinomialNB(alpha=0.1))
])

bigram_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='word',
        ngram_range=(2, 2), 
        min_df=5, 
        max_df=0.7
    )),
    ('classifier', MultinomialNB(alpha=0.1))
])

vectorizer = word_pipeline.named_steps['vectorizer']
classifier = word_pipeline.named_steps['classifier']
feature_names = vectorizer.get_feature_names_out()

feature_importances = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

top_polite_indices = np.argsort(feature_importances)[-200:] 
top_polite_features = [(feature_names[i], feature_importances[i]) for i in top_polite_indices]
top_polite_features = sorted(top_polite_features, key=lambda x: x[1], reverse=True)

top_nonpolite_indices = np.argsort(feature_importances)[:200] 
top_nonpolite_features = [(feature_names[i], feature_importances[i]) for i in top_nonpolite_indices]
top_nonpolite_features = sorted(top_nonpolite_features, key=lambda x: x[1])

top_polite_features = [(f, s) for f, s in top_polite_features if len(f) > 2 or ' ' in f]
top_nonpolite_features = [(f, s) for f, s in top_nonpolite_features if len(f) > 2 or ' ' in f]

print("\nTop 20 polite features (words & n-grams):")
for feature, importance in top_polite_features[:20]:
    print(f"{feature}: {importance:.4f}")

print("\nTop 20 non-polite features (words & n-grams):")
for feature, importance in top_nonpolite_features[:20]:
    print(f"{feature}: {importance:.4f}")

print("\nSaving model and important features...")
os.makedirs('model', exist_ok=True)

with open('model/polite_classifier.pkl', 'wb') as f:
    pickle.dump(word_pipeline, f)
polite_features_dict = {feature: importance for feature, importance in top_polite_features}
with open('model/polite_features.pkl', 'wb') as f:
    pickle.dump(polite_features_dict, f)

print("Model and important features saved to 'model/' directory")
