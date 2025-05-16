import os
import spacy
import pickle
import numpy as np
from utils import lemmatize_text
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

intel_ds = load_dataset("Intel/polite-guard", split="train")
wiki_ds = load_dataset("JaehyungKim/p2c_polite_wiki", split="train")
p_corpus = load_dataset("frfede/politeness-corpus", split="train")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

polite_texts = [r["text"] for r in intel_ds if r["label"] in ("polite", "somewhat polite")]
nonpolite_texts = [r["text"] for r in intel_ds if r["label"] in ("impolite", "neutral")]
wiki_polite = [r["sentence"] for r in wiki_ds if r["label"] == 1]
wiki_nonpolite = [r["sentence"] for r in wiki_ds if r["label"] == 0]
pc_polite     = [r["text"] for r in p_corpus if r["label"] == 2]
pc_nonpolite  = [r["text"] for r in p_corpus if r["label"] != 2]
polite_texts += wiki_polite

nonpolite_texts += wiki_nonpolite
polite_texts    += pc_polite
nonpolite_texts += pc_nonpolite

lemmatized_polite = [lemmatize_text(text) for text in polite_texts]
lemmatized_nonpolite = [lemmatize_text(text) for text in nonpolite_texts]

X = lemmatized_polite + lemmatized_nonpolite
y = [1] * len(lemmatized_polite) + [0] * len(lemmatized_nonpolite)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

word_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        sublinear_tf=True
    )),
    ('classifier', MultinomialNB()) 
])

param_grid = {
    'vectorizer__ngram_range' : [(1,1), (1,2), (1,3)],
    'vectorizer__min_df'      : [3, 5],
    'vectorizer__max_df'      : [0.7, 0.9],
    'classifier__alpha'       : [0.01, 0.1, 0.5]
}

grid = GridSearchCV(
    estimator   = word_pipeline,
    param_grid  = param_grid,
    scoring     = 'f1_macro', 
    cv          = 3,
    n_jobs      = -1,
    verbose     = 2
)

grid.fit(X_train, y_train)

print("Best hyper-parameters:")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["non-polite", "polite"]))

vectorizer = best_model.named_steps['vectorizer']
classifier = best_model.named_steps['classifier']
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
    pickle.dump(best_model, f)
polite_features_dict = {feature: importance for feature, importance in top_polite_features}
with open('model/polite_features.pkl', 'wb') as f:
    pickle.dump(polite_features_dict, f)

print("Model and important features saved to 'model/' directory")
