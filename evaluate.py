import joblib
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, auc,
                             confusion_matrix)
import matplotlib.pyplot as plt

ds = load_dataset("Intel/polite-guard", split="train")
polite_texts = [r["text"] for r in ds if r["label"] in ("polite", "somewhat polite")]
nonpolite_texts = [r["text"] for r in ds if r["label"] in ("neutral", "impolite")]
all_texts = polite_texts + nonpolite_texts
all_labels = [1] * len(polite_texts) + [0] * len(nonpolite_texts)

y = np.array(all_labels)
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    all_texts, y, test_size=0.2, stratify=y, random_state=42
)

data = joblib.load("politeness_models.joblib")
pipe = data["pipeline_tfidf_lr"]
cv = data["count_vectorizer"]
log_odds = data["log_odds"]
regex_pattern = data["regex_pattern"]

y_pred_pipe = pipe.predict(X_test_texts)
y_proba_pipe = pipe.predict_proba(X_test_texts)[:, 1]

def logodds_score(texts, cv, log_odds_vector):
    X = cv.transform(texts)
    scores = X.toarray().dot(log_odds_vector)
    return scores 

scores_ld = logodds_score(X_test_texts, cv, log_odds)
y_pred_ld = (scores_ld > 0).astype(int)
y_proba_ld = 1 / (1 + np.exp(-scores_ld))

print("=== Pipeline (TF-IDF + LR) Performance ===")
print(classification_report(y_test, y_pred_pipe))
print("ROC AUC:", roc_auc_score(y_test, y_proba_pipe))

print("=== Log-Odds Classifier Performance ===")
print(classification_report(y_test, y_pred_ld))
print("ROC AUC:", roc_auc_score(y_test, y_proba_ld))

fpr_pipe, tpr_pipe, _ = roc_curve(y_test, y_proba_pipe)
fpr_ld, tpr_ld, _ = roc_curve(y_test, y_proba_ld)
plt.figure()
plt.plot(fpr_pipe, tpr_pipe, label=f"Pipe AUC={auc(fpr_pipe, tpr_pipe):.2f}")
plt.plot(fpr_ld, tpr_ld, label=f"LogOdds AUC={auc(fpr_ld, tpr_ld):.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.show()

precision_pipe, recall_pipe, _ = precision_recall_curve(y_test, y_proba_pipe)
precision_ld, recall_ld, _ = precision_recall_curve(y_test, y_proba_ld)
plt.figure()
plt.plot(recall_pipe, precision_pipe, label=f"Pipe AP={auc(recall_pipe, precision_pipe):.2f}")
plt.plot(recall_ld, precision_ld, label=f"LogOdds AP={auc(recall_ld, precision_ld):.2f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.tight_layout()
plt.show()

def plot_confusion(cm, title):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center")
    plt.tight_layout()

cm_pipe = confusion_matrix(y_test, y_pred_pipe)
cm_ld = confusion_matrix(y_test, y_pred_ld)
plot_confusion(cm_pipe, "Confusion Matrix: Pipeline")
plot_confusion(cm_ld, "Confusion Matrix: Log-Odds")
plt.show()
