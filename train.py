"""Train and evaluate the emotion classifier.

1. Remove artist names (anti-leakage)
2. Compare 4 models with 5-fold cross-validation grouped by artist
3. Tune Logistic Regression
4. Report on held-out ARTISTS never seen in training
5. Refit on all data, save model + metrics
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import preprocess

os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)

df = pd.read_csv("data/lyrics.csv").dropna(subset=["lyrics", "emotion"]).drop_duplicates(subset="lyrics")

# 1. artist names -> stop words
extra = {"slim", "shady", "marshall", "mathers", "yonce", "bey", "bieber", "swift", "taylor", "nicki", "minaj", "cardi"}
for name in set(df["Artist"].astype(str)) | set(df["Source"].astype(str)):
    extra |= {w for w in __import__("re").findall(r"[a-z]+", name.lower()) if len(w) > 2}
json.dump(sorted(extra), open(preprocess.TOKENS_PATH, "w"))
preprocess.set_extra_stop(extra)

df["clean"] = df["lyrics"].apply(preprocess.clean_lyrics)
df = df[df["clean"].str.len() > 0].reset_index(drop=True)
groups = df["Source"]
print(f"{len(df)} songs, {groups.nunique()} artists\n{df['emotion'].value_counts()}\n")


def make(clf):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True, min_df=2)), ("clf", clf)])


# 2. model comparison (cross-validation, folds never share an artist)
cv = GroupKFold(n_splits=5)
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
    "Complement NB": ComplementNB(),
    "Multinomial NB": MultinomialNB(),
}
cv_results = {}
for name, clf in candidates.items():
    f1 = cross_val_score(make(clf), df["clean"], df["emotion"], groups=groups, cv=cv, scoring="f1_macro")
    acc = cross_val_score(make(clf), df["clean"], df["emotion"], groups=groups, cv=cv, scoring="accuracy")
    cv_results[name] = {"macro_f1": round(f1.mean(), 3), "accuracy": round(acc.mean(), 3)}
    print(f"{name:20s} macro-F1 {f1.mean():.3f}  accuracy {acc.mean():.3f}")

# 3. tune logistic regression
grid = GridSearchCV(make(LogisticRegression(max_iter=1000, class_weight="balanced")),
                    {"clf__C": [0.5, 1, 2, 5, 10]}, cv=cv, scoring="f1_macro")
grid.fit(df["clean"], df["emotion"], groups=groups)
best = grid.best_params_
print("\nBest LR params:", best, "CV macro-F1:", round(grid.best_score_, 3))

# 4. held-out artists
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
tr, te = next(gss.split(df, groups=groups))
held = sorted(groups.iloc[te].unique())
model = make(LogisticRegression(max_iter=1000, class_weight="balanced", C=best["clf__C"]))
model.fit(df["clean"].iloc[tr], df["emotion"].iloc[tr])
pred = model.predict(df["clean"].iloc[te])
y_te = df["emotion"].iloc[te]
print("\nHeld-out artists:", held)
print(classification_report(y_te, pred, zero_division=0))
labels = list(model.classes_)
cm = confusion_matrix(y_te, pred, labels=labels)
report = classification_report(y_te, pred, zero_division=0, output_dict=True)

cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5, 4.2), dpi=130)
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=30); ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Held-out artists (row-normalised)")
fig.tight_layout(); fig.savefig("static/confusion_matrix.png"); plt.close(fig)

# 5. final model on all data
final = make(LogisticRegression(max_iter=1000, class_weight="balanced", C=best["clf__C"]))
final.fit(df["clean"], df["emotion"])
vocab = np.array(final["tfidf"].get_feature_names_out())
top_words = {l: vocab[np.argsort(final["clf"].coef_[i])[-8:][::-1]].tolist() for i, l in enumerate(final.classes_)}
for l, w in top_words.items():
    print(f"{l:>8}: {', '.join(w)}")

joblib.dump(final, "model/emotion_model.joblib")
json.dump({
    "n_songs": int(len(df)), "n_artists": int(groups.nunique()),
    "class_counts": df["emotion"].value_counts().to_dict(),
    "cv_results": cv_results, "best_params": best, "heldout_artists": held,
    "heldout": {"accuracy": round(report["accuracy"], 3), "macro_f1": round(report["macro avg"]["f1-score"], 3),
                "per_class": {l: round(report[l]["f1-score"], 3) for l in labels}},
    "top_words": top_words,
}, open("model/metrics.json", "w"), indent=2)
print("\nSaved model/emotion_model.joblib and model/metrics.json")
