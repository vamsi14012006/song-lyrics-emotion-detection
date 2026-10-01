# Lyric Mood: Song Lyrics Emotion Detection

Classifies song lyrics into five emotions (joy, sadness, anger, fear, trust) with an NLP pipeline, and serves it through a small web app.

**Stack:** Python, Pandas, NumPy, NLTK, scikit-learn (TF-IDF + Logistic Regression), Flask, HTML/CSS/JavaScript

## Run it
```bash
pip install -r requirements.txt
python app.py              # model is already trained, open http://127.0.0.1:5000
```
To rebuild from scratch:
```bash
python prepare_dataset.py  # merge raw/*.csv and create labels -> data/lyrics.csv
python train.py            # compare models, tune, evaluate, save model + metrics
python -m pytest -q tests
```

## Pipeline
1. **Data:** 6,027 songs by 20 artists (`raw/`). Duplicates, empty and very short lyrics are removed.
2. **Labels:** the dataset has no emotion column, so each song is labelled by its dominant emotion using the NRC Emotion Lexicon. Songs with weak or ambiguous signal are dropped, leaving about 2,960 labelled songs.
3. **Preprocessing (NLTK):** lowercase, strip `[Chorus]` tags and punctuation, tokenize, remove stop words and ad-libs (negations kept), lemmatize, and **remove artist names** so the model cannot identify the artist instead of the emotion.
4. **Features:** TF-IDF with 1-2 grams.
5. **Models compared:** Logistic Regression, Linear SVM, Complement NB, Multinomial NB, using 5-fold cross-validation **grouped by artist** (no artist is in both train and test folds).
6. **Final model:** Logistic Regression (C tuned with grid search), chosen because it gives class probabilities and interpretable word weights.

## Results
| Model | Macro F1 | Accuracy |
|---|---|---|
| Logistic Regression | 0.54 | 0.70 |
| Linear SVM | 0.56 | 0.71 |
| Complement NB | 0.27 | 0.57 |
| Multinomial NB | 0.14 | 0.53 |

On 6 held-out artists the tuned Logistic Regression scored 0.81 accuracy and 0.67 macro F1. About two thirds of songs are "joy", so macro F1 is the fairer number. Per-class results and the confusion matrix are in `model/metrics.json` and shown in the app.

## Limitations
- Labels come from a word list, not human listeners, so scores measure agreement with the lexicon, not true perceived emotion. `data/manual_check.csv` has 150 songs to hand-label for a real check.
- Classes are imbalanced (mostly joy); anger, fear and trust are weaker.
- Bag-of-words ignores context, sarcasm and music. A transformer model would be a natural next step.

## App features
Probability bars for every emotion, the words that drove the prediction, and a verse-by-verse emotion timeline for pasted lyrics.

## Deploy
`Procfile` is included (`gunicorn app:app`) for Render or similar hosts.
