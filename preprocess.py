"""Text cleaning shared by training and prediction."""
import json
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

# Keep negations: "not happy" must differ from "happy".
STOP = set(stopwords.words("english")) - {"no", "not", "nor", "never"}
# Ad-libs and filler that carry no emotion
STOP |= {"yeah", "ooh", "oh", "uh", "na", "la", "hey", "whoa", "ayy", "ah", "mmm", "woo", "huh", "yo", "gonna", "wanna"}
LEMMA = WordNetLemmatizer()
TOKENS_PATH = os.path.join(os.path.dirname(__file__), "model", "artist_tokens.json")


def set_extra_stop(words):
    STOP.update(words)


# Artist names are removed so the model cannot "recognise" an artist instead of an emotion.
if os.path.exists(TOKENS_PATH):
    set_extra_stop(json.load(open(TOKENS_PATH)))


def clean_lyrics(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)          # [Chorus], [Verse 1]
    text = re.sub(r"[^a-z\s']", " ", text)        # punctuation, digits
    tokens = word_tokenize(text)
    tokens = [LEMMA.lemmatize(t) for t in tokens if t not in STOP and len(t) > 2]
    return " ".join(tokens)
