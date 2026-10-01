"""Merge the artist CSVs in raw/ and create emotion labels -> data/lyrics.csv

The dataset has NO emotion labels, so labels are generated with the NRC
Emotion Lexicon (pip install NRCLex): count how many words in a song belong to each
emotion and keep the dominant one. Songs with weak or ambiguous signal are dropped.
"""
import glob
import json
import os
import re
from collections import Counter

import nrclex
import pandas as pd

EMOTIONS = ["joy", "sadness", "anger", "fear", "trust"]
MIN_HITS = 8        # minimum emotion-word hits in a song
MIN_LEAD = 1.25     # top emotion must beat the runner-up by 25%

lex_path = os.path.join(os.path.dirname(nrclex.__file__), "data", "nrc_en.json")
LEX = json.load(open(lex_path, encoding="utf-8"))   # word -> [emotions]


def label(text):
    c = Counter()
    for w in re.findall(r"[a-z']+", text.lower()):
        for e in LEX.get(w, []):
            if e in EMOTIONS:
                c[e] += 1
    if sum(c.values()) < MIN_HITS:
        return None
    top = c.most_common(2)
    if len(top) > 1 and top[0][1] < MIN_LEAD * top[1][1]:
        return None
    return top[0][0]


frames = []
for f in glob.glob("raw/*.csv"):
    d = pd.read_csv(f)
    d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])   # some files have an index column, some do not
    d["Artist"] = d["Artist"].fillna(os.path.splitext(os.path.basename(f))[0])
    d["Source"] = os.path.splitext(os.path.basename(f))[0]   # used to split train/test by artist
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=["Lyric"]).drop_duplicates(subset=["Artist", "Title"])
df = df[df["Lyric"].str.split().str.len() >= 30]
df["emotion"] = df["Lyric"].apply(label)
out = df.dropna(subset=["emotion"]).rename(columns={"Lyric": "lyrics"})
out[["Source", "Artist", "Title", "lyrics", "emotion"]].to_csv("data/lyrics.csv", index=False)

print(f"{len(df)} usable songs -> {len(out)} labelled")
print(out["emotion"].value_counts())
# 150 random songs to hand-check the automatic labels
out.sample(min(150, len(out)), random_state=1)[["Artist", "Title", "emotion"]].assign(my_label="") \
    .to_csv("data/manual_check.csv", index=False)
