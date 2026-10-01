"""Creates a tiny synthetic dataset so the pipeline runs end to end.
Replace data/lyrics.csv with a real labelled lyrics dataset (e.g. from Kaggle)."""
import random, pandas as pd
random.seed(1)
bank = {
 "joy":     "dance sunshine smile laugh happy bright celebrate love golden morning sing together alive wonderful".split(),
 "sadness": "cry tears alone lonely goodbye rain broken empty miss gone heartache dark lost fade".split(),
 "anger":   "hate fight scream burn rage fire war lies betray hell blood destroy shout break".split(),
 "love":    "kiss darling heart hold forever touch baby tender embrace yours sweet devotion romance close".split(),
 "fear":    "afraid shadow run scared nightmare trembling hide danger whisper panic cold dread haunt escape".split(),
}
filler = "the and you me i in my of to a on we it".split()
rows = []
for emo, words in bank.items():
    for _ in range(120):
        toks = random.choices(words, k=14) + random.choices(filler, k=10)
        random.shuffle(toks); rows.append((" ".join(toks), emo))
pd.DataFrame(rows, columns=["lyrics", "emotion"]).to_csv("data/lyrics.csv", index=False)
print("wrote data/lyrics.csv")
