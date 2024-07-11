import pandas as pd

from nltk.tokenize import sent_tokenize

rt_path = "data/processed_rts/dundee_rts.csv"

rts = pd.read_csv(rt_path)
sentences = sent_tokenize(" ".join(rts['token'].astype(str)))
with open("dundee.txt", "w") as f:
    for sentence in sentences:
        f.write(sentence + "\n")
