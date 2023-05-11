import os
import pandas as pd
import re

CORPUS_DIR="data/human_rts/dundee"

rt_data = []
averaged_times = [f for f in os.listdir(CORPUS_DIR) if f.endswith("_avg.txt")] # file name

for filename in averaged_times:
    transcript_id = re.findall("[0-9]+", filename).pop() # finds numbers in transcript
    file_lines = open(os.path.join(CORPUS_DIR, filename), 'r').readlines()
    rt_data += [{"token": line.split()[0].strip(), "RT": float(line.split()[1]), 
                "transcript_id": transcript_id} for line in file_lines]

rt_df = pd.DataFrame(rt_data)
rt_df.index.name = "token_uid"
rt_df["token"] = rt_df["token"].str.replace('[^\w\s]','')
rt_df = rt_df[rt_df['token'] != ''] # getting rid of punctuation (FOR NOW)
rt_df.to_csv("../dundee_rts.csv")