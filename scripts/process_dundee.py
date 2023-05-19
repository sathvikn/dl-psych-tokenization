import os
import pandas as pd
import re
import string

CORPUS_DIR="data/human_rts/dundee"

rt_data = []
averaged_times = [f for f in os.listdir(CORPUS_DIR) if f.endswith("_avg.txt")] # file name

for filename in averaged_times:
    transcript_id = re.findall("[0-9]+", filename).pop() # finds numbers in transcript
    file_lines = open(os.path.join(CORPUS_DIR, filename), 'r').readlines()
    for line in file_lines:
        token, rt = line.split()
        first, last = token[0], token[-1]
        rt_data.append({"token": token, "RT": float(rt), "following_punct": first in string.punctuation, 
                    "preceding_punct": last in string.punctuation,
                    "is_punct_non_alpha": not token.isalpha(),
                    "transcript_id": transcript_id})

rt_df = pd.DataFrame(rt_data)
rt_df.index.name = "token_uid"
# TODO document the function
exclusion_index = lambda df, bool_column, value: df[df[bool_column]].index.union(df[df[bool_column]].index + value)
exclude_token_preceding_punct = exclusion_index(rt_df, 'preceding_punct', 1)
exclude_token_following_punct = exclusion_index(rt_df, 'following_punct', -1)
exclude_token_after_non_alpha = exclusion_index(rt_df, 'is_punct_non_alpha', 1)
exclude = exclude_token_preceding_punct.union(exclude_token_following_punct).union(exclude_token_after_non_alpha)
without_punct = rt_df["token"].str.replace('[^\w\s]','') # getting rid of punctuation
exclude = exclude.union(without_punct[without_punct == ''].index.union(without_punct[without_punct == ''].index - 1)) # token after would have been covered by non_alpha case
exclusion_column = pd.Series(0, range(len(rt_df.index)))
for i in range(len(exclusion_column)):
    if i in exclude.values:
        exclusion_column.at[i] = 1
rt_df['exclude'] = exclusion_column
rt_df = rt_df[rt_df['token'] != '']
rt_df.to_csv("dundee_rts_v1.csv")