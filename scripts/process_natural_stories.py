import pandas as pd
import string
import pdb

CORPUS_PATH="data/human_rts/natural-stories/natural-stories.csv"

rt_data = pd.read_csv(CORPUS_PATH)
rt_data.columns = ['index', 'token_uid', 'token', 'RT']
rt_data['transcript_id'] = [1] * len(rt_data.index) # the surprisal computation code needs a transcript ID column

def exclude_token(df, index):
    # exclude the token if it precedes or follows punctuation, or if it is not alphabetic
    token = df.iloc[index]['token']
    if token[0] in string.punctuation or token[-1] in string.punctuation or not token.isalpha():
        return 1
    if index != 0:
        prev_token = df.iloc[index - 1]['token']
        # exclude this token if the previous token ended w/a punctuation mark or if it's non alphabetic
        if prev_token[-1] in string.punctuation or not prev_token.isalpha():
            return 1
    if index != len(df.index - 1):
        next_token = df.iloc[index + 1]['token']
        # exclude this token if the next token starts w/a punctuation mark or isn't alphabetic
        if next_token[0] in string.punctuation or not next_token.isalpha():
            return 1
    return 0
rt_data['exclude_rt'] = [exclude_token(rt_data, i) for i in range(len(rt_data.index))]
rt_data.to_csv("natural_stories_rts.csv", index = False)
