import pandas as pd

def combine_data(rt_data: pd.DataFrame, surprisals: pd.DataFrame, word_boundary = ''):
    rt_surprisals = []
    rt_iterator, rt_columns = rt_data.itertuples(name = None), rt_data.columns.values.tolist()
    surprisal_iterator, surprisal_columns = surprisals.itertuples(name = None), surprisals.columns.values.tolist()
    token_index = 0
    while token_index < len(surprisals.index):
        current_word, current_token = next(rt_iterator), next(surprisal_iterator)
        token_index += 1
        current_word, current_token = current_word[1:], current_token[1:] # getting rid of index column
        buffer = {column:value for value, column in zip(current_token[1:], surprisal_columns[1:])}
        if word_boundary and word_boundary in buffer['token']:
            buffer['token'] = buffer['token'][1:] # remove the word boundary character
        mismatch = buffer['token'].lower() != current_word[rt_columns.index('token')].lower()
        while mismatch:
            current_token = next(surprisal_iterator)[1:]
            token_index += 1
            buffer['token'] += current_token[surprisal_columns.index('token')]
            buffer['token_score'] += current_token[surprisal_columns.index('token_score')]
            if not buffer['oov'] and current_token[surprisal_columns.index('oov')]:
                buffer['oov'] = True
            if buffer['token'] == current_word[rt_columns.index('token')].lower():
                mismatch = False
        buffer['rt'] = current_word[rt_columns.index('RT')]
        buffer['token_uid'] = current_word[rt_columns.index('token_uid')]
        rt_surprisals.append(buffer)
        buffer = {}
    return pd.DataFrame(rt_surprisals)