import numpy as np
import pandas as pd

from typing import Dict, List


def align_surprisal(rt_data: pd.DataFrame, surprisals: pd.DataFrame, model_config: Dict, corpus_name: str, use_lookup: bool = False):
    rt_surprisals = []
    lookup_table = pd.DataFrame()
    lookup_iterator = iter([])
    word_boundary = model_config['word_boundary']
    if use_lookup:
        lookup_table = read_lookup_table(word_boundary, model_config['lookup_tbl'][corpus_name])
        lookup_iterator = lookup_table.itertuples(name = None)
        assert len(lookup_table.index) == len(rt_data.index)
    rt_iterator, rt_columns = rt_data.itertuples(name = None), rt_data.columns.values.tolist()
    surprisal_iterator, surprisal_columns = surprisals.itertuples(name = None), surprisals.columns.values.tolist()
    token_index = 0
    while token_index < len(surprisals.index):
        current_word, current_token = next(rt_iterator), next(surprisal_iterator)
        token_index += 1
        current_word, current_token = current_word[1:], current_token[1:] # getting rid of index column
        buffer = {column:value for value, column in zip(current_token[1:], surprisal_columns[1:])}
        buffer['num_tokens'] = 1 # for later analysis
        if word_boundary and word_boundary in buffer['token']:
            buffer['token'] = buffer['token'][1:].lower() # remove the word boundary character
        ref = current_word[rt_columns.index('token')].lower()
        if use_lookup:
            ref = next(lookup_iterator)[2].lower()
        mismatch = buffer['token'].lower() != ref
        while mismatch:
            current_token = next(surprisal_iterator)[1:]
            token_index += 1
            buffer['num_tokens'] += 1
            if use_lookup:
                buffer['token'] += " "
            buffer['token'] += current_token[surprisal_columns.index('token')]
            buffer['surprisal'] += current_token[surprisal_columns.index('surprisal')]
            if not buffer['oov'] and current_token[surprisal_columns.index('oov')]:
                # mark the word as OOV if this is a case for ANY token in the input 
                buffer['oov'] = True
            if buffer['token'] == ref:
                mismatch = False
        buffer['rt'] = current_word[rt_columns.index('RT')]
        buffer['token_uid'] = current_word[rt_columns.index('token_uid')]
        buffer['exclude_rt'] = current_word[rt_columns.index('exclude')]
        if use_lookup:
            # assign it to the word, since its value in the buffer is the morphological tokenization
            buffer['token'] = current_word[rt_columns.index('token')].lower()
        rt_surprisals.append(buffer)
        buffer = {}
    return pd.DataFrame(rt_surprisals)

def word_length(rt_data: pd.DataFrame, token_col: str):
    return rt_data.apply(lambda row: len(row[token_col]), axis = 1)

def join_log_freq(filepath: str, rt_data: pd.DataFrame):
    freq_table = pd.read_table(filepath, delim_whitespace=True, names=('prob', 'word', 'backoff_weight'))
    rt_data = rt_data.merge(freq_table[['prob', 'word']], how = 'left', left_on = 'token', right_on = 'word')
    rt_data.rename(columns={'prob': 'log_freq'}, inplace = True)
    return rt_data

def prev_token_predictors(rt_data: pd.DataFrame, num_tokens: int):
    for i in range(1, num_tokens + 1):
        rt_data[f'prev_freq_{str(i)}'] = rt_data['log_freq'].shift(i)
        rt_data[f'prev_len_{str(i)}'] = rt_data['word_length'].shift(i)
        rt_data[f'prev_surprisal_{str(i)}'] = rt_data['surprisal'].shift(i)
        rt_data[f'prev_num_tokens_{str(i)}'] = rt_data['surprisal'].shift(i)
    return rt_data

def generate_predictors(rt_data: pd.DataFrame, prev_tokens: int):
    rt_data['word_length'] = word_length(rt_data, 'token')
    rt_data = join_log_freq('data/word_freqs.txt', rt_data)
    rt_data = prev_token_predictors(rt_data, prev_tokens)
    return rt_data.dropna()

def read_lookup_table(word_boundary: str, table_path: str):
    tokenizer_lookup = pd.read_table(table_path, delimiter = "\t", names = ["token", "morph_tokenization"])
    separate_tokens_by_space = lambda word: " ".join(word.split(word_boundary))
    tokenizer_lookup['morph_tokenization'] = tokenizer_lookup['morph_tokenization'].apply(separate_tokens_by_space)
    return tokenizer_lookup

def process_surprisal_data(rts: pd.DataFrame, surprisal_path: str, config: Dict, corpus_name: str, prev_tokens = 1, use_lookup = False):
    surprisal = pd.read_csv(surprisal_path)
    combined_data = align_surprisal(rts, surprisal, config, corpus_name, use_lookup)
    combined_data = generate_predictors(combined_data, prev_tokens)
    combined_data = combined_data[(~combined_data['oov']) & (~combined_data['exclude_rt'])]
    return combined_data

def preprocess_rt_data(path: str):
    rts = pd.read_csv(path)
    rts["token"] = rts["token"].str.replace('[^\w\s]','')
    rts['token'].replace('', np.nan, inplace=True)
    rts = rts.dropna()
    return rts

def combine_corpus_data(data: List[pd.DataFrame], tok_schemes: List[str], common_indices: pd.Series):
    combined_data = []
    for surprisal_rts, col_name in zip(data, tok_schemes):
        surprisal_rts = surprisal_rts[surprisal_rts.index.isin(common_indices)]
        surprisal_rts['tokenization'] = col_name
        combined_data.append(surprisal_rts)
    return pd.concat(combined_data)

def extract_one_sentence(df, sentence_id, corpus_name, transcript_id):
    return df[(df['sentence_id'] == sentence_id) &
     (df['corpus'] == corpus_name) & (df['transcript_id'] == transcript_id)]

def generate_token_counts(column: pd.Series, names: List[str]):
    # for the num_tokens column - make a table of how many tokens each word got split into
    token_counts = pd.DataFrame(column.value_counts()).reset_index()
    token_counts.columns = names
    return token_counts