import argparse
import json
import os
from typing import List
from tqdm import tqdm

import torch

from trans import transducer
from trans import vocabulary
from trans import sed
from trans import optimal_expert_substitutions
from trans import utils
from trans.train import decode

def tokenize_sentence(model: transducer, vocab: vocabulary, sentence: str, wb_char: str):
    # the results contain the original token and the tab-separated output from the transducer 
    dataloader = prep_data([sentence], vocab, batch_size = 5)
    results = predict(model, dataloader)
    tokens = [insert_wb_char(token.split("\t")[1], wb_char) for token in results]
    return tokens, results
    
def tokenize_corpus(model_dir: str, sentence_list: List[str], wb_char: str):
    sentence_list = sentence_list
    model, vocab = load_model_and_vocab(model_dir)
    dataloader = prep_data(sentence_list, vocab, batch_size = 1000)
    results = predict(model, dataloader)
    return output_as_sentences(results, sentence_list, wb_char)

def load_model_and_vocab(model_dir:str):
    model_path = os.path.join(model_dir, "best.model")
    config_path = os.path.join(model_dir, "config.json")
    vocab_path = os.path.join(model_dir, "vocabulary.pkl")
    stochastic_edit_distance_params = os.path.join(model_dir, "sed.pkl")
    vocab = vocabulary.Vocabularies.from_pickle(vocab_path)

    stochastic_edit_distance_aligner = sed.StochasticEditDistance.from_pickle(stochastic_edit_distance_params)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(stochastic_edit_distance_aligner)

    with open(config_path, "r") as c:
            model_config: dict = json.load(c)
    args_dict = {}
    for key, value in model_config.items():
        args_dict[key.replace("-", "_")] = value
    args_dict['device'] = 'cpu'
    #print("loading model")
    model = transducer.Transducer(vocab, expert, argparse.Namespace(**args_dict))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, vocab

def prep_data(sentence_list: List[str], vocab: vocabulary.Vocabularies, batch_size: int):
    # the transducer accepts word-by-word (or token-by-token)
    test_data = utils.Dataset()
    #print("reading sentences into Dataloader")
    for sentence in sentence_list:
        sentence_tokens = sentence.strip().split()
        for token in sentence_tokens:
            encoded_input = torch.tensor(vocab.encode_unseen_input(token))
            sample = utils.Sample(token, None, encoded_input) # the target token goes where None should be, this isn't necessary since we're just running inference
            test_data.add_samples(sample)
    return test_data.get_data_loader(batch_size=batch_size)   

def predict(model: transducer, data_loader: torch.utils.data.DataLoader):
    results = []
    #print("making predictions")
    with torch.no_grad():
        predictions = decode(model, data_loader).predictions
        results += predictions
    return results

def output_as_sentences(decoder_output: List[str], corpus_sentences: List[str], wb_char: str):
    token_sentences = []
    start_index = 0
    words_per_sentence = [len(s.split(" ")) for s in corpus_sentences]
    #print("combining word-level predictions into sentences")
    for i in range(len(words_per_sentence)):
        end_index = start_index + words_per_sentence[i]
        processed_decoder_output = [insert_wb_char(token.split("\t")[1], wb_char) for token in decoder_output[start_index:end_index]]
        token_sentences.append(" ".join(processed_decoder_output))
        start_index += words_per_sentence[i]
    assert len(token_sentences) == len(corpus_sentences)
    return token_sentences

def insert_wb_char(word: str, wb_character: str):
    # Since we are training LMs on tokens, we should add word boundary characters (like GPT2)
    # this means kenLM is predicting individual tokens, separated by spaces
    if wb_character in word:
        word = word.split(wb_character)
        word = " ".join(word)
    return wb_character + word
