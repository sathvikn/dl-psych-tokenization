import random

ratio_threshold = 0.2
length = 8
total = 20
orthographic_ids=[]
coca_path="corpora/public_coca_orthographic.txt"

sentences = open(coca_path).readlines()

for i in range(len(sentences)):
    curr_sentence = sentences[i]
    words = curr_sentence.split()
    above_length_threshold = 0
    for word in words:
        if len(word) > length:
            above_length_threshold += 1

    long_word_ratio = float(above_length_threshold) / len(words)
    if long_word_ratio >= ratio_threshold and len(words) > 10:
        orthographic_ids.append(i)

index_sample = random.sample(orthographic_ids, 20)

bpe_sentences = open("corpora/public_coca_bpe.txt").readlines()
morph_sentences = open("corpora/public_coca_transducer.txt").readlines()

for i in index_sample:
    print(bpe_sentences[i])
    print(morph_sentences[i])

