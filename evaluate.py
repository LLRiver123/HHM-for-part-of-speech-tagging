from utils import get_word_tag, preprocess  
from hmm import create_dictionaries, initialize, viterbi_forward, viterbi_backward
import pandas as pd
from collections import defaultdict
import math
import numpy as np
from numpy import load
import pickle

file = open('vocab.pkl', 'rb')
vocab = pickle.load(file)
file.close()

with open("WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()

with open("WSJ_24.pos", 'r') as f:
    y = f.readlines()
    
A = load('A.npy')
B = load('B.npy')    

tokens = [line.split('\t')[0] for line in y]
_,prep = preprocess(vocab, tokens) 

emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
states = sorted(tag_counts.keys())

# print("--s--" in states)  # Ensure `--s--` is in `states`
# print(states)

# print(len(states))
# print(len(prep))

best_probs, best_paths = initialize(A, B, tag_counts, vocab, states, prep)
# print(best_probs.shape)
# print(A.shape)  # Should be (len(states), len(states))
# print(B.shape)
best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)
pred = viterbi_backward(best_probs, best_paths, states)
    
num_correct = 0
total = 0
for prediction, y in zip(pred, y):
    word_tag_tuple = y.split()
    if len(word_tag_tuple)!=2: 
        continue 
    word, tag = word_tag_tuple
    if prediction == tag: 
        num_correct += 1
    total += 1
print(num_correct/total)    