from numpy import load
import argparse
import pickle
from nltk import word_tokenize
from hmm import  create_dictionaries, initialize, viterbi_forward, viterbi_backward
from utils import processing
from hmm import training_data

corpus_path = "WSJ_02-21.pos"
alpha = 0.001

def parse_argument():
    parser = argparse.ArgumentParser(description='Predict Part of Speech Tags')
    parser.add_argument('--sent', help='Enter your sentence.')
    return parser.parse_args()

def predict():
    args = parse_argument()
    sample = args.sent
    sample = str(sample) + ' #'
    tokens = word_tokenize(sample)
  
    file = open('vocab.pkl', 'rb')
    vocab2idx = pickle.load(file)
    file.close()
    
    prep_tokens = processing(vocab2idx, tokens)
    training_corpus = training_data(corpus_path)
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    
    A = load('A.npy')
    B = load('B.npy')
    best_probs, best_paths = initialize(A, B, tag_counts, vocab2idx, states, prep_tokens)
    best_probs, best_paths = viterbi_forward(A, B, prep_tokens, best_probs, best_paths, vocab2idx)
    pred = viterbi_backward(best_probs, best_paths, states)

    res = []
    for tok, tag in zip(prep_tokens[:-1], pred[:-1]):
        res.append((tok, tag))
    print(res)
if __name__ == "__main__":
    predict()