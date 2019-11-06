import torch
from essential_generators import DocumentGenerator
from experiment_impact_tracker.compute_tracker import ImpactTracker
from experiment_impact_tracker.utils import get_flop_count_tensorflow
import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', help="conv.wmt14.en-fr, transformer.wmt14.en-fr")
parser.add_argument('seed', type=int)
parser.add_argument('min_words', type=int)
parser.add_argument('max_words', type=int)
parser.add_argument('log_dir', type=str)

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]
en2de = torch.hub.load('pytorch/fairseq', args.model, tokenizer='moses', bpe='subword_nmt')

tracker = ImpactTracker(os.path.join(args.log_dir, "{}_seed{}_min{}_max{}".format(args.model, args.seed, args.min_words, args.max_words)))
tracker.launch_impact_monitor()

# Translate a sentence
gen = DocumentGenerator()

cuda_model = en2de.cuda()

for i in range(1000):
    sentence = gen.gen_sentence(min_words=args.min_words, max_words=args.max_words)
    translation = cuda_model.translate(sentence)
    print("{} -> {}".format(sentence, translation))

