from gensim.models import Word2Vec
import json
import logging
import numpy as np
import pandas as pd
import sys
from multiprocessing import cpu_count
import os

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

W2V_TXT_FILE="_word2vec.txt"
cpus = cpu_count()
word2vec_params = {
    'sg': 1,  # 0 ： CBOW； 1 : skip-gram
    "vector_size": 300,
    "alpha": 0.01,
    "min_alpha": 0.0005,
    'window': 10,
    'min_count': 1,
    'seed': 1,
    "workers": cpus,
    "negative": 0,
    "hs": 1,  # 0: negative sampling, 1:hierarchical  softmax
    'compute_loss': True,
    'epochs': 50,
    'cbow_mean': 0,
}
dataset_path=None
sep=None
args = sys.argv[1:]
print(args)
if len(args)==1:
    dataset_path=args[0]
    sep = None

if len(args)==2:
    dataset_path=args[0]
    sep = args[1]


print('dataset_path',dataset_path)
print('sep',sep)
df=pd.read_csv(dataset_path,sep=sep)
print(df)
sentences = list(df.text)
#sentences=sentences[:5]
print(sentences[:5])
print(len(sentences))

model = Word2Vec(**word2vec_params)
model.build_vocab(sentences)
trained_word_count, raw_word_count = model.train(sentences, compute_loss=True,
                                                 total_examples=model.corpus_count,
                                                 epochs=model.epochs)

#model_name = 'skipgram_model_gensim.pk'
#model.save(model_name)


model.wv.save_word2vec_format(os.path.split(dataset_path)[1].split('.')[0]+W2V_TXT_FILE)

