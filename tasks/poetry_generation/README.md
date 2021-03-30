## Build word2vec from dataset by gensim
w2v_gensim.py  <datasetpath>

this script will produce skipgram_model_gensim.pk and word2vec.txt 

## Train
python3 ../../one_torch_main.py -a train -d ../../data/chinese_tradition_poetry.csv --src char_rnn_experiment.py -c


## Interact
python3 ../../one_torch_main.py -a interact -e experiment_home/char_rnn_experiment#xxxxx/  --src char_rnn_experiment.py  --no_cuda
