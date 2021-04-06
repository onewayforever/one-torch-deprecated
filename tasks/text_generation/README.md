## Build word2vec from dataset by gensim
w2v_gensim.py  <datasetpath>

this script will produce <dataset>_word2vec.txt 

## Train
python3 ../../one_torch_main.py -a train -d ../../data/chinese_tradition_poetry.csv --src char_rnn_experiment.py -c

### by config file
python3 ../../one_torch_main.py -a train --hparam=poetry.yml --src char_rnn_experiment.py -c 


## Interact
python3 ../../one_torch_main.py -a interact -e experiment_home/char_rnn_experiment#xxxxx/  --src char_rnn_experiment.py  --no_cuda
