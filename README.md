### Interactive Match LSTM

the interactive-match-lstm model for questioin duplicate recognization

word embedding: glove(6B), randomly initialize words not in glove

hidden state unit number in LSTMs: 150

lr: 0.1

optimizor: SGD

only save the model that achieved the lowest loss on dev set

pre procession: download glove(6B), rename the .txt file of 300d word embeddings to `vector.txt` and put it under `data`, download quora question pairs' train set, remove the first row, rename it to `all.csv` and put it under `data`, run `pre_procession.py` to generate `queries.txt`, `docs.txt`, `validate_queries.txt`, `validate_docs.txt` , `train_ground_truths.txt`, `validate_ground_truths.txt`, `test_queries.txt`, `test_docs.txt` and `test_ground_truths.txt`(data will be shuffled before each time of pre procession)

train: `python main.py`

(pre_procession.py is written under py3, while the other parts under py2)

the sketch of model is under designs/
