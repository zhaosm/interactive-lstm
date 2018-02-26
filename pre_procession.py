import os
import csv
import random
import nltk
from nltk import word_tokenize
nltk.download('punkt')

data_dir = './data'


def create_dataset():
    _train_queries = []
    _train_docs = []
    _train_ground_truths = []
    validate_queries = []
    validate_docs = []
    validate_ground_truths = []
    with open(os.path.join(data_dir, 'all.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            query = word_tokenize(row[3])
            doc = word_tokenize(row[4])
            if not len(query) == 0 and not len(doc) == 0:
                _train_queries.append(' '.join(query) + '\n')
                _train_docs.append(' '.join(doc) + '\n')
                _train_ground_truths.append(' '.join(row[5]) + '\n')
    data_size = len(_train_queries)
    random_idxs = [i for i in range(data_size)]
    random.shuffle(random_idxs)
    train_queries = [_train_queries[i] for i in random_idxs][:data_size]
    train_docs = [_train_docs[i] for i in random_idxs][:data_size]
    train_ground_truths = [_train_ground_truths[i] for i in random_idxs][:data_size]
    validate_size = data_size // 100
    validate_queries = train_queries[:validate_size]
    validate_docs = train_docs[:validate_size]
    validate_ground_truths = train_ground_truths[:validate_size]
    train_queries = train_queries[validate_size:]
    train_docs = train_docs[validate_size:]
    train_ground_truths = train_ground_truths[validate_size:]
    with open(os.path.join(data_dir, 'queries.txt'), 'w') as f:
        f.writelines(train_queries)
    with open(os.path.join(data_dir, 'docs.txt'), 'w') as f:
        f.writelines(train_docs)
    with open(os.path.join(data_dir, 'validate_queries.txt'), 'w') as f:
        f.writelines(validate_queries)
    with open(os.path.join(data_dir, 'validate_docs.txt'), 'w') as f:
        f.writelines(validate_docs)
    with open(os.path.join(data_dir, 'train_ground_truths.txt'), 'w') as f:
        f.writelines(train_ground_truths)
    with open(os.path.join(data_dir, 'validate_ground_truths.txt'), 'w') as f:
        f.writelines(validate_ground_truths)


def create_testset():
    test_queries = []
    test_docs = []
    test_ground_truths = []
    with open(os.path.join(data_dir, 'test.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            query = word_tokenize(row[1])
            doc = word_tokenize(row[2])
            if (query) == 0 or len(doc) == 0:
                print("empty query or doc: query: " + str(query) + ", doc: " + str(doc))
            test_queries.append(' '.join(query) + '\n')
            test_docs.append(' '.join(doc) + '\n')
            test_ground_truths.append('1\n')
    with open(os.path.join(data_dir, 'test_queries.txt'), 'w') as f:
        f.writelines(test_queries)
    with open(os.path.join(data_dir, 'test_docs.txt'), 'w') as f:
        f.writelines(test_docs)
    with open(os.path.join(data_dir, 'test_ground_truths.txt'), 'w') as f:
        f.writelines(test_ground_truths)

if __name__ == '__main__':
    create_dataset()
    create_testset()
