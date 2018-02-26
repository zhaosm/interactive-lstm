import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from model import InteractiveMatchLSTM, _START_VOCAB
import os
import time
import random
import csv

random.seed(1229)
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_boolean("read_graph", False, "Set to False to build graph.")
tf.app.flags.DEFINE_integer("symbols", 35939, "vocabulary size.")
tf.app.flags.DEFINE_integer("labels", 2, "Number of labels.")
tf.app.flags.DEFINE_integer("epoch", 100000, "Number of epoch.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 150, "Size of each model layer.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")

FLAGS = tf.app.flags.FLAGS


def load_data(path, fname):
    print('Creating dataset...')
    data = []
    with open('%s/%s' % (path, fname)) as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            tokens = line.split()
            data.append(tokens)
    return data


def build_vocab(path, data, get_embed=False):
    print("Creating vocabulary...")
    words = set()
    for line in data:
        for word in line:
            if len(word) == 0:
                continue
            words.add(word)
    words = list(words)
    vocab_list = _START_VOCAB + words
    FLAGS.symbols = len(vocab_list)
    if not get_embed:
        return vocab_list
    print("Loading word vectors...")
    embed = np.random.normal(0.0, np.sqrt(1. / (FLAGS.embed_units)), [len(vocab_list), FLAGS.embed_units])
    # debug
    # embed = np.array(embed, dtype=np.float32)
    # return vocab_list, embed
    with open(os.path.join(path, 'vector.txt')) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            info = line.split()
            if info[0] not in vocab_list:
                continue
            embed[vocab_list.index(info[0])] = [float(num) for num in info[1:]]
    embed = np.array(embed, dtype=np.float32)
    np.savetxt(os.path.join(FLAGS.data_dir, 'embed.txt'), embed, delimiter=',')
    return vocab_list, embed


def gen_batch_data(data):
    def padding(sent, l):
        return sent + ['_PAD'] * (l - len(sent))

    max_len = max([max(len(item[0]), len(item[1])) for item in data])
    texts1, texts2, texts_length1, texts_length2, labels = [], [], [], [], []

    for item in data:
        texts1.append(padding(item[0], max_len))
        texts2.append(padding(item[1], max_len))
        texts_length1.append(len(item[0]))
        texts_length2.append(len(item[1]))
        labels.append(np.array(item[2]))

    batched_data = {'texts1': np.array(texts1), 'texts2': np.array(texts2), 'texts_length1': texts_length1,
                    'texts_length2': texts_length2, 'labels': labels}

    return batched_data


def train(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    while ed + FLAGS.batch_size < len(dataset):
        st, ed = ed, ed + FLAGS.batch_size if ed + \
                                              FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs = model.train_step(sess, batch_data)
        # debug
        # print("train loss for debug: %.8f" % outputs[0])
        if np.isnan(outputs[0]):
            print("loss is nan, texts1: " + str(batch_data['texts1']) + ", texts2: " + str(batch_data['texts2']) +
                  ", texts_length1: " + str(batch_data['texts_length1']) + ", texts_length2: " +
                  str(batch_data['texts_length2']) + ", labels: " + str(batch_data['labels']))  #  + ", final_hr: " + str(outputs[-2]) +
                 #  ", logits: " + str(outputs[-1]))
            raise Exception
        loss += outputs[0]
        accuracy += outputs[1]
    sess.run(model.epoch_add_op)

    return loss / len(dataset), accuracy / len(dataset)


def evaluate(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    while ed + FLAGS.batch_size < len(dataset):
        st, ed = ed, ed + FLAGS.batch_size if ed + FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        # print batch_data['texts2']
        outputs = sess.run(['loss:0', 'accuracy:0'],
                           {'texts1:0': batch_data['texts1'], 'texts2:0': batch_data['texts2'],
                            'texts_length1:0': batch_data['texts_length1'],
                            'texts_length2:0': batch_data['texts_length2'],
                            'labels:0': batch_data['labels']})
        # debug
        # print("evaluate loss for debug: %.8f" % outputs[0])
        loss += outputs[0]
        accuracy += outputs[1]
    return loss / len(dataset), accuracy / len(dataset)


def test(model, sess, dataset):
    st, ed = 0, 0
    count = 0
    start_time = time.time()
    with open('submission_remain.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['test_id', 'is_duplicate'])
        while ed  < len(dataset):
            st, ed = ed, ed + FLAGS.batch_size if ed + FLAGS.batch_size < len(dataset) else len(dataset)
            batch_data = gen_batch_data(dataset[st:ed])
            # print batch_data['texts2']
            outputs = sess.run([model.logits],
                               {'texts1:0': batch_data['texts1'], 'texts2:0': batch_data['texts2'],
                                'texts_length1:0': batch_data['texts_length1'],
                                'texts_length2:0': batch_data['texts_length2'],
                                'labels:0': batch_data['labels']})
            logits = np.exp(outputs[0])
            sum = np.sum(logits, axis=1)
            predicts = np.transpose(np.transpose(logits) / sum)
            for predict in predicts:
                print("test No.%d" % (count + 1))
                if np.isnan(predict[1]):
                    predict[1] = 0.0
                writer.writerow([count, float(predict[1])])
                count += 1
    print("test time: %.8f" % (time.time() - start_time))


def run():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        data_queries = load_data(FLAGS.data_dir, 'queries.txt')
        data_docs = load_data(FLAGS.data_dir, 'docs.txt')
        train_ground_truths = []
        with open(os.path.join(FLAGS.data_dir, 'train_ground_truths.txt')) as f:
            for row in f:
                train_ground_truths.append(int(row.strip('\n')))
        data_train = zip(data_queries, data_docs, train_ground_truths)

        # validate data
        validate_queries = load_data(FLAGS.data_dir, 'validate_queries.txt')
        validate_docs = load_data(FLAGS.data_dir, 'validate_docs.txt')
        validate_ground_truths = []
        with open(os.path.join(FLAGS.data_dir, 'validate_ground_truths.txt')) as f:
            for row in f:
                validate_ground_truths.append(int(row.strip('\n')))
        data_dev = zip(validate_queries, validate_docs, validate_ground_truths)

        # debug
        # embed = np.zeros(shape=[FLAGS.symbols, FLAGS.embed_units], dtype=np.float32)
        try:
            embed = np.loadtxt(os.path.join(FLAGS.data_dir, 'embed.txt'), delimiter=',', dtype=np.float32)
            vocab = build_vocab(FLAGS.data_dir, data_queries + data_docs + validate_queries + validate_docs,
                                get_embed=False)
        except Exception:
            vocab, embed = build_vocab(FLAGS.data_dir, data_queries + data_docs + validate_queries + validate_docs,
                                       get_embed=True)

        model = InteractiveMatchLSTM(
            num_lstm_units=FLAGS.units,
            num_labels=FLAGS.labels,
            embed=embed
        )
        if FLAGS.log_parameters:
            model.print_parameters()

        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.symbol2index.insert(constant_op.constant(vocab),
                                              constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
            sess.run(op_in)

        if FLAGS.is_train:
            summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
            pre_losses = [1e18] * 3
            best_val_loss = 1000
            while model.epoch.eval() < FLAGS.epoch:
                epoch = model.epoch.eval()
                # comment below for debugging
                random.shuffle(data_train)
                start_time = time.time()
                train_loss, train_acc = train(model, sess, data_train)

                summary = tf.Summary()
                summary.value.add(tag='loss/train', simple_value=train_loss)
                summary.value.add(tag='accuracy/train', simple_value=train_acc)

                print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (
                    epoch, model.learning_rate.eval(), time.time() - start_time, train_loss, train_acc))
                val_loss, val_acc = evaluate(model, sess, data_dev)
                summary.value.add(tag='loss/dev', simple_value=val_loss)
                summary.value.add(tag='accuracy/dev', simple_value=val_acc)
                print("        dev_set, loss %.8f, accuracy [%.8f]" % (val_loss, val_acc))

                if val_loss < best_val_loss:  # when valid_accuracy > best_valid_accuracy, save the model
                    best_val_loss = val_loss
                    best_epoch = epoch
                    print("best epoch on validate set: %d" % best_epoch)
                    model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)

                    summary_writer.add_summary(summary, epoch)
                    # print("        test_set, loss %.8f, accuracy [%.8f]" % (test_loss, test_acc))

                if train_loss > max(pre_losses):
                    op = tf.assign(model.learning_rate, model.learning_rate * 0.5)
                    sess.run(op)
                pre_losses = pre_losses[1:] + [train_loss]
        else:
            test_queries = load_data(FLAGS.data_dir, 'test_queries.txt')
            test_docs = load_data(FLAGS.data_dir, 'test_docs.txt')
            test_ground_truths = []
            with open(os.path.join(FLAGS.data_dir, 'test_ground_truths.txt')) as f:
                for row in f:
                    test_ground_truths.append(int(row.strip('\n')))
            data_test = zip(test_queries, test_docs, test_ground_truths)
            test(model, sess, data_test)

if __name__ == '__main__':
    run()
