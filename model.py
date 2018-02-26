import tensorflow as tf
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class InteractiveMatchLSTM(object):
    def __init__(self,
                 num_lstm_units,
                 num_labels,
                 embed,
                 max_gradient_norm=5.0
                 ):
        self.num_lstm_units = num_lstm_units
        self.texts1 = tf.placeholder(tf.string, [None, None], name='texts1')  # batch_size*max_len
        self.texts2 = tf.placeholder(tf.string, [None, None],
                                     name='texts2')  # batch_size*max_len, PAD THE TWO TEXTS TO SAME LENGTH
        self.texts_length1 = tf.placeholder(tf.int32, [None], name='texts_length1')  # shape: batch
        self.texts_length2 = tf.placeholder(tf.int32, [None], name='texts_length2')  # shape: batch
        self.labels = tf.placeholder(tf.int64, [None], name='labels')  # shape: batch
        self.symbol2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True)
        self.learning_rate = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.index_input1 = self.symbol2index.lookup(self.texts1)  # batch*max_len
        self.index_input2 = self.symbol2index.lookup(self.texts2)  # batch*max_len
        self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        self.embed_input1 = tf.nn.embedding_lookup(self.embed, self.index_input1)  # batch*max_len*embed_unit
        self.embed_input2 = tf.nn.embedding_lookup(self.embed, self.index_input2)  # batch*max_len*embed_unit

        # zero padding
        self._batch_size = tf.shape(self.texts_length1)[0]
        self._max_length = tf.shape(self.texts1)[1]
        self.mask1 = tf.sequence_mask(self.texts_length1, maxlen=self._max_length,
                                      dtype=tf.float32)  # shape: batch*max_len
        self.mask1_extended = tf.concat([tf.zeros([self._batch_size, 1], tf.float32), self.mask1], 1)
        self.mask2 = tf.sequence_mask(self.texts_length2, maxlen=self._max_length,
                                      dtype=tf.float32)  # shape: batch*max_len
        self.mask2_extended = tf.concat([tf.zeros([self._batch_size, 1], tf.float32), self.mask2], 1)
        # debug
        print("mask1 size: " + str(self.mask1.shape))
        self.embed_input1 = tf.transpose(self.embed_input1, [2, 0, 1]) * self.mask1  # shape: embed_unit*batch*max_len
        self.embed_input1 = tf.transpose(self.embed_input1, [2, 1, 0])  # shape: max_len*batch*embed_units
        self.embed_input2 = tf.transpose(self.embed_input2, [2, 0, 1]) * self.mask2  # shape: embed_unit*batch*max_len
        self.embed_input2 = tf.transpose(self.embed_input2, [2, 1, 0])  # shape: max_len*batch*embed_units

        zero_state = tf.zeros(shape=[self._batch_size, self.num_lstm_units], dtype=tf.float32)
        h_s1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        c_s1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        h_s1 = h_s1.write(0, zero_state)
        c_s1 = c_s1.write(0, zero_state)

        h_s2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        c_s2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        h_s2 = h_s2.write(0, zero_state)
        c_s2 = c_s2.write(0, zero_state)

        h_r = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        c_r = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        h_r = h_r.write(0, zero_state)
        c_r = c_r.write(0, zero_state)

        self._initializer = tf.truncated_normal_initializer(stddev=0.1)

        t = tf.constant(1, dtype=tf.int32)  # TO DO: check this
        c = lambda x, hs1, cs1, hs2, cs2, hr, cr: tf.less(x, self._max_length + 1)
        b = lambda x, hs1, cs1, hs2, cs2, hr, cr: self._match_step(x, hs1, cs1, hs2, cs2, hr, cr)
        t, self.h_s1, self.c_s1, self.h_s2, self.c_s2, self.h_r, self.c_r = tf.while_loop(cond=c, body=b, loop_vars=(
        t, h_s1, c_s1, h_s2, c_s2, h_r, c_r))

        self.h_r = tf.transpose(self.h_r.stack(), [1, 0, 2])  # shape: [batch_size, max_len, num_lstm_units]
        # get final states. don't need to subtract seqlen by 1 because we take zero states also in count
        self.final_h_r = tf.gather_nd(self.h_r, tf.stack(
            [tf.range(self._batch_size), tf.maximum(self.texts_length1, self.texts_length2)],
            axis=1))  # shape: [batch_size, num_lstm_units]

        with tf.variable_scope('fully_connect'):
            self.w_fc = tf.get_variable(shape=[num_lstm_units, num_labels], initializer=self._initializer, name='w_fc')
            self.b_fc = tf.get_variable(shape=[num_labels], initializer=self._initializer, name='b_fc')
        self.logits = tf.matmul(self.final_h_r, self.w_fc) + self.b_fc

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits),
                                  name='loss')
        mean_loss = self.loss / tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
        predict_labels = tf.argmax(self.logits, 1, 'predict_labels')
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.labels, predict_labels), tf.int64), name='accuracy')
        self.params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=3, pad_step_number=True,
                                    keep_checkpoint_every_n_hours=1.0)

    def _match_step(self, t, h_s1, c_s1, h_s2, c_s2, h_r, c_r):
        """

        :param t: time index(start from 1)
        :self.embed_input1: tensor, shape: [max_length, batch_size, embed_units]
        :param h_s1: TensorArray, hidden states of text1 till last time step, t tensors of size [batch_size, num_lstm_units]
        :self.embed_input2: similar to input1
        :param h_s2: similar to h_s1
        :param h_r: TensorArray, hidden states of lstmr till last time step, t tensors of size [batch_size, num_lstm_units]
        :return: new t and new h_r(t tensors)
        """
        # lstms calculate first
        inputs_s1 = tf.concat([self.embed_input1[t - 1, :, :], h_r.read(t - 1)],
                              axis=1)  # shape: [batch_size, num_lstm_units * 2]
        inputs_s2 = tf.concat([self.embed_input2[t - 1, :, :], h_r.read(t - 1)], axis=1)
        with tf.variable_scope('lstm_s'):
            newc_s1, newh_s1 = self._lstm(inputs=inputs_s1, states=(c_s1.read(t - 1), h_s1.read(t - 1)))
        with tf.variable_scope('lstm_s', reuse=True):
            newc_s2, newh_s2 = self._lstm(inputs=inputs_s2, states=(c_s2.read(t - 1), h_s2.read(t - 1)))
        c_s1 = c_s1.write(t, newc_s1)
        h_s1 = h_s1.write(t, newh_s1)
        c_s2 = c_s2.write(t, newc_s2)
        h_s2 = h_s2.write(t, newh_s2)

        # calculate attention
        with tf.variable_scope('attention'):
            at1 = self._attention(t, h_s1, h_s2, h_r, self.mask1_extended[:, :t + 1])
        with tf.variable_scope('attention', reuse=True):
            at2 = self._attention(t, h_s2, h_s1, h_r, self.mask2_extended[:, :t + 1])

        # lstmr update
        inputs_r = tf.concat([at1, at2], axis=1)  # shape: [batch_size, num_lstm_units * 2]
        with tf.variable_scope('lstm_r'):
            newc_r, newh_r = self._lstm(inputs=inputs_r, states=(c_r.read(t - 1), h_r.read(t - 1)))
        c_r = c_r.write(t, newc_r)
        h_r = h_r.write(t, newh_r)

        t = tf.add(t, 1)
        return t, h_s1, c_s1, h_s2, c_s2, h_r, c_r

    def _attention(self, t, h_self, h_other, h_r, mask_self):
        """

        :param t: time index(start from 1)
        :param h_self: TensorArray, hidden states of self till last time step, t + 1 tensors of size [batch_size, num_lstm_units]
        :param h_other: TensorArray, hidden states of other, size and tensor shape: same as above
        :param h_r: TensorArray, hidden states of rlstm, t tensors of shape: [batch_size, num_lstm_units]
        :return: a attention-based presentation of 'self', shape: [batch_size, num_lstm_units]
        """
        We = tf.get_variable(shape=[self.num_lstm_units, 1], initializer=self._initializer, name='W_e')
        Wo = tf.get_variable(shape=[self.num_lstm_units, self.num_lstm_units], initializer=self._initializer,
                             name='W_other')
        Ws = tf.get_variable(shape=[self.num_lstm_units, self.num_lstm_units], initializer=self._initializer,
                             name='W_self')
        Wa = tf.get_variable(shape=[self.num_lstm_units, self.num_lstm_units], initializer=self._initializer,
                             name='W_attention')  # shape: batch_size

        etj = tf.einsum('ijk,kl->ijl', h_self.stack(), Ws) + tf.matmul(h_other.read(t), Wo) + tf.matmul(h_r.read(t - 1),
                                                                                                        Wa)
        etj = tf.transpose(etj, [1, 0, 2])  # shape: [batch_size, t, num_lstm_units]
        etj = tf.squeeze(tf.einsum('ijk,kl->ijl', tf.tanh(etj), We), axis=2)  # shape: [batch_size, t]
        etj = tf.exp(etj) * mask_self
        etj_sums = tf.reduce_sum(etj, axis=1)
        atj = tf.transpose(tf.transpose(etj) / etj_sums)
        at = tf.transpose(tf.transpose(h_self.stack(), [2, 1, 0]) * atj, [1, 2, 0])
        at = tf.reduce_sum(at, axis=1)  # shape: [batch_size, num_lstm_units]
        return at

    def _lstm(self, inputs, states):
        c, h = states
        _wi = tf.get_variable('lstm_cell_wi', dtype=tf.float32,
                              shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self.num_lstm_units],
                              initializer=tf.orthogonal_initializer())
        _bi = tf.get_variable('lstm_cell_bi', dtype=tf.float32, shape=[self.num_lstm_units],
                              initializer=tf.constant_initializer(0.0))
        _wo = tf.get_variable('lstm_cell_wo', dtype=tf.float32,
                              shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self.num_lstm_units],
                              initializer=tf.orthogonal_initializer())
        _bo = tf.get_variable('lstm_cell_bo', dtype=tf.float32, shape=[self.num_lstm_units],
                              initializer=tf.constant_initializer(0.0))
        _wf = tf.get_variable('lstm_cell_wf', dtype=tf.float32,
                              shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self.num_lstm_units],
                              initializer=tf.orthogonal_initializer())
        _bf = tf.get_variable('lstm_cell_bf', dtype=tf.float32, shape=[self.num_lstm_units],
                              initializer=tf.constant_initializer(1.0))
        _wc = tf.get_variable('lstm_cell_wc', dtype=tf.float32,
                              shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self.num_lstm_units],
                              initializer=tf.orthogonal_initializer())
        _bc = tf.get_variable('lstm_cell_bc', dtype=tf.float32, shape=[self.num_lstm_units],
                              initializer=tf.constant_initializer(0.0))
        i = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), _wi) + _bi)
        o = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), _wo) + _bo)
        f = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), _wf) + _bf)
        _c = tf.tanh(tf.matmul(tf.concat([inputs, h], 1), _wc) + _bc)
        new_c = f * c + i * _c
        new_h = o * tf.tanh(new_c)
        return new_c, new_h

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def train_step(self, session, data):
        input_feed = {self.texts1: data['texts1'],
                      self.texts2: data['texts2'],
                      self.texts_length1: data['texts_length1'],
                      self.texts_length2: data['texts_length2'],
                      self.labels: data['labels']}
        # for debug
        # output_feed = [self.loss, self.accuracy, self.update, self.embed_input1, self.embed_input2, self.h_r, self.final_h_r]
        output_feed = [self.loss, self.accuracy, self.update, self.final_h_r, self.logits]
        return session.run(output_feed, input_feed)


