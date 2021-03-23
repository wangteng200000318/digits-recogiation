import tensorflow as tf

batch_size = 128
total_words = 10000
# the max length of sentence is 80, if length of sentence lower than 80, the sentence will be padded.
# If length of sentence more than 80, the sentence will be truncated
max_sentence_length = 80

embedding_len = 100
(train_data, train_label), (test_data, test_label) = tf.keras.datasets.imdb.load_data(num_words=total_words)

word_index = tf.keras.datasets.imdb.get_word_index()

# set a encode and decode table
# '<PAD>' corresponding to 0, '<START>' corresponding to 1, '<UNK>' corresponding to 2, '<UNUSED>' corresponding to 3
word_index = {key: val + 3 for key, val in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = {val: key for key, val in word_index.items()}


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# truncating and padding the sentence, to realise the sentences have the same length
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_sentence_length)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=max_sentence_length)

# use shuffle to change the sequence of data
data_train = tf.data.Dataset.from_tensor_slices((train_data, train_label))
data_train = data_train.shuffle(1000).batch(batch_size=batch_size, drop_remainder=True)
data_test = tf.data.Dataset.from_tensor_slices((test_data, test_label))
data_test = data_test.batch(batch_size=batch_size, drop_remainder=True)


class TestRNN(tf.keras.Model):
    def __init__(self, units):
        super(TestRNN, self).__init__()
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]

        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_sentence_length)
        self.rnn_cell0 = tf.keras.layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = tf.keras.layers.SimpleRNNCell(units, dropout=0.5)
        self.out_layers = tf.keras.Sequential([tf.keras.layers.Dense(units),
                                               tf.keras.layers.Dropout(0.5),
                                               tf.keras.layers.ReLU(),
                                               tf.keras.layers.Dense(1)])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.out_layers(out1, training)
        prob = tf.sigmoid(x)
        return prob


if __name__ == '__main__':
    units = 64
    epochs = 20
    model = TestRNN(units)
    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'],
                  experimental_run_tf_function=False)
    model.fit(data_train, epochs=epochs, validation_data=data_test)
    model.evaluate(data_test)
