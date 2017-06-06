'''
Author: Caleb Moses
Date: 04-06-2017

This file trains a character-level multi-layer RNN on text data.

Code is based on Andrej Karpathy's implementation in Torch at:
https://github.com/karpathy/char-rnn/blob/master/train.lua

I modified the model to run using TensorFlow and Keras. Supports GPUs, 
as well as many other common model/optimization bells and whistles.
'''

import sys, argparse
import os, re
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils  import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard


def parse_args():
    '''Parses all keyword arguments for model and returns them.

       Returns:
        - data_dir:   (str) The directory to the text file(s) for training.
        - rnn_size:   (int) The number of cells in each hidden layer in 
                      the network.
        - num_layers: (int) The number of hidden layers in the network.
        - dropout:    (float) Dropout value (between 0, 1 exclusive).'''

    # initialise parser
    parser = argparse.ArgumentParser()

    # add arguments, set default values and expected types
    parser.add_argument("-data_dir",
        help="The directory to the text file(s) for training.")
    parser.add_argument("-seq_length", type=int, default=25,
        help="The length of sequences to be used for training")
    parser.add_argument("-batch_size", type=int, default=100,
        help="The number of minibatches to be used for training")
    parser.add_argument("-rnn_size", type=int, default=128,
        help="The number of cells in each hidden layer in the network")
    parser.add_argument("-num_layers", type=int, default=2,
        help="The number of hidden layers in the network")
    parser.add_argument("-dropout", type=float, default=0.1,
        help="Dropout value (between 0, 1 exclusive)")
    parser.add_argument("-epochs", type=int, default=1,
        help="Number of epochs for training")
    parser.add_argument("-tensorboard", type=int, default=1,
        help="Save model statistics to TensorBoard")

    # parse arguments and return their values
    args = parser.parse_args()
    return args.data_dir, args.seq_length, args.batch_size, args.rnn_size, \
           args.num_layers, args.dropout, args.epochs, args.tensorboard


def print_data(text):
    '''Re-encodes text so that it can be printed to command line 
       without raising a UnicodeEncodeError.
       Incompatible characters are dropped before printing.

       Args:
       - text: (str) The text to be printed'''
    print(text.encode(sys.stdout.encoding, errors='replace'))


def load_data(data_dir, encoding='utf-8'):
    '''Appends all text files in data_dir into a single string and returns it.
       All files are assumed to be utf-8 encoded, and of type '.txt'.

       Args:
       - data_dir: (str) The directory to text files for training.
       - encoding: (str) The type of encoding to use when decoding each file.

       Returns:
       - text_data: (str) Appended files as a single string.'''
    print("Loading data from %s" % os.path.abspath(data_dir))
    # Initialise text string
    text_data = ''
    # select .txt files from data_dir
    for filename in filter(lambda s: s.endswith(".txt"), os.listdir(data_dir)):
        # open file with default encoding
        print("loading file: %s" % filename)
        filepath = os.path.abspath(os.path.join(data_dir, filename))
        with open(filepath,'r', encoding = encoding) as f:
            text_data += f.read() + "\n"
    return text_data


def pre_processing(text_data, seq_length):
    '''Preprocesses text_data for RNN model.

       Args:
       - text: (str) text file to be processed.
       - seq_length: (int) length of character sequences to be considered 
                     in the training set.

       Returns:
       - char_to_int: (dict) Maps characters in the character set to ints.
       - int_to_char: (dict) Maps ints to characters in the character set.
       - n_chars: (int) The number of characters in the text.
       - n_vocab: (int) The number of unique characters in the text.'''

    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(set(text_data))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    # summarize the loaded data
    n_chars = len(text_data)
    n_vocab = len(chars)
    
    return char_to_int, int_to_char, n_chars, n_vocab


def generate_train_batches(text_data, seq_length, batch_size):
    '''A generator that returns training sequences of length seq_length, in
       batches of size batch_size.

       Args:
       - text_data: (str) The text for training.
       - seq_length: (int) The length of each training sequence.
       - batch_size: (int) The size of minibatches for training.

       Returns:
       - X: (numpy.array) An array of sequences for training. Each value is
            normalized to between 0 and 1.
            shape.X = (batch_size, seq_length, 1)
            X.dtype = np.float32
       - y: (numpy.array) An array of next characters for each sequence in X.
            Each character is one-hot encoded using the pre-defined vocabulary
            shape.y = (batch_size, n_vocab)
            y.dtype = np.int32'''
    # while loop ensure generator runs forever
    char_to_int, int_to_char, n_chars, n_vocab = \
    			pre_processing(text_data, seq_length)
    
    while True:
        for batch in range(n_chars // batch_size):
            # prepare the dataset of input to output pairs encoded as integers
            dataX = []
            dataY = []
            for start in range(batch * batch_size, 
            				   batch * batch_size + batch_size):        
                seq_in  = text_data[start:start + seq_length]
                seq_out = text_data[start + seq_length]

                dataX.append([char_to_int[char] for char in seq_in])
                dataY.append(char_to_int[seq_out])

            X = np.reshape(dataX, (batch_size, seq_length, 1))

            # normalise X to [0, 1]
            X = X / n_vocab

            # one hot encode the output variable
            y = np_utils.to_categorical(dataY, num_classes=n_vocab)

            yield X, y


def build_model(batch_size, seq_length, n_vocab, 
                rnn_size, num_layers, drop_prob):
    '''Defines the RNN LSTM model.
       Args:
        - batch_size: (int) The size of each minibatches.
        - seq_length: (int) The length of each sequence for the model.
        - rnn_size: (int) The number of cells in each hidden layer.
        - num_layers: (int) The number of hidden layers in the network.
        - drop_prob: (float) The proportion of cells to drop in each dropout 
                             layer.
       Returns:
        - model: (keras.models.Sequential) The constructed Keras model.'''

    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            # add first hidden layer
            model.add(LSTM(rnn_size, 
                           batch_input_shape=(batch_size, seq_length, 1),
                           return_sequences=True))
        elif i == num_layers - 1:
            # add last hidden layer
            model.add(LSTM(rnn_size, return_sequences=False))
        else:
            # add middle hidden layer
            model.add(LSTM(rnn_size, return_sequences=True))
        model.add(Dropout(drop_prob))
    # add output layer
    model.add(Dense(n_vocab, activation='softmax'))
    return model

def set_callbacks(tensorboard):
	'''Set callbacks for Keras model.

	Args:
	 - tensorboard: (int) Add tensorboard callback if tensorboard == 1

	Returns:
	 - callbacks: (list) list of callbacks for model'''

	callbacks = [ModelCheckpoint(
	    			'checkpoints\\weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
	if tensorboard:
		callbacks.append(TensorBoard(log_dir=r'..\logs', histogram_freq=0.01))  

	return callbacks

def Main():
    # load text data to memory
    text_data = load_data(data_dir)

    print("Here is a sample of the text:\n", text_data[:100])

    # preprocess the text - construct character dictionaries etc
    char_to_int, int_to_char, n_chars, n_vocab = \
            pre_processing(text_data, seq_length=seq_length)

    # build and compile Keras model
    model = build_model(batch_size, seq_length, n_vocab,
                        rnn_size, num_layers, drop_prob)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metric=['accuracy'])  	

    # fit model using generator
    model.fit_generator(
        generate_train_batches(text_data, seq_length, batch_size), 
        epochs = epochs,
        steps_per_epoch=n_chars // batch_size,
        callbacks=set_callbacks(tensorboard))


if __name__ == "__main__":
	# parse keyword arguments
    data_dir, seq_length, batch_size, rnn_size, \
    num_layers, drop_prob, epochs, tensorboard = parse_args()

    Main()