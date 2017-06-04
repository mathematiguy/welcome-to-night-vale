import os, re
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import TensorBoard

ROOT_PATH = r'/home/ubuntu/pynb/welcome-to-night-vale'
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODEL_PATH = os.path.join(DATA_PATH, 'models')
LOG_PATH = os.path.join(ROOT_PATH, 'logs')

def load_text(filepath):
	'''Load text file from DATA_PATH'''
	with open(os.path.join(DATA_PATH, filepath),
	          'r', encoding='utf-8') as f:
	    text = f.read()
	    return text

def pre_processing(text, seq_length=100):
	'''Preprocesses text file for model.
	   Lowercases text, converts to integer arrays of length seq_length.

	   Args:
	  	text - text file to be processed
	  	seq_length - length of character sequences to be considered 
	   				 in the training set
		
	   Returns:
		X - Array of integers representing character sequences from
			the training text with length seq_length.
			X.shape = (n_chars - seq_length, seq_length, 1)
		y - Array of integers representing next characters for each
			sequence in X.
			y.shape = (n_chars - seq_length, n_vocab)'''

	# lowercase text
	text = text.lower()

	# create mapping of unique chars to integers, and a reverse mapping
	chars = sorted(list(set(text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))

	# summarize the loaded data
	n_chars = len(text)
	n_vocab = len(chars)
	print("Total Characters:", n_chars)
	print("Total Vocab:", n_vocab)

	# prepare the dataset of input to output pairs encoded as integers
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = text[i:i + seq_length]
		seq_out = text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])

	n_patterns = len(dataX)
	print("Total Patterns:", n_patterns)

	# reshape X to be [samples, time steps, features]
	X = np.reshape(dataX, (n_patterns, seq_length, 1))

	# normalize
	X = X / n_vocab

	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)

	return X, y

def build_model(X, y):
	'''define the LSTM model'''
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	return model

def Main():

	# load text file
	text_file = r'Welcome To Night Vale.txt'
	print("Loading text file: %s" % text_file)
	wtnv_text = load_text(text_file)

	# create input + target data
	print("Pre-processing text for model")
	X, y = pre_processing(wtnv_text)
	
	# check if a checkpoint for the file already exists
	model_file = os.path.join(MODEL_PATH, 'wtnv-keras-model.hd5')
	
	if os.path.exists(model_file):
		# load checkpoint

		print("Checkpoint exists, loading from file...")
		model = load_model(model_file)

	else:
		# build model from scratch
		print("No checkpoint available. Building model from scratch..")

		# build and compile mode
		model = build_model(X, y)
		model.compile(loss='categorical_crossentropy', optimizer='adam',
	              	  metric=['accuracy'])

	# tb_callback = TensorBoard(log_dir=LOG_PATH,
 #                          	  histogram_freq=0.01, write_graph=True, 
 #                          	  write_images=True)

	model.fit(X, y, batch_size=100, validation_split=0.3, 
			  verbose=1, epochs=10, 
			  # callbacks=[tb_callback]
			  )

	print("Saving model to file...")
	model.save(os.path.join(MODEL_PATH, 'wtnv-keras-model.hd5'))
	print("Done!")

if __name__ == "__main__":
	Main()
