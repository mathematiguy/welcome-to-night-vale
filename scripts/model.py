import os, re
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import TensorBoard

DATA_PATH = r'C:\Users\caleb\Documents\Data Science\welcome-to-night-vale\data'
TRANSCRIPTS_PATH = os.path.join(DATA_PATH, 'transcripts')

def load_data(file):
	# load ascii text and covert to lowercase
	with open(os.path.join(DATA_PATH, 'Welcome To Night Vale.txt'),
	          'r', encoding='utf-8') as f:
	    wtnv_text = f.read().lower()