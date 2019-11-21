from flask import Flask, render_template, request, jsonify

import re
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import heapq
import datetime

# NER imports
from flair.data import Corpus, Sentence
from flair.models import SequenceTagger
import torch

# Category imports
from keras.preprocessing import text, sequence
from keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

#sess = tf.Session()
#graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
#set_session(sess)

print(datetime.datetime.now(), 'loading Bi-LSTM.h5')
cat_model = load_model('Bi-LSTM.h5')
print(datetime.datetime.now(), 'Category model loaded')


ner_model = SequenceTagger.load('checkpoint.pt')
print(datetime.datetime.now(), 'NER model loaded')

with open('./token.pkl','rb') as infile:
	token = pickle.load(infile)
		
with open('./cols.pkl','rb') as infile:
	cols = pickle.load(infile)


app = Flask(__name__)


def predict_category(sentence):	
	seq_x = sequence.pad_sequences(token.texts_to_sequences([sentence]), maxlen=60)
	
	#global sess
	#global graph
	#with graph.as_default():
		#set_session(sess)
	predictions = cat_model.predict(seq_x)
		
	index = predictions.argmax(axis=1)[0]
	category = cols[index]
	
	indices = heapq.nlargest(2, range(len(predictions[0])), predictions[0].take)
	categories = [cols[idx] for idx in indices]
	
	print(categories)
	
	if 'politics' in categories[0]:
		return categories[0]
	elif 'business' in categories[0]:
		return categories[0]
	elif 'environment' in categories:
		return 'environment'
	elif 'film' in categories:
		categories.remove('film')
		return (', '.join(categories))
	else:
		return (', '.join(categories))

	
def print_ner_tags(tags, sentence):
	type2color = {
                    'PERSON': 'tag is-info',
					'NORP': 'tag is-light',
					'FAC': 'tag is-black',
					'ORG': 'tag is-success',
					'GPE': 'tag is-danger',
					'LOC': 'tag is-primary',
					'PRODUCT': 'tag is-dark',
					'EVENT': 'tag is-warning',
					'WORK_OF_ART': 'tag is-link'
                }
	
	output = "<p>" + sentence + "</p>"
	for entity in tags:
		tag = type2color[tags[entity]]
		span = "<span class='{}'> {} </span>".format(tag, entity)
		output = output.replace(entity, span)
	
	# output = "<p>"
	# for word in sentence.split():
		# if word in tags:
			# _id = type2color[tags[word]]
			# span = "<span class='{}'> {} </span>".format(_id, word)
			# output += span
		# else:
			# output += word
			
		# output += " "
	
	print(output)
	
	return output
	
	
def predict_ner(sent):	
	sentence = Sentence(sent)
	ner_model.predict(sentence)
	
	print(sentence.to_tagged_string())
	
	tags = {}
	for entity in sentence.get_spans('ner'):		
		tags[entity.text] = entity.tag
			
	print(tags)
	output = print_ner_tags(tags, sent)
	
	return output
	

@app.route('/')
def index():		
	return render_template('sam_ui.html')
	

@app.route('/_get_data/', methods=['POST'])
def _get_data():

	if request.method == "POST":
		sentence = request.form["sentence"]
	
	print(type(sentence))
	print('Sentence:', sentence)
		
	category = predict_category(sentence)
	ner = predict_ner(sentence)
	
	data = "<p>" + ner + "</p> <br> <p class = 'tag is-medium is-dark'> Category: " + category + "</p>"
	
	return data


if __name__=="__main__":
    app.run(debug=True)
	