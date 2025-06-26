import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',']

# Load data
data = json.load(open('intents.json'))

# Preprocess data
for intent in data['intents']:
for pattern in intent['patterns']:
# Tokenize patterns
word = nltk.word_tokenize(pattern)
words.extend(word)
# Add to documents
documents.append((word, intent['tag']))
# Add to classes
if intent['tag'] not in classes:
classes.append(intent['tag'])

# Create training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
# Bag of words
bag = []
word_patterns = doc[0]
word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
for word in words:
bag.append(1) if word in word_patterns else bag.append(0)

# Output row
output_row = list(output_empty)
output_row[classes.index(doc[1])] = 1

training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training)

# Split into input and output
train_x = list(training[:,0])
train_y = list(training[:,1])

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Define function to predict class
def predict_class(sentence):
# Bag of words
bag = [0]*len(words)
sentence_words = nltk.word_tokenize(sentence)
sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
for se in sentence_words:
for i,w in enumerate(words):
if w == se:
bag[i] = 1
res = model.predict(np.array([bag]))[0]
ERROR_THRESHOLD = 0.25
results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
results.sort(key=lambda x: x[1], reverse=True)
return_list = []
for r in results:
return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
return return_list

# Define function to get response
def get_response(intents_list,intents_json):
tag = intents_list[0]['intent']
list_of_intents_tag = intents_json['intents']
for i in list_of_intents_tag:
if(i['tag']== tag):
result = random.choice(i['responses'])
break
return result

# Test chatbot
while True:
message = input("")
ints = predict_class(message)
res = get_response(ints, data)
print(res)
```

