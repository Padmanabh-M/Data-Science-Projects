import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy as np
import tflearn  #TFlearn is a modular and transparent deep learning library
                # built on top of Tensorflow. It was designed to provide a higher-level
                # API to TensorFlow in order to facilitate and speed-up experimentations,
                # while remaining fully transparent and compatible with it.

import tensorflow
import random
import json
import pickle



# Using json to read file
with open("intents.json") as file:
    data = json.load(file)
    
# print(data)


# The list inside the intents dictionary...
# print(data['intents'])



# ------------------------------------------------  Preprocessing  ---------------------------------------------------------



# We need get all the patterns and figure out what group (tag) they are in...

try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
    
except:
    words = []

    labels = []

    docs_x = []

    docs_y = []

    for intent in data['intents']:
        # we need to tokenize (split by space) and stem (root wordize) patterns...
        for pattern in intent['patterns']:
            # Tokenizing...
            wrds = nltk.word_tokenize(pattern) #returns list with all different words in that pattern

            words.extend(wrds) # words holds ['How', 'are', 'you', 'Is', 'anyone', 'there', '?'.........]

            docs_x.append(wrds) # [ ['Hi'], ['How', 'are', 'you'], ['Is', 'anyone', 'there', '?'].........]

            docs_y.append(intent['tag']) # docs_y holds ['greetings', 'greetings' x 5, 'goodbye' x 5  ....]
                                         # ..to classify each one of our patterns...

            if intent['tag'] not in labels:
                labels.append(intent['tag'])


    # Stem all the words from the words list and remove any duplicate elements so that we can figure out the 
    # vocabulary size of the model...(How many words it has seen already)

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # [ 'hi', 'how', 'ar', 'you', 'is', 'anyon', 'ther', '?', 'hello', 'good', 'day', .........]

    words = sorted(list(set(words)))

    labels = sorted(labels) # ['greeting', 'goodbye', 'age', 'name', 'shop', 'hours']



    # Neural networks only understand numbers....
    training = []
    output = []



    # Final output....class lighter
    out_empty = [0 for _ in range(len(labels))]


    # Creating bag of words...
    for x, doc in enumerate(docs_x): # docs_x holds [ ['Hi'], ['How', 'are', 'you'], ['Is', 'anyone', 'there', '?'].........]
        bag = []

        wrds = [stemmer.stem(w.lower()) for w  in doc] # doc holds ['How', 'are', 'you']
                            #  Word-bag example....
        for w in words:     #  words holds ['how', 'ar', 'you', 'is', 'anyon', 'ther', '?', 'hello', 'good', 'day', .........]
            if w in wrds:    #  wrds holds ['how', 'ar', 'you']
                bag.append(1) #  bag value [1    ,  1  ,  1,    0   , 0      ,   0   ,  0 ,  0     ,  0    ,  0   , ......]
            else:
                bag.append(0)

        output_row = out_empty[:]  # out_empty = [0,0,0,0,0,0] as many tags/labels...
                                                  # Example...
        output_row[labels.index(docs_y[x])] = 1   # ['age', 'goodbye', 'greeting', 'hours', 'name', 'shop']
                                                  # [0,0,1,0,0,0]

        training.append(bag)
        output.append(output_row)
        
        with open('data.pickle', 'wb') as f:
            pickle.dump((words, labels, training, output), f )


    training = np.array(training)
    output = np.array(output)






    # ------------------------------------------------  Training the model  ---------------------------------------------------

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])

net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, len(output[0]), activation="softmax")

net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
    
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")






# ----------------------------------------------  Testing functions  ------------------------------------------------------------

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)
                
                
def chat():
    print('Start talking with the bot! (Type quit to stop)')
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])[0]
        
        results_index = np.argmax(results)
        
        tag = labels[results_index]
        
        if results[results_index] > 0.6:
        
            for tg in data['intents']:
                if tag == tg['tag']:
                    responses = tg['responses']
                    
            print(random.choice(responses))
                    
        else:
            print('Wait what?....try again....')      
        
        
        
chat()