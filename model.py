import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pythainlp.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional, SimpleRNN, Flatten
from keras.utils import to_categorical

import service

# --------- Loading Data ---------

conn = service.Spreadsheet("1RO_bhae8ID7yrdFCysKirNi46123JgciCsmQ67uFSyE", "datasets!A:B")
sheet = conn.read()

data = pd.DataFrame(sheet[1:], columns=sheet[0])
# print(data)

X = data.text
y = data["class"]
# print(X, y)

y_nunique = y.nunique()
# print(y_nunique)

# --------- Tokenization ---------

X_tokens = X.apply(word_tokenize, keep_whitespace=False)
# print(X_tokens)

maxlen = X_tokens.apply(len).max()
# print(maxlen)

# --------- Indexing ---------

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_tokens)
# print(tokenizer.word_index)

vocab_size = len(tokenizer.word_index) + 1
# print(vocab_size)

X_tts = tokenizer.texts_to_sequences(X_tokens)
# print(tts)

# --------- Padding ---------

X_pad = pad_sequences(X_tts, maxlen=maxlen, padding="post")
# print(X_pad)

# --------- One-hot Encoding ---------

y_1hot = to_categorical(y)
# print(y_1hot)

# --------- Data Splits ---------

X_train, X_test, y_train, y_test = train_test_split(X_pad, y_1hot, train_size=.8, random_state=19)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# --------- Create Model ---------

#-------------Bi-LSTM-------------
Bi_LSTM = Sequential()
Bi_LSTM.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))
Bi_LSTM.add(Bidirectional(LSTM(64, activation="relu")))
Bi_LSTM.add(Dense(y_nunique, activation="softmax"))
Bi_LSTM.summary()

Bi_LSTM.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

Bi_LSTM.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1)

# --------- Visualization  ---------

loss, accuracy = Bi_LSTM.evaluate(X_test, y_test, verbose=1)
print(f"{'Test loss':15}: {loss}")
print(f"{'Test accuracy':15}: {accuracy}")

y_predict = Bi_LSTM.predict(X_test)

y_p = [np.argmax(i) for i in y_predict]
y_t = [np.argmax(i) for i in y_test]

cm = confusion_matrix(y_t, y_p)
print(cm)

Bi_LSTM.save("model.keras")

# message = "ขอคำแนะนำหน่อย"
# message = word_tokenize(message, keep_whitespace=False)
# message = tokenizer.texts_to_sequences([message])
# message = pad_sequences(message, maxlen=maxlen, padding="post")
# y_pred = Bi_LSTM.predict(message)

# print(np.argmax(y_pred))

#--------------RNN--------------

RNN = Sequential()
RNN.add(Embedding(input_dim=vocab_size, output_dim=128,input_length=maxlen))
RNN.add(SimpleRNN(64,return_sequences=True))
RNN.add(SimpleRNN(64))
RNN.add(Dense(y_nunique, activation='sigmoid'))
RNN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
RNN.summary

RNN.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1)

loss, accuracy = RNN.evaluate(X_test, y_test, verbose=1)
print(f"{'Test loss':15}: {loss}")
print(f"{'Test accuracy':15}: {accuracy}")

y_predict = RNN.predict(X_test)

y_p = [np.argmax(i) for i in y_predict]
y_t = [np.argmax(i) for i in y_test]

cm2 = confusion_matrix(y_t, y_p)
print(cm2)


#--------Fully Connected----------

Fully_Connected = Sequential()
Fully_Connected.add(Embedding(input_dim=vocab_size,output_dim=128,input_length=maxlen))
Fully_Connected.add(Flatten())
Fully_Connected.add(Dense(y_nunique,activation='relu'))
Fully_Connected.add(Dense(y_nunique,activation='sigmoid'))
Fully_Connected.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
Fully_Connected.summary()

Fully_Connected.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1)

loss, accuracy = Fully_Connected.evaluate(X_test, y_test, verbose=1)
print(f"{'Test loss':15}: {loss}")
print(f"{'Test accuracy':15}: {accuracy}")

y_predict = Fully_Connected.predict(X_test)

y_p = [np.argmax(i) for i in y_predict]
y_t = [np.argmax(i) for i in y_test]

cm3 = confusion_matrix(y_t, y_p)
print(cm3)


#--------------LSTM-------------

LSTM_Model = Sequential()
LSTM_Model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))
LSTM_Model.add(LSTM(64,activation = "relu", return_sequences=True))
LSTM_Model.add(LSTM(64, activation="relu"))
LSTM_Model.add(Dense(y_nunique, activation='sigmoid'))
LSTM_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
LSTM_Model.summary()

LSTM_Model.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1)

loss, accuracy = LSTM_Model.evaluate(X_test, y_test, verbose=1)
print(f"{'Test loss':15}: {loss}")
print(f"{'Test accuracy':15}: {accuracy}")

y_predict = LSTM_Model.predict(X_test)

y_p = [np.argmax(i) for i in y_predict]
y_t = [np.argmax(i) for i in y_test]

cm4 = confusion_matrix(y_t, y_p)
print(cm4)