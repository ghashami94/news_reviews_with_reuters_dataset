#import the neccesary libraries

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt






def vectorize_sequences(sequences,dimension=10000):
  results = np.zeros((len(sequences),dimension))
  for i,sequence in enumerate(sequences):
    results[i,sequence] = 1
  return results



 
#load the data and split the data the training set and test set
#in this loading num_words: just keep the top 10000 most frequently occuring words is the traing data
                          ## in this dataset make a dictionary that avery 10.000 word has an index

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = 10000)  

#preprocessing



x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



#define the model

movie_review_model = models.Sequential()
movie_review_model.add(layers.Dense(16,activation = 'relu',input_shape=(10000,)))
movie_review_model.add(layers.Dense(16,activation = 'relu'))
movie_review_model.add(layers.Dense(1,activation = 'sigmoid'))

#comile the model
#movie_review_model.compile(optimizer=optimizers.RMSprop(lr=0.001),
 #                          loss=losses.binary_crossentropy,
  #                         metrics=[metrics.binary_accuracy])
movie_review_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#validataion data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


#train the model

#movie_review_model.fit(x_train,y_train, epochs =20, batch_size=512)
#results = movie_review_model.evaluate(x_test,y_test)


history= movie_review_model.fit(partial_x_train,partial_y_train,epochs=12,batch_size = 512,validation_data = (x_val,y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title("Trainig and Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs,acc_values,'bo',label='Training acc')
plt.plot(epochs,val_acc_values,'b',label='Validation acc')
plt.title("Trainig and Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()




