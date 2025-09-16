# -*- coding: utf-8 -*-

TRAIN_PATH = "CovidDataset/train"
VAL_PATH = "CovidDataset/val"

#import numpy as np
#import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
import pickle

# CNN Based Model in Keras

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

model.summary()

# Train from scratch

#to prepare the data for the model - image augmentation and rescale,
#shear and zoom augmentation
#vertical flip not doing coz we should not invert the xray

train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)

test_dataset = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'CovidDataset/train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')

train_generator.class_indices

validation_generator = test_dataset.flow_from_directory(
    'CovidDataset/val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps=2
)

model = hist.model

"""
model.save(
    model,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)
"""

#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))

#model.save_weights("model.h5")
model.save("model.h5")
print("Saved model to disk")

keras.models.save_model(model,'my_model.hdf5')
print("Saved model to disk as HDF5")


"""
Model.save(
    model.h5,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)




y_pred = model.predict(VAL_PATH, batch_size=batch_size)
prediction=y_pred[0:10]
for index, probability in enumerate(prediction):
  if probability[1] > 0.5:
    plt.title('%.2f' % (probability[1]*100) + '% COVID')
  else:
    plt.title('%.2f' % ((1-probability[1])*100) + '% NonCOVID')
  plt.imshow(X_test[index])
  plt.show()
"""