
# coding: utf-8

# In[2]:

import csv
import cv2
import numpy as np

#Funtion to 
#1.add the images and the steering measurements from the csv file
#2. Flip the images

def gatherRecords(dir,images,measurements):
    lines=[]
    csvPath = dir + '/driving_log.csv'
    with open(csvPath) as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    
    for line in lines:
        source_path=line[0]
        filename=source_path.split('/')[-1]
        current_path= dir + '/IMG/' + filename
        image=cv2.imread(current_path)
        images.append(image)
        measurement=float(line[3])
        measurements.append(measurement)
       ''' 
        #Left Image
        source_path=line[1]
        filename=source_path.split('/')[-1]
        current_path= dir + '/IMG/' + filename
        left_image=cv2.imread(current_path)
        
        #Right Image
        source_path=line[2]
        filename=source_path.split('/')[-1]
        current_path= dir + '/IMG/' + filename
        right_image=cv2.imread(current_path)
        
        #Add Left and Right Images
        correction=0.2
        images.append(left_image)
        measurements.append(measurement+correction)
        images.append(right_image)
        measurements.append(measurement-correction)
      ''' 
        #Flipped Images
        image = np.fliplr(image)
        measurement = -measurement
        images.append(image)
        measurements.append(measurement)
        
    return images,measurements

images=[]
measurements=[]
images,measurements=gatherRecords('data',images,measurements)    
#images,measurements=gatherRecords('Training_Data1',images,measurements)
#images,measurements=gatherRecords('Backward_Data',images,measurements)

X_train=np.array(images)
y_train=np.array(measurements)

# The architecture

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D

model=Sequential()

#Add normalization to the model

model.add(Lambda(lambda x: x/250.0 - 0.5,input_shape=(160,320,3)))

#Add 5 convolutional layers to the model

model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), str `ides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#Add 4 fully connected layers

model.add(Flatten())

model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

#See the summary of the architecture
model.summary()

# Add mse as the loss function
model.compile(loss='mse',optimizer='adam')
#Shuffle the training data and split it with 20% of training data into validation set and run for 5 epochs
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

#Save the model
model.save('model.h5')


# In[ ]:



