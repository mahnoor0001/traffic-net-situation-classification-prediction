#mount drive and load csv data

from google.colab import drive

drive.mount('/content/drive') 



train_dir="/content/drive/MyDrive/trafficnet_dataset_v1/trafficnet_dataset_v1/train"



import tensorflow as tf



data_dir = "/content/drive/MyDrive/trafficnet_dataset_v1/trafficnet_dataset_v1"                                          







test_dir="/content/drive/MyDrive/trafficnet_dataset_v1/trafficnet_dataset_v1/test"



from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf



train_datagen = ImageDataGenerator(rescale = 1./255)#initialize train generator 

                                 

test_datagen = ImageDataGenerator(rescale = 1.0/255.) #initialize validation generator





train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128,128),batch_size=32,class_mode='categorical')



#validation_generator = valid_datagen.flow_from_directory(validation_ds, target_size=(128,128),batch_size=32,class_mode='categorical')



test_generator = test_datagen.flow_from_directory(test_dir, target_size=(128,128),batch_size=32,class_mode='categorical')



img=train_generator[0]

img



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Softmax

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

import pandas as pd



seq_model = Sequential([ 

    Flatten(input_shape=(128,128,3), name='input_layer'),

    Dense(64, activation='relu', name='layer1'),

    # Dense(64, activation='relu', name='layer2'),

    Dense(32, activation='relu', name='layer3'),

    # Dense(32, activation='relu', name='layer4'),

    Dense(4, activation='softmax', name='output_layer')

])



seq_model.summary()



type(data_dir)



seq_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])  



har=seq_model.fit(train_generator,validation_data=test_generator,epochs=30, batch_size=32)



import numpy as np 



from keras.preprocessing.image import ImageDataGenerator

import os



from tensorflow.keras.preprocessing.image import ImageDataGenerator

train=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test=ImageDataGenerator(rescale=1./255)



import pandas as pd



df = pd.DataFrame(har.history)

df.head()



import matplotlib.pyplot as plt



# summarize history for accuracy

plt.plot(har.history['accuracy'])

plt.plot(har.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(har.history['loss'])

plt.plot(har.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



opt = tf.keras.optimizers.SGD(learning_rate=0.001)

acc = tf.keras.metrics.SparseCategoricalAccuracy()

los = tf.keras.losses.SparseCategoricalCrossentropy()





seq_model.compile(

    

    optimizer = opt,

    loss = los,

    metrics=acc

)



loss_plot = df.plot(y='loss', title='Loss vs Epochs', legend=False)

loss_plot.set(xlabel='Epochs', ylabel='Loss')



acc_plot = df.plot(y='accuracy', title='Accuracy vs Epochs', legend=False)

acc_plot.set(xlabel='Epochs', ylabel='Accuracy')





import cv2



t=cv2.imread('/content/drive/MyDrive/trafficnet_dataset_v1/trafficnet_dataset_v1/test/accident/images_016.jpg')

plt.imshow(t)



labels = ['accident', 'dense_traffic','fire','sparse_traffic']

img_size = 224

def get_data(data_dir):

    data = [] 

    for label in labels: 

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        for img in os.listdir(path):

            try:

                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format

                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size

                data.append([resized_arr, class_num])

            except Exception as e:

                print(e)

    return np.array(data)





import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

testimg=cv2.resize(t,(128,128))

testimg=img_to_array(testimg)/255

h=np.expand_dims(testimg,axis=0)

r=seq_model.predict(h)

classnames=["fire","sparse_traffic","dense_traffic","accident"]

ypred=classnames[np.argmax(r)]

ypred



from tensorflow.keras.preprocessing.image import ImageDataGenerator

train=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test=ImageDataGenerator(rescale=1./255)



t=cv2.imread('/content/drive/MyDrive/trafficnet_dataset_v1/trafficnet_dataset_v1/test/sparse_traffic/images_007 (2).jpg')

plt.imshow(t)





import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

testimg=cv2.resize(t,(128,128))

testimg=img_to_array(testimg)/255

h=np.expand_dims(testimg,axis=0)

r=seq_model.predict(h)

classnames=["fire","sparse_traffic","dense_traffic","accident"]

ypred=classnames[np.argmax(r)]

ypred



t=cv2.imread('/content/drive/MyDrive/trafficnet_dataset_v1/trafficnet_dataset_v1/test/dense_traffic/images_011.jpg')

plt.imshow(t)



import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

testimg=cv2.resize(t,(128,128))

testimg=img_to_array(testimg)/255

h=np.expand_dims(testimg,axis=0)

r=seq_model.predict(h)

classnames=["fire","sparse_traffic","dense_traffic","accident"]

ypred=classnames[np.argmax(r)]

ypred
