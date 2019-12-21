#import library
import numpy
import cv2
import glob
import os
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#%%preprocessing data
def preprocessing(path, paths):
    for bb,file in enumerate (glob.glob(path)):
        images = cv2.imread(file)
        resized_image = cv2.resize(images, (512, 512))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        y = 100
        x = 0
        h = 256
        w = 256
        crop = gray_image[y:y+h, x:x+w]
        cv2.imwrite(paths.format(bb), crop)
    
preprocessing(r'D:\data\1000\*.jpg', r'D:\data\preprocessing\input_data_resize\001000_{}.jpg')
preprocessing(r'D:\data\2000\*.jpg', r'D:\data\preprocessing\input_data_resize\002000_{}.jpg')
preprocessing(r'D:\data\5000\*.jpg', r'D:\data\preprocessing\input_data_resize\005000_{}.jpg')
preprocessing(r'D:\data\10000\*.jpg', r'D:\data\preprocessing\input_data_resize\010000_{}.jpg')
preprocessing(r'D:\data\20000\*.jpg', r'D:\data\preprocessing\input_data_resize\020000_{}.jpg')
preprocessing(r'D:\data\50000\*.jpg', r'D:\data\preprocessing\input_data_resize\050000_{}.jpg')
preprocessing(r'D:\data\100000\*.jpg', r'D:\data\preprocessing\input_data_resize\100000_{}.jpg')
preprocessing(r'D:\data\palsu\*.jpg', r'D:\data\preprocessing\input_data_resize\palsu_{}.jpg')

#%%make array from data
path = r'D:\data\preprocessing\input_data_resize'
listing = os.listdir(path) 
num_samples = len(listing)
print (num_samples)

#%%create matrix to store all flattened images
immatrix = numpy.array([numpy.array(Image.open(r'D:\data\preprocessing\input_data_resize'+ '\\' + im2)).flatten()
            for im2 in listing],'f')
    
label = numpy.ones((num_samples,),dtype = int)
label[0:100] = 0
label[100:200] = 1
label[200:300] = 2
label[300:400] = 3
label[400:500] = 4
label[500:600] = 5
label[600:700] = 6
label[700:] = 7

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

#%%parameter for CNN
bs = input("Batch size : ")
batch_size = int(bs)
nbf = input("Number of filters : ")
nb_filters = int(nbf)
nbe = input("Number of epoch : ")
nb_epoch = int(nbe)
nbc1 = input("Number of convolution 1 : ")
nb_conv1 = int(nbc1)
nbc2 = input("Number of convolution 2 : ")
nb_conv2 = int(nbc2)
nbst = input("Number of strides : ")
nb_strides = int(nbst)
nbp = input("number of pool : ")
nb_pool = int(nbp)
nbt = input("Number of split between training and testing : ")
test_split = float(nbt)
nbs = input("Number of split between training and validation : ")
val_split = float(nbs)
nb_classes = 8
img_channels = 1

#%%split data test and data train
(X,y) = (train_data[0], train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split, random_state = 4)

X_train = X_train.reshape(X_train.shape[0], 1, 256, 256)
X_test = X_test.reshape(X_test.shape[0], 1, 256, 256)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#%%convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#%%building CNN
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv1, strides = nb_strides,
    input_shape=(1, 256, 256), data_format = 'channels_first'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Convolution2D(nb_filters, nb_conv2, nb_strides))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
#%%train cnn
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    verbose=1, validation_split = val_split)

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(nb_epoch)

fname = "weights.hdf5"
model.save_weights(fname, overwrite = True)

#%%validate train
model.load_weights(fname)

y_pred1 = model.predict_classes(X_train)
print(y_pred1)

p1 = model.predict_proba(X_train) # to predict probability
print(p1)

target_names = ['class 0(1000)', 'class 1(2000)', 'class 2(5000)', 'class 3(10000)', 'class 4(20000)', 'class 5(50000)', 'class 6(100000)', 'class 7(palsu)']
print(confusion_matrix(numpy.argmax(Y_train,axis=1), y_pred1))
print(classification_report(numpy.argmax(Y_train,axis=1), y_pred1, target_names=target_names))

score1 = model.evaluate(X_train, Y_train, verbose = 0)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])
print(model.predict_classes(X_train[0:10]))
print(Y_train[0:10])

#%%validate test
model.load_weights(fname)

y_pred2 = model.predict_classes(X_test)
print(y_pred2)

p2 = model.predict_proba(X_test) # to predict probability
print(p2)

print(confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred2))
print(classification_report(numpy.argmax(Y_test,axis=1), y_pred2, target_names=target_names))

score2 = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])
print(model.predict_classes(X_test[0:10]))
print(Y_test[0:10])