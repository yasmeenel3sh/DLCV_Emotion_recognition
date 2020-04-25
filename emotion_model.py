import keras
import numpy as np
import glob
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

num_classes = 3
rows, cols = 48, 48
batch_size = 32

train_data = []
labels = []
for i, emotion in enumerate(['anger','sadness', 'happy']):
    filelist = glob.glob(emotion + '/*.png')
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    if(len(train_data) > 0):
        train_data = np.append(train_data, x, axis=0)
    else:
        train_data = x
    labels = np.append(labels, np.ones(x.shape[0]) * i)
train_data = train_data.reshape(train_data.shape[0],rows,cols,1)

x_train, x_valid, y_train, y_valid = train_test_split(train_data, labels, test_size=0.33, shuffle= True)

model = Sequential()

# Block 1
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(rows,cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(rows,cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),2))
model.add(Dropout(0.4))

# Block 5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(Dropout(0.5))

# Block 7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

epochs = 30

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_one_hot_labels = keras.utils.to_categorical(y_train, num_classes=num_classes)
valid_one_hot_labels = keras.utils.to_categorical(y_valid, num_classes=num_classes)

history = model.fit(x_train, train_one_hot_labels, epochs=epochs, batch_size=batch_size, validation_data=(x_valid,valid_one_hot_labels))

test = np.array(Image.open('2.jpg'))
test_set = []
test_set.append(test)
test_set.append(test)
test_set = np.array(test_set)
m = model.predict(test_set.reshape(test_set.shape[0],48,48,1))
print(m)
print(np.where(m==np.max(m)))