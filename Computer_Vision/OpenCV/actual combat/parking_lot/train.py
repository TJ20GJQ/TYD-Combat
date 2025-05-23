import numpy
import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.initializers import TruncatedNormal
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


files_train = 0
files_validation = 0

cwd = os.getcwd()
folder = 'train_data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_train += len(files)


folder = 'train_data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_validation += len(files)

print(files_train, files_validation)

img_width, img_height = 48, 48
train_data_dir = "train_data/train"
validation_data_dir = "train_data/test"
batch_size = 32
nb_train_steps = files_train // batch_size
nb_validation_steps = files_validation // batch_size
epochs = 15
num_classes = 2

model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in model.layers[:10]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model_final = Model(inputs=model.input, outputs=predictions)

model_final.compile(loss="categorical_crossentropy", 
                    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9), 
                    metrics=["accuracy"])

train_datagen = ImageDataGenerator(
rescale=1./255,
horizontal_flip=True,
fill_mode="nearest",
zoom_range=0.1,
width_shift_range=0.1,
height_shift_range=0.1,
rotation_range=5)

test_datagen = ImageDataGenerator(
rescale=1./255,
horizontal_flip=True,
fill_mode="nearest",
zoom_range=0.1,
width_shift_range=0.1,
height_shift_range=0.1,
rotation_range=5)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode="categorical")

checkpoint = ModelCheckpoint("car1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch")
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

history_object = model_final.fit(
train_generator,
steps_per_epoch=nb_train_steps,
epochs=epochs,
validation_data=validation_generator,
validation_steps=nb_validation_steps,
callbacks=[checkpoint, early])
