import sys
from keras_preprocessing import image
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
from keras.preprocessing.image import load_img
from PIL import Image
import os
import glob
from PIL import Image
import pandas as pd

image_size = (300, 300)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

#graphing dataset to make sure it imported correctly
import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")


train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # no image augmentation, we have tons of images
    x = inputs

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)



# old code, found better way to import/do everything:

# image_list = []
# for filename in glob.glob('archive/chest_xray/train/NORMAL/*'): 
#     im=load_img(filename).resize((300,300), resample = 0).convert('L')
#     image_list.append(im)    

# sickimage_list = []
# for filename in glob.glob('archive/chest_xray/train/PNEUMONIA/*'): 
#     im=load_img(filename).resize((300,300), resample = 0).convert('L')
#     sickimage_list.append(im)    


# image_list_test = []
# for filename in glob.glob('archive/chest_xray/test/NORMAL/*'): 
#     im=load_img(filename).resize((300,300), resample = 0).convert('L')
#     image_list_test.append(im)    

# sickimage_list_test = []
# for filename in glob.glob('archive/chest_xray/test/PNEUMONIA/*'): 
#     im=load_img(filename).resize((300,300), resample = 0).convert('L')
#     sickimage_list_test.append(im)   

# image_tuple_list = []
# for elt in image_list:
#     image_tuple_list.append((elt, 0))

# sickimage_tuple_list = []
# for elt in sickimage_list:
#     sickimage_tuple_list.append((elt, 1))

# image_tuple_list_test = []
# for elt in image_list_test:
#     image_tuple_list.append((elt, 0))

# sickimage_tuple_list_test = []
# for elt in sickimage_list_test:
#     sickimage_tuple_list_test.append((elt, 1))

# train = image_tuple_list.append(sickimage_tuple_list)

# test = image_tuple_list_test.append(sickimage_tuple_list_test)

# train = tf.data.Dataset.from_tensor_slices(train)
# test = tf.data.Dataset.from_tensor_slices(test)

# (image_list.pop(0)).show() # do not delete

