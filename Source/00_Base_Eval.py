# %% md
# Preparation
# %%
seed_value = 42

import random

random.seed(seed_value)

import numpy as np

np.random.seed(seed_value)

import tensorflow as tf

tf.random.set_seed(seed_value)
tf.keras.utils.set_random_seed(seed_value)
# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
# %%
from tensorflow import keras
from tensorflow.keras import layers

# %%
num_epochs = 100
batch_size = 16
num_classes = 10
shape = (28, 28, 1)
lr = 0.0003
opt = keras.optimizers.Adamax(learning_rate=lr)
los = keras.losses.CategoricalCrossentropy()
mtr = ["accuracy"]


# %% md
# Dataset
# %%
def prepare_data(main_path, validation_split=0.1):
    with np.load(main_path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']

    # Normalize and reshape the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Split the training data into train and validation sets
    val_size = int(len(x_train) * validation_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Print the shapes
    print(f"Shape of training images: {x_train.shape}")
    print(f"Shape of validation images: {x_val.shape}")
    print(f"Shape of testing images: {x_test.shape}")

    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = prepare_data(main_path="../Dataset/mnist.npz")
# %% md
# Model
# %%
model = keras.Sequential([keras.layers.InputLayer(shape),
                          layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
                          layers.LeakyReLU(alpha=0.2),
                          layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                          layers.LeakyReLU(alpha=0.2),
                          layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                          layers.LeakyReLU(alpha=0.2),
                          layers.GlobalMaxPooling2D(),
                          layers.Dense(64, activation="relu"),
                          layers.Dense(num_classes, activation="softmax")],
                         name="discriminator")

model.compile(optimizer=opt, loss=los, metrics=mtr)
model.summary()


# %% md
# Train
# %%
def callback():
    main_chk = keras.callbacks.ModelCheckpoint(filepath="Checkpoints/Zero", monitor='val_loss', mode='min', verbose=0,
                                               save_best_only=True)
    early_st = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=0)
    rduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=5, verbose=1,
                                                 min_lr=0.00001)

    return [main_chk, early_st, rduce_lr]


model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, batch_size=batch_size, callbacks=callback())
# %% md
# Evaluation
# %%
test_model = tf.keras.models.load_model("Checkpoints/Zero")
test_model.evaluate(test_dataset, verbose=1, batch_size=batch_size)