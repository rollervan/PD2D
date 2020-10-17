import tensorflow as tf
import tensorflow_datasets as tfds
import datasets.brats_ds
from tensorflow.keras import backend as K
from utils import PD_v2
from PD_v2 import PDNN
def normalize_img(x):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(x['image'], tf.float32) / 255., tf.cast(x['mask'], tf.float32) / 255.


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


ds = tfds.load('brats_ds', download=True)

(ds_train, ds_test), ds_info = tfds.load(
    'brats_ds',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)

SHUFFLE_SIZE = 1000
BATCH_SIZE = 8
# Train
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.shuffle(SHUFFLE_SIZE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Create Model and Fit
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(240, 240, 3)),
    tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same'),
])

# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc', f1_m, precision_m, recall_m])

history = model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(ds_test, verbose=0)
