import tensorflow as tf
import tensorflow_datasets as tfds


ds_train: tf.data.Dataset
ds_test: tf.data.Dataset

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",  # The name of the dataset
    split=["train", "test"],  # split data for training and testing
    shuffle_files=True,  # shuffle the data
    as_supervised=True,  # return the data as a tuple of images and labels
    with_info=True,  # print info about the dataset
)

# dataset gives uint8, we want to convert to float32 for the model
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


# cast images to float32
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# cache dataset for better performance
ds_train = ds_train.cache()
# after caching, shuffle the dataset
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# group the dataset into batches of 128 at each epoch
ds_train = ds_train.batch(128)
# enable prefetching for performance, (read next batch one step ahead)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# same steps for the test datasetm, but no need to shuffle
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)  # same batch size
ds_test = (
    ds_test.cache()
)  # cache after batching because they can be the same for eacho epoch
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


model = tf.keras.models.Sequential(  # create a sequential model
    [
        tf.keras.layers.Flatten(
            input_shape=(28, 28)
        ),  # flatten images into one dimension
        tf.keras.layers.Dense(
            128, activation="relu"
        ),  # fully connected layer with 128 neurons
        tf.keras.layers.Dense(10),  # output layer with 10 neurons (one for each digit)
    ]
)
model.compile(  # compile the model
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # use crossentropy because there is >=2 classes
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
