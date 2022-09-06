import numpy as np
import tensorflow as tf

path = '/root/.keras/datasets/mnist.npz'
with np.load(path) as np_data:
    train_exa = np_data['x_train']
    train_labels = np_data['y_train']
    test_exa = np_data['x_test']
    test_labels = np_data['y_test']

train_dataset = tf.data.Dataset.from_tensor_slices((train_exa, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_exa, test_labels))

print(train_dataset, test_dataset)

train_dataset = train_dataset.shuffle(128).batch(64)
test_dataset = test_dataset.batch(64)

print(train_dataset, test_dataset)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy'])

model.fit(train_dataset, epochs=20)

model.evaluate(test_dataset)


