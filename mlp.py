import tensorflow as tf
from tensorflow.keras import Sequential, layers

x = tf.random.normal([2, 3])

model = Sequential([
    layers.Dense(2, activation='relu'),
    layers.Dense(2, activation='relu'),
    layers.Dense(2)
])

model.build(input_shape=[None, 3])
model.summary()

for p in model.trainable_variables:
    print(p.name, p.shape)
