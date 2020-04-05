import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', input_shape=([1]))
])

model.compile(optimizer='sgd', loss='mean_squared_error', metrics='acc')

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([5.0, 10.0, 15.0, 20.0, 25.0], dtype=float)

model.fit(xs, ys, epochs=500, verbose=1)

print(model.predict([7.0]))