import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', input_shape=([1]))
])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([5.0, 10.0, 15.0, 20.0, 25.0], dtype=float)

history = model.fit(xs, ys, epochs=500, verbose=1)

loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.title("Linear Regression Analysis")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epochs, loss)
plt.show()


print(model.predict([7.0]))