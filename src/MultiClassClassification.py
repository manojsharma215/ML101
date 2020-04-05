import tensorflow as tf
import tensorflow.keras.datasets as tfds
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tfds.fashion_mnist.load_data();

x_train , x_test = x_train/255.0, x_test/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.title("Training  Accuracy")

plt.xlabel("Epochs")
plt.ylabel("accuracy")

plt.plot(epochs, accuracy)
plt.show()





