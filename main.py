import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train = x_train / 255.0
x_test = x_test / 255.0

plt.imshow(x_train[0, :, :, ])
print()
pylab.show()


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(49, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])


def train(x, y):
    model = build_model()
    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    model.evaluate(x_test, y_test)

    # predict
    # print(np.argmax(model.predict(x_test[0])))
    print(model.predict(x_test)[0])

    return model


model = train(x_train, y_train)
