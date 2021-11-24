# First we want to import the libraries we need
# this is TensorFlow and the Keras libs from tensorflow for machine learning
# and some usefull libs like numpy (handling data) and matplotlib (print graphs)

import tensorflow as tf
import numpy as np

# Main function
# In PyCharm you can execute the code, by clicking on the run button at the top
if __name__ == '__main__':

    #########################
    # Step 1: load the data #
    #########################

    # let's load the fashion_mnist dataset, which is included in the tensorflow sample datasets
    # so no need to download it manually from the internet

    fashion_mnist_full_dataset = tf.keras.datasets.fashion_mnist

    # let's split the data into a training and a testset
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist_full_dataset.load_data()

    # let's define the output classes
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    ############################
    # Step 2: prepare the data #
    ############################
    # the pixels have values between 0 and 255, which is normal for a grey scaled image, but
    # the algorithm works better if the data is scaled between 0 and 1, so lets divide all values by 255
    # and then print the first 16 in order to check whether we' done it right
    # the Conv2D is designed for 3 dimensions, that why we add 1 as third dimensions
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    ###########################
    # Step 3: build the model #
    ###########################
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(training_images, training_labels, epochs=50)
    # -> this will lead to an accuracy of our training set of about 91%

    ##############################################
    # Step 4: calculate the quality of the model #
    ##############################################
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\naccuracy of our model against our test set:', test_acc)
    # -> accuracy of our model against our test set: 0.8813999891281128
    # so our accuracy of the model against the test set was 91%, but it is only 88% against the test set
    # So we have a case of overfitting
    # we accept this in our sample

    ########################################
    # Step 5: let's predict a sample image #
    ########################################
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print('\nshow probabilities for different classes:', predictions[0])
    print('\nthe most likely class for our sample is:', np.argmax(predictions[5]))
    print('\nthe most likely class name for our sample is:', class_names[np.argmax(predictions[5]) + 1])