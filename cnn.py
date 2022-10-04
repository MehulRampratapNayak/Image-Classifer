
import tensorflow as tf
from tensorflow import keras

# Initialising the CNN
classifier = tf.keras.models.Sequential()

# Step 1 - Convolution
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(tf.keras.layers.Flatten())

# Step 4 - Full connection
classifier.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("C:\\Mehul\\Data Scientist\\Computer Vision\\dogcat_new\\cats_and_dogs_filtered\\train",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory("C:\\Mehul\\Data Scientist\\Computer Vision\\dogcat_new\\cats_and_dogs_filtered\\validation",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = classifier.fit(training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set,    
                         validation_steps = 2000)

classifier.save("model.h5")
print("Saved model to disk")

# Part 3 - Making new predictions
import numpy as np
test_image = tf.keras.utils.load_img("C:\\Users\\mehul\\Downloads\\depositphotos_4869272-stock-photo-bengal-cat-light-brown-cream.jpg", target_size = (64, 64))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)

