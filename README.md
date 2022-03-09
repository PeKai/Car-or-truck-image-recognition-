# Introduction to Convelutional Neural Network

Our objective is to figure out how a neural network can interpret a natural image good enough to solve issues similar to those that a human can. The neural networks which exel in this area are called convolutional neural network. The idea of convolutional neural networks are inspired by the human visual cortex. Next I will be taking you through a simple CNN moldel, predicting if an image is a Car or a Truck.

## Preprocessing

<details>
  <summary>Imports</summary>
  
```python
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from tensorflow.keras.layers.experimental import preprocessing
import datetime
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
```
</details>

```python
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
```

To start processing our images into a neural network approved form, we will first create a instance the ImageDataGenerator class. This class will help us generate batches from an image, to use for our model. Here, one of the parameters is called 'rescale'. What this does is it turns all the RGB values of every pixel into a number ranging between 1 and 0.

```python
train_generator = train_datagen.flow_from_directory(
    '/Users/maximilian/Downloads/archive (3)/train',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
    )

test_generator = test_datagen.flow_from_directory(
    '/Users/maximilian/Downloads/archive (3)/valid',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=False,
    )
```

The flow_from_directory() function generates a tf.data.Dataset, from images files in a directory. This function acctually makes it that every image in this directory has a RGB value ranging between 0 and 1. I won't go into detail of every parameter I will only be going through the most imporatant. Firstly, the target_size parameter - this simply resize's the image to a specific amount of pixels by zooming into the picture. This is important, because the neurons can only take a specific amount of pixels. Secondly, the class_mode - this simply is telling the fuction if I have one or more classes to classify. The two main strings herer are 'binray' and 'catagorical'. 'Binary' means the labels will be 1D binary labels, whilst the 'catagorical' string means the labels will be 2D one-hot encoded.

## Model

```python
model = Sequential([

    preprocessing.RandomContrast(0.3),

    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(units=12, activation="relu"),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(units=12, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
])
```

### Data Augmentation

The best way to improve the performance of a any model is to let it learn from more data. One way of getting more data is by transforming the data which you already have. This process is called data augmentation. In my case (as shown below), only the contrast is changed, but you can change it depending on the type of image classification you are performing.

```python
preprocessing.RandomContrast(0.3)
```

For example this image here is from the data set. This image will be augemnetion various times, and the model can than learn frmo 16x more images.

Before data augmentation   |  After data augmentation
:-------------------------:|:-------------------------:
![](https://i.ibb.co/kJM3jzz/Screenshot-2022-03-06-at-16-01-26.png)  |  ![](https://i.imgur.com/UaOm0ms.png)

### Convelution, ReLu and Max Pooling

Now the CNN; The purpose of a CNN is feature extraction (in other words, to extract certain feutures of a image). There are three basic operations in the feature extraction process. 

1. Filter an image for a particular feature (convolution)
2. Detect that feature within the filtered image (ReLU)
3. Condense the image to enhance the features (maximum pooling)

The convolution is layers.Conv2D(), the ReLu is layers.ReLU and the max pooling is layers.MaxPool2D(). The entire process looks similar to this picture.

<div align="center">
  <img src="https://i.imgur.com/IYO9lqp.png"><br><br>
</div>

During the filtering step a kernel scans over over an image. Without going into detail, the kernel creates a new image, which has certain features attached to it. In this case the kernel filtered in horizontal lines of the car. After this process, this convoluted image is then passed onto the ReLU function, which futhremore puts everything unimportant eqivilantly unimporatant (so -1000 and -10 both will be eqivilantly 0). In this case all pixels which are not horizontal lines will be turned to 0 and all pixels which are horizontal will be turned into yellow. Finally, maxpooling will be used to reduce the amount of vairables inside the model, by in simple terms decreasing the quiality of the picture. 

Whilst, thier are alot of Hyperparameters for MaxPooling and ConvNets, the most important hyperparameter is the stride and input_shape. In short, the input shape is the shape inwhich the image is being fed into the network, where the last element is telling us what color-type the image is ([height,widtch,color-type], 3 is RGP, 0 is black and white etc.). Moreover, the stride (which is also mentioned as a hyperparameter in the MaxPooling layer), determines how good the quiality of the image will be, in short if the stride if high important information will get lost. Its recommended to have a stride of 1 during the Convnet and reLU funktion, as compared to the MaxPooling where a high stride is possiable. This is because potential valuble information will get lost during the Convnet and reLU funktion stage, in comparision to the MaxPooling inwhich its porpose is to reduce data points. 

Lastly, the CNN, reLU and MaxPooling are often referred as the base, and following dense layers are referred to the head. Here the idea is that the base does all the feuture extractions, which than gets passed onto the dense layer which than classfies the image. Heres a picture shown below. 

<div align="center">
  <img src="https://i.ibb.co/f2R9HwZ/U0n5xjU.png"><br><br>
</div>


### Special Neurons
#### Dropout
A model can use dropout classes; These are computationally cheap and very effective in reducing overfitting, and essetially 'drops out' out nodes randomly during training. The hyperparameter as shown is the percentage of neurons randomly being dropped out. 

#### BatchNormalization!
Often its preferred to normalize data before it gets fed into the model, mabye normalizing in the middle of the network is better, which is exacly what the BatchNormalization does. 

## Fitting 

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    callbacks=[callbacks=[EarlyStopping(monitor='loss'), TerminateOnNaN()]]
)

```

In the first set of line, we are compiling the code or in other words configuring the model. I wont go into detail about each hyperparameter, however I will talk about the optimizer and loss funktion. A optimizer inside a neural network is the algorithm or method that optimizes the weights such that the neural network can perform. The optimizer.Adam, is just one of many optimizer in the framework, futhremore the loss hyperparameter determines how the optimizer will determine its losses. Often is the case that loss funktions are seperated between catagorial loss funktions and numerical data, in our case since its catagorical data we are using 'binary_crossentropy'. Furthremore, a loss funktion is also called a cost funktion, both words are often intertwined.

Secondly, the fit() funktion will put the neural network into action, meaning it will optimize its weights etc., moreover the important hyperpaparmeter here is the epoch. The epoch is how many iterations it will go through the batch or data for it to corrent its weights. 

Lastly, the accuracy will be printed out in the output, and it will be called 'val_binary_accuracy'.

<!-- ## Optemizing

```python

def fit_with(batch_size, learning_rate, de_1, de_2, dp):
    model = Sequential([

    preprocessing.RandomContrast(0.3),

    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(units=12, activation="relu"),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(units=12, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
    ])
    model.fit(train_generator, validation_data=test_generator, epochs=50, callbacks=[callbacks=[EarlyStopping(monitor='loss'), TerminateOnNaN()]])
    batch_test = test_generator.next()
    prediction = model_1.predict(batch_test)
    acc = mean_absolute_error(y_test, predictions)
    return float(acc)


pbounds = {'batch_size': (10, 150),
           'learning_rate': (0.01, 0.5),
           'de_1': (8, 24),
           'de_2': (8, 24),
           'dp': (0.2, 0.7)}

optimizer = BayesianOptimization(f=fit_with, pbounds=pbounds)

optimizer.maximize(init_points=50, n_iter=150)

```

Here is a optimizing method that I use sometimes, I however won't go into detail.

![BayesianOptimization in action](./bayesian_optimization.gif) -->
