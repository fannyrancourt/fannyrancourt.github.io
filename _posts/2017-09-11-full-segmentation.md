---
layout: post
title: A full-fledge segmentation pipeline using Keras and Keras-Transform
---

Here's a basic pipeline which handles data augmentation and allows you to quickly start training. In this exemple, `create_model` and `get_paths` should be created by you.

## Dependencies
You will need [Keras](https://github.com/fchollet/keras), [Keras-Transform](https://github.com/Dref360/keras-transform) and *OpenCV*.

The quickest possible way to install OpenCV is through *conda* :

`conda install -c menpo opencv3`

## Building a Sequence

Let's begin by importing *Keras*, *Keras-Transform* and *OpenCV*. We'll design our *Sequence* object in the most simple way. It will be provided by a list of paths to the input image and its segmentation mask. The *Sequence* will load the image and resize it.

```python
import cv2
import numpy as np
from keras.utils import Sequence
from model import create_model  # Function to create a model which does segmentation (U-Net, Tiramisu, etc)
from transform.sequences import (RandomRotationTransformer, SequentialTransformer,
                                 RandomZoomTransformer, RandomHorizontalFlipTransformer)
from your_dataset import get_paths  # Method which will return the path to both the image and its segmentation mask

INPUT_SHAPE = (300, 300)
BATCH_SIZE = 10


class YourDatasetSequence(Sequence):
    def __init__(self, paths: [(str, str)], input_shape: (int, int), batch_size: int):
        self.paths = paths
        self.input_shape = input_shape  # We assume that the input_shape is the same as the output
        self.batch_size = batch_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        data = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, mask = zip(*data)  # Split the data between input and groundtruth

        # Load and resize the images, we also apply some preprocessing.
        X = [cv2.resize(cv2.imread(fi), self.input_shape) for fi in X]
        X = [self.apply_preprocessing(x) for x in X]

        # Load the masks
        mask = np.array([cv2.resize(cv2.imread(fi), self.input_shape) for fi in mask])

        return X, mask

    def apply_preprocessing(self,x: np.array):
      """Dummy function that puts `x` between 0,1."""
      return x / 255.
```

## Using the Sequence

With this simple *Sequence* object, you are now able to data augmentation and multiprocessing loading. Now we need to instanciate those *Sequences* and apply the data augmentation to the training *Sequence*. Using *Keras-Transform*, we will be able to apply random transformations on both the input and the mask.

```python
def get_sequences(paths, batch_size=1):
    """Will create the sequences from `paths`"""
    # We split `paths` in 3 and then create 3 sequences.
    train, val, test = np.split(paths, [0.6 * len(paths), 0.8 * len(paths)])
    return YourDatasetSequence(train, INPUT_SHAPE, batch_size), \
           YourDatasetSequence(val, INPUT_SHAPE, batch_size), \
           YourDatasetSequence(test, INPUT_SHAPE, batch_size)

# We create our three Sequences
paths = get_paths()  # [(str,str)]
train_seq, val_seq, test_seq = get_sequences(paths, BATCH_SIZE)

# We create a sequence of transformer to do DA
sequence = SequentialTransformer([RandomZoomTransformer(zoom_range=(0.8, 1.2)),
                                  RandomHorizontalFlipTransformer()])

# By supplying a mask, we can perform transformation on the groundtruth as well.
train_seq = sequence(train_seq, mask=[True, True])

```

Now that we have everything in place, we only need to instanciate the model and train it!

```python
# Create the model
model = create_model()  # Model

# We can now train and evaluate!
model.fit_generator(train_seq, len(train_seq), validation_data=val_seq, validation_steps=len(val_seq),
                    use_multiprocessing=True, workers=6, epochs=15)

model.evaluate_generator(test_seq, len(test_seq), use_multiprocessing=True, workers=6)

```


# Conclusion
In this post, we've created a pipeline for segmentation using *Keras* and *Keras-Transform*. With *Sequences*, we can safely train our model using multiprocessing. If you have any question or comment, feel free to e-mail me!
