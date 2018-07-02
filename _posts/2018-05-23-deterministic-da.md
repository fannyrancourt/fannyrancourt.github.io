---
layout: post
title:  Deterministic data augmentation - Keras 2.2.0
---

In a previous [post](/full-segmentation), I showed how to use [Keras-Transform](https://github.com/Dref360/keras-transform), a library I created to perform data augmentation on segmentation datasets. After discussion with [Francois Chollet](https://github.com/fchollet), I think that his idea is much better and it's been recently merged on Keras. I was excited to try it out!

The code is available in this [gist](https://gist.github.com/Dref360/d4eafccdd1c9d6e87764c43da50ffb19).

### High-level

Instead of building a pipeline of *Sequences* (like in Keras-Transform), the data augmentation will be done inside one. So we should get something like this :

```python
class MySequence(keras.utils.Sequence):
  def __init__(self):
    ...
  def __getitem__(self, idx):
    X, y = ... # Get the inputs
    for i in len(X):
      rotation = get_random_rotation()
      # Apply the same transformation to X and y
      X[i] = rotate_image(X[i], rotation)
      y[i] = rotate_image(y[i]. rotation)
    return X, y
```


### In Keras 2.1.7

Thanks to [vkk800](https://github.com/vkk800), this API has made it into Keras and will be released soon. Here's a **simple** example.

---
We first import OpenCV and updated ImageDataGenerator
```python
import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
```
---

We only need to inherit a Sequence. Since this is an example, this will only return a batch of cats.

```python
class MySequence(Sequence):
    def __init__(self):
        self.path = '/home/fred/Images/cat.jpg'
        self.imgaug = ImageDataGenerator(rotation_range=20,
                                         rescale=1/255.,
                                         width_shift_range=10)

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        X = np.array([cv2.resize(cv2.imread(self.path), (100, 100)) for _ in range(10)]).astype(np.float32)  # Fake batch of cats
        y = np.copy(X)
        for i in range(len(X)):
            # This creates a dictionary with the params
            params = self.imgaug.get_random_transform(X[i].shape)
            # We can now deterministicly augment all the images
            X[i] = self.imgaug.apply_transform(self.imgaug.standardize(X[i]), params)
            y[i] = self.imgaug.apply_transform(self.imgaug.standardize(y[i]), params)
        return X, y
```

### Results

Using this `Sequence`, we can get the following results. (`X` is on the top row and `y` on the second)

![](/images/cat_aug.jpg.png "Results")


Making this work for a real-life application is really easy!
For example, we can use the `params` variable in other situations like bounding boxes augmentation.
The code is available in this [gist](https://gist.github.com/Dref360/d4eafccdd1c9d6e87764c43da50ffb19).

Thank you for reading and I'm always available on Keras' Slack (@Dref360) if you have any question!

Cheers,
Frédéric Branchaud-Charron
